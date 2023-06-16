module RandomElements

export IndependentRandomElement, TimeSeries, IID, Time

using Distributions
using Random

Base.eltype(::Type{<:MvNormal{T}}) where {T <: Integer} = Float64

abstract type AbstractRandomElement{T}
end

Base.eltype( ::Type{U} ) where {T, U <: AbstractRandomElement{T}} = T

mutable struct IndependentRandomElement{T,D <: Distribution} <: AbstractRandomElement{T}
    dist::D
end

# make these Float64 whether they are continuous or not because 
IndependentRandomElement( dist::D ) where {D <: Distribution{Univariate}} =
    IndependentRandomElement{Float64,D}( dist )

IndependentRandomElement( dist::D ) where {D <: Distribution{Multivariate}} =
    IndependentRandomElement{Vector{Float64},D}( dist )

struct TransformedRandomElement{Op,T,U <: AbstractRandomElement{T}} <: AbstractRandomElement{T}
    # for now, all elements must be of the same type
    args::Vector{U}
end

for op in [:+,:*,:/,:-]
    e1 = :( Base.$op( x::Number, y::AbstractRandomElement ) = $op( promote( x, y )... ) )
    eval( e1 )

    e2 = :( Base.$op( x::AbstractRandomElement, y::Number ) = $op( promote( x, y )... ) )
    eval( e2 )
    
    e3 = quote
        Base.$op( x::AbstractRandomElement{T}, y::AbstractRandomElement{T} ) where T =
            TransformedRandomElement{$op, T, AbstractRandomElement{T}}( AbstractRandomElement{T}[x,y] )
    end
    eval( e3 )
end

Base.promote_rule( ::Type{T}, ::Type{V} ) where {T <: Number, U <: Number, V <: AbstractRandomElement{U}} =
    AbstractRandomElement{promote_type(T,U)}

Base.promote_rule( ::Type{U}, ::Type{W} ) where {T <: Number, U <: AbstractRandomElement{T}, V <: Number, W <: AbstractRandomElement{V}} =
    AbstractRandomElement{promote_type(T,V)}

Base.convert( ::Type{AbstractRandomElement{T}}, x::U ) where {T <: Number, U <: Number} =
    IndependentRandomElement(Dirac(convert(T, x)))

mutable struct Node{T}
    calculation::Function
    dependencies::Vector{Node}
    cache::Vector{T}
    visited::Bool
    calculated::Bool
end

Node( calculation::Function, dependencies::Vector{Node}, T ) = Node( calculation, dependencies, T[], false, false )

struct Dependencies
    re2sym::Dict{AbstractRandomElement, Symbol}
    sym2node::Dict{Symbol, Node}
end

Dependencies() = Dependencies( Dict{AbstractRandomElement, Symbol}(), Dict{Symbol, Node}() )

Base.length( dependencies::Dependencies ) = length(dependencies.re2sym )
Base.haskey( dependencies::Dependencies, re::AbstractRandomElement ) = haskey( dependencies.re2sym, re )
Base.getindex( dependencies::Dependencies, re::AbstractRandomElement ) = dependencies.re2sym[re]
Base.getindex( dependencies::Dependencies, sym::Symbol ) = dependencies.sym2node[sym]
symbols( dependencies::Dependencies ) = collect(keys(dependencies.sym2node))
nodes( dependencies::Dependencies ) = collect(values(dependencies.sym2node))

function Base.setindex!( dependencies::Dependencies, sym::Symbol, re::AbstractRandomElement{T}, node::Node{T} ) where T
    dependencies.re2sym[re] = sym
    dependencies.sym2node[sym] = node
end

function rand_expr( re::IndependentRandomElement{T}, dependencies::Dependencies ) where {T}
    if !haskey( dependencies, re )
        node = Node( (rng, a) -> rand!( rng, re.dist, a ), Node[], T )
        dependencies[re, node] = Symbol("x" * string(length(dependencies)))
    end
    return dependencies[re]
end

function rand_expr( re::TransformedRandomElement{Op,T}, dependencies::Dependencies ) where {Op,T}
    exprs = rand_expr.( re.args, [dependencies] )
    return Expr( :call, Op, exprs... )
end

node( s::Symbol, dependencies::Dependencies, T ) = dependencies[s]

function node( expr::Expr, dependencies::Dependencies, T )
    f = eval( Expr( :->, Expr( :tuple, :rng, :a, symbols(dependencies)... ), expr ) )
    return Node( f, nodes(dependencies), T )
end

function Random.Sampler( rng::AbstractRNG, re::AbstractRandomElement{T}, ::Random.Repetition ) where {T}
    dependencies = Dependencies()
    expr = rand_expr( re, dependencies )
    return node( expr, dependencies, T )
end

Random.gentype( ::Node{T} ) where T = T

function Random.rand!( rng::AbstractRNG, node::Node{T} ) where T
    n = length(node.cache)
    for dep in node.dependencies
        if !dep.calculated
            if dep.visited
                error( "Circular reference found" )
            end
            dep.visited = true
            if length( dep.cache ) != n
                dep.cache = Array{Random.gentype(dep)}( undef, n )
            end
            rand!( rng, dep )
        end
    end
    node.cache[:] = invokelatest( node.calculation, rng, node.cache, getfield.( node.dependencies, :cache )... )
    return node.cache
end

function clear!( node::Node )
    node.visited = false
    node.calculated = false
    for dep in node.dependencies
        if dep.visited
            clear!( dep )
        end
    end
end

function Random.rand!( rng::AbstractRNG, a::AbstractArray{T}, node::Node{T} ) where T
    node.cache = a
    rand!( rng, node )
    clear!( node )
    return a
end

abstract type AbstractSequence{T}
end

const AbstractTimeSeries{T} = AbstractRandomElement{AbstractSequence{T}}

struct Time
    lag::Int
end

Time() = Time( 0 )

Base.:+( i::Integer, t::Time ) = Time( t.lag + i )
Base.:+( t::Time, i::Integer ) = Time( t.lag + i )

Base.:-( t::Time, i::Integer ) = Time( t.lag - i )

mutable struct TimeSeries{T, U <: AbstractTimeSeries{T}} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractRandomElement{T}
    induction::Union{U,Nothing}
    t::Union{Time,Nothing}
end

TimeSeries{T,U}( base::AbstractRandomElement{T} ) where {T,U} = TimeSeries{T,U}( base, nothing, nothing )

TimeSeries( dist::Distribution = Dirac(0.0) ) =
    TimeSeries{Float64,AbstractTimeSeries{Float64}}( IndependentRandomElement( dist ) )

function Base.setindex!( ts0::TimeSeries{T,U}, ts1::U, t::Time ) where {T,U}
    ts0.induction = ts1
    ts0.t = t
end

struct LaggedTimeSeries{T,U <: AbstractTimeSeries{T}} <: AbstractRandomElement{AbstractSequence{T}}
    base::U
    t::Time
end

Base.getindex( ts::AbstractTimeSeries{T}, t::Time ) where {T} = LaggedTimeSeries( ts, t )

struct IID{T} <: AbstractTimeSeries{T}
    dist::Distribution
end

IID( dist::Distribution{Univariate} ) = IID{Float64}( dist )

struct SequenceNode{T}
    calc::Union{Nothing,Function}
    dependencies::Vector{SequenceNode}
    lags::Vector{UInt}
    cache::Vector{T}
end

end # module

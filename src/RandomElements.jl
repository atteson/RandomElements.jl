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

    vop = Meta.parse( "." * string(op) )
    e3 = quote
        Base.$op( x::AbstractRandomElement{T}, y::AbstractRandomElement{T} ) where T =
            TransformedRandomElement{$vop, T, AbstractRandomElement{T}}( AbstractRandomElement{T}[x,y] )
    end
    eval( e3 )
end

Base.promote_rule( ::Type{T}, ::Type{V} ) where {T <: Number, U <: Number, V <: AbstractRandomElement{U}} =
    AbstractRandomElement{promote_type(T,U)}

Base.promote_rule( ::Type{U}, ::Type{W} ) where {T <: Number, U <: AbstractRandomElement{T}, V <: Number, W <: AbstractRandomElement{V}} =
    AbstractRandomElement{promote_type(T,V)}

Base.convert( ::Type{AbstractRandomElement{T}}, x::U ) where {T <: Number, U <: Number} =
    IndependentRandomElement(Dirac(convert(T, x)))

# world age problem prohibits us from compiling larger chunks of code
# this uses the "numpy" approach of relying on large-scale vectorization instead
mutable struct Node{F,T,N}
    calculation::F
    dependencies::Vector{N}
    cache::T
    visiting::Bool
    calculated::Bool
end

Node( calculation::Function, dependencies::Vector{N}, T ) where {N <: Node} =
    Node( calculation, dependencies, T[], false, false )

function run!( node::Node{F,T} ) where {F,T}
    if node.visiting
        error( "Already visiting $node" )
    elseif !node.calculated
        node.visiting = true
        args = run!.( node.dependencies )
        node.cache = node.calculation( args... )
        node.calculated = true
        node.visiting = false
    end
    return node.cache
end

function clear!( node::Node{F,T} ) where {F,T}
    if node.visiting
        node.visiting = false
        node.calculated = false
        clear!.( node.args )
    end
end

const Dependencies = Dict{AbstractRandomElement, Node}

node( s::Symbol, dependencies::Dependencies, T ) = dependencies[s]

function node( expr::Expr, dependencies::Dependencies, T )
    f = eval( Expr( :->, Expr( :tuple, :rng, :a, symbols(dependencies)... ), expr ) )
    return Node( f, nodes(dependencies), T )
end

function rand_graph!( rng::AbstractRNG, re::IndependentRandomElement{T}, dependencies::Dependencies, dims::Dims ) where {T}
    if !haskey( dependencies, re )
        node = Node( () -> rand( rng, re.dist, dims ), Node[], T )
        dependencies[re] = node
    end
    return dependencies[re]
end

function rand_graph!( rng::AbstractRNG, re::TransformedRandomElement{Op,T}, dependencies::Dependencies, dims::Dims ) where {Op,T}
    if !haskey( dependencies, re )
        deps = rand_graph!.( rng, re.args, [dependencies], [dims] )
        node = Node( Op, deps, T )

        dependencies[re] =  node
    end
    return dependencies[re]
end

Random.rand( rng::AbstractRNG, re::AbstractRandomElement{T}, dims::Dims ) where {T} = rand( rng, [re], dims )[1]

function Random.rand( rng::AbstractRNG, res::AbstractVector{R}, dims::Dims ) where {R <: AbstractRandomElement}
    dependencies = Dependencies()
    graphs = rand_graph!.( [rng], res, [dependencies], [dims] )
    return run!.( graphs )
end

Random.rand( res::AbstractVector{R} ) where {R <: AbstractRandomElement} = getindex.( rand( Random.default_rng(), res, (1,) ), 1 )

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
    cache::Array{T}
end

rand_graph!( rng::AbstractRNG, ts::LaggedTimeSeries{T,IID{T}}, dims::Dims, t::Time ) where {T} =
    SequenceNode( () -> rand( rng, ts.base.dist, dims ), SequenceNode[], T[] )

function rand_graph!(
    rng::AbstractRNG,
    ts::TransformedRandomElement{Op,AbstractSequence{T}},
    dims::Dims,
    t::Time,
) where {T,Op}
    return SequenceNode( Op, ts.args, UInt[], T[] )
end

function rand_graph!( rng::AbstractRNG, ts::TimeSeries{T,U}, dims::Dims ) where {T,U}
    assert( ts.induction != nothing )
    assert( ts.time != nothing )

    return rand_graph!( rng, ts.induction, dims::Dims, ts.t )
end

max_lag( ts::LaggedTimeSeries{T,U}, inside::Bool = true ) where {T,U} =
    ts.t.lag + max_lag( ts.base, inside )

max_lag( ts::TransformedRandomElement{Op,AbstractSequence{T}}, inside::Bool = true ) where {Op,T} =
    max( max_lag.( ts.args, inside )... )

max_lag( ts::IID{T}, inside::Bool = true ) where {T} = 0

max_lag( ts::TimeSeries{T,U}, inside::Bool = true ) where {T,U} =
    inside ? 0 : max_lag( ts.induction )

end # module

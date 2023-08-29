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

mutable struct TimeSeries{T} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractRandomElement{T}
    induction::Union{AbstractRandomElement{T}, Nothing}
    t::Union{Time, Nothing}
end

TimeSeries{T}( base::AbstractRandomElement{T} ) where {T} =
    TimeSeries{T}( base, nothing, nothing )

TimeSeries( dist::Distribution = Dirac(0.0) ) =
    TimeSeries{Float64}( IndependentRandomElement( dist ) )

function Base.setindex!( ts0::TimeSeries{T}, ts1::AbstractRandomElement{T}, t::Time ) where {T}
    ts0.induction = ts1
    ts0.t = t
end

struct IndexedTimeSeries{T, U <: AbstractTimeSeries{T}} <: AbstractRandomElement{T}
    base::U
    t::Time
end

Base.getindex( ts::AbstractTimeSeries{T}, t::Time ) where {T} = IndexedTimeSeries( ts, t )

struct IID{T} <: AbstractTimeSeries{T}
    dist::Distribution
end

IID( dist::Distribution{Univariate} ) = IID{Float64}( dist )

struct SequenceNode{T}
    calc::Union{Nothing,Function}
    dependencies::Vector{SequenceNode}
    cache::Array{T}
end

rand_graph!( rng::AbstractRNG, ts::IndexedTimeSeries{T,IID{T}}, dims::Dims, t::Time ) where {T} =
    SequenceNode( () -> rand( rng, ts.base.dist, dims ), SequenceNode[], T[] )

function rand_graph!(
    rng::AbstractRNG,
    ts::TransformedRandomElement{Op,AbstractSequence{T}},
    dims::Dims,
    t::Time,
) where {T,Op}
    return SequenceNode( Op, ts.args, UInt[], T[] )
end

function rand_graph!( rng::AbstractRNG, ts::TimeSeries{T}, dims::Dims ) where {T}
    assert( ts.induction != nothing )
    assert( ts.time != nothing )

    return rand_graph!( rng, ts.induction, dims::Dims, ts.t )
end

max_lag( ts::IndexedTimeSeries{T,U}, t0::Int ) where {T,U} =
    t0 - ts.t.lag + max_lag( ts.base, t0 )

max_lag(
    ts::TransformedRandomElement{Op,T,U}, t0::Int
) where {Op, T, U <: AbstractRandomElement{T}} =
    max( max_lag.( ts.args, t0 )... )

max_lag( _, ::Int ) = 0

max_lag( ts::TimeSeries{T} ) where {T} = max_lag( ts.induction, ts.t.lag )

end # module

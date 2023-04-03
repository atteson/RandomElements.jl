module RandomElements

export IndependentRandomElement

using Distributions
using Random

abstract type AbstractRandomElement{T}
end

Base.eltype( ::Type{U} ) where {T, U <: AbstractRandomElement{T}} = T

mutable struct IndependentRandomElement{T,D <: Distribution} <: AbstractRandomElement{T}
    dist::D
end

IndependentRandomElement( dist::D ) where {D <: Distribution} = IndependentRandomElement{eltype(dist),D}( dist )

struct TransformedRandomElement{O,T,U <: AbstractRandomElement{T}} <: AbstractRandomElement{T}
    # for now, all elements must be of the same type
    args::Vector{U}
end

const NRE = Union{AbstractRandomElement,Number}

for op in [:+,:*,:/,:-]
    e1 = :( Base.$op( x::NRE, y::NRE ) = $op( promote( x, y )... ) )
    eval( e1 )
    
    e2 = quote
        Base.$op( x::AbstractRandomElement{T}, y::AbstractRandomElement{T} ) where T =
            TransformedRandomElement{$op, T, AbstractRandomElement{T}}( AbstractRandomElement{T}[x,y] )
    end
    eval( e2 )
end

Base.promote_rule( ::Type{T}, ::Type{V} ) where {T <: Number, U, V <: AbstractRandomElement{U}} =
    AbstractRandomElement{promote_type(T,U)}

Base.convert( ::Type{AbstractRandomElement{T}}, x::U ) where {T,U <: Number} =
    IndependentRandomElement(Dirac(convert(T, x)))

function memoize( d::Dict, k, f::Function )
    if !haskey( d, k )
        d[k] = f()
    end
    return d[k]
end

Base.rand(
    rng::AbstractRNG,
    ire::IndependentRandomElement{T};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
) where T = 
    memoize( assigned, ire, () -> rand( rng, ire.dist ) )

Base.rand(
    rng::AbstractRNG,
    tre::TransformedRandomElement{O,T,U};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
) where {O,T,U} =
    memoize( assigned, tre, () -> O( rand.( rng, tre.args, assigned=assigned )... ) )

Base.rand(
    rng::AbstractRNG,
    sp::Random.SamplerTrivial{T},
) where {T <: AbstractRandomElement} =
    rand( rng, sp[], assigned=Dict{AbstractRandomElement,Any}() )

struct RandomElementSampler{T} <: Random.Sampler{T}
    re::T
end

Random.gentype( ::Type{Vector{U}} ) where {T, U <: AbstractRandomElement{T}} = Vector{T}

Random.Sampler( ::Type{<:AbstractRNG}, vre::Vector{<:AbstractRandomElement}, repetition::Random.Repetition ) =
    RandomElementSampler( vre )

Base.rand( rng::AbstractRNG, sp::RandomElementSampler ) = rand.( rng, sp.re, assigned=Dict{AbstractRandomElement,Any}() )

abstract type AbstractSequence{T}
end

abstract type AbstractTimeSeries{T} <: AbstractRandomElement{AbstractSequence{T}}
end

struct TimeSeries{T, U <: AbstractTimeSeries{T}} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractRandomElement{T}
    induction::U
end

struct LaggedTimeSeries{T} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractRandomElement{AbstractSequence{T}}
end

struct IIDTimeSeries{T} <: AbstractRandomElement{AbstractSequence{T}}
    dist::Distribution
end

struct IndexedTimeSeries{T, U <: AbstractVector{Int}, V <: AbstractTimeSeries{T}} <: AbstractRandomElement{Vector{T}}
    indices::U
    ts::V
end

Base.getindex( ts::AbstractTimeSeries{T}, indices::AbstractVector{Int} ) where {T} = IndexedTimeSeries( indices, ts )

struct Node{F,T}
    calc::F
    dependencies::Vector{Node}
    cache::Vector{T}
end

function Base.getindex( node::Node{F,T}, i::Int ) where {T,F}
    j = length(node.cache)
    while i > j
        j += 1
        node.cache.push!( node.calc( getindex.( node.dependencies, j )... ) )
    end
    return node.cache[i]
end

rand_node( ts::IIDTimeSeries{T} ) where {T} = Node( () -> rand( ts.dist ), Node[], T[] )

rand_node( ts::

function Base.rand( rng::AbstractRNG, ts::IndexedTimeSeries{T,U,V} ) where {T,U,V}
    v = Vector{T}( undef, length(ts.indices) )
    for i = 0:maximum(ts.indices)
    end
    return v
end
    

end # module

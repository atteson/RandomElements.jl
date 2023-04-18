module RandomElements

export IndependentRandomElement, TimeSeries, IID, lag

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
>>>>>>> 100ec5ed39d4e0d9846d1f0e5b25ce8111ce6d8e
    # for now, all elements must be of the same type
    args::Vector{U}
end


const NRV = Union{AbstractRandomVariable,Number}

for op in [:+,:*,:/,:-]
    e1 = :( Base.$op( x::NRV, y::Number ) = $op( promote( x, y )... ) )
    eval( e1 )
    

    e2 = quote
        Base.$op( x::AbstractRandomElement{T}, y::AbstractRandomElement{T} ) where T =
            TransformedRandomElement{$op, T, AbstractRandomElement{T}}( AbstractRandomElement{T}[x,y] )
    end
    
    eval( e2 )
    
    e3 = quote
        Base.$op( x::U, y::V ) where {T, U <: AbstractRandomVariable{T}, V <: AbstractRandomVariable{T}} =
            TransformedRandomVariable( $op, AbstractRandomVariable{T}[x,y] )
    end
    eval( e3 )
end

Base.promote_rule( ::Type{T}, ::Type{V} ) where {T <: Number, U, V <: AbstractRandomVariable{U}} =
    AbstractRandomVariable{promote_type(T,U)}

Base.convert( ::Type{AbstractRandomVariable{T}}, x::U ) where {T,U <: Number} =
    IndependentRandomVariable(Dirac(convert(T, x)))

function memoize( d::Dict, k, f::Function )
    if !haskey( d, k )
        d[k] = f()
    end
    return d[k]
end

Base.rand(
    rng::AbstractRNG,
    irv::IndependentRandomVariable{T};
    assigned::Dict{AbstractRandomVariable,Any} = Dict{AbstractRandomVariable,Any}(),
) where T = 
    memoize( assigned, irv, () -> rand( rng, irv.dist ) )

Base.rand(
    rng::AbstractRNG,

    tre::TransformedRandomElement{O,T,U};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
) where {O,T,U} =
    memoize( assigned, tre, () -> O( rand.( rng, tre.args, assigned=assigned )... ) )

Base.rand(
    rng::AbstractRNG,
    sp::Random.SamplerTrivial{T},
) where {T <: AbstractRandomVariable} =
    rand( rng, sp[], assigned=Dict{AbstractRandomVariable,Any}() )

struct RandomVariableSampler{T} <: Random.Sampler{T}
    rv::T
end

Random.gentype( ::Type{Vector{U}} ) where {T, U <: AbstractRandomVariable{T}} = Vector{T}

Random.Sampler( ::Type{<:AbstractRNG}, vre::Vector{<:AbstractRandomVariable}, repetition::Random.Repetition ) =
    RandomVariableSampler( vre )

Base.rand( rng::AbstractRNG, sp::RandomElementSampler ) = rand.( rng, sp.re, assigned=Dict{AbstractRandomElement,Any}() )

abstract type AbstractSequence{T}
end

const AbstractTimeSeries{T} = AbstractRandomElement{AbstractSequence{T}}

mutable struct TimeSeries{T, U <: AbstractTimeSeries{T}} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractRandomElement{T}
    induction::Union{U, Nothing}
end

TimeSeries( dist::Distribution = Dirac(0.0) ) =
    TimeSeries{Float64,AbstractTimeSeries{Float64}}( IndependentRandomElement( dist ), nothing )

function Base.setindex!( ts0::TimeSeries{T,U}, ts1::U ) where {T,U}
    ts0.induction = ts1
end

struct LaggedTimeSeries{T} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractTimeSeries{T}
end

lag( ts::AbstractTimeSeries{T} ) where T = LaggedTimeSeries( ts )

struct IID{T} <: AbstractTimeSeries{T}
    dist::Distribution
end

IID( dist::Distribution{Univariate} ) = IID{Float64}( dist )

Base.getindex( ts::AbstractTimeSeries{T}, indices::AbstractVector{Int} ) where {T} = IndexedTimeSeries( indices, ts )

struct Node{F,T} <: AbstractSequence{T}
    calc::F
    dependencies::Vector{Node}
    cache::Vector{T}
end

function Base.getindex( node::Node{F,T}, i::Int ) where {T,F}
    j = length(node.cache)
    while i > j
        j += 1
        push!( node.cache, node.calc( getindex.( node.dependencies, j )... ) )
    end
    return node.cache[i]
end

Base.rand(
    rng::AbstractRNG,
    sp::RandomVariableSampler,
    ts::IID{T};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
) where {T} =
    memoize( assigned, ts, () -> Node( () -> rand( ts.dist ), Node[], T[] ) )

end # module

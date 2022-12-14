module RandomElements

export IndependentRandomElement

using Distributions
using Random

abstract type AbstractRandomElement{T} end

Base.eltype( ::Type{U} ) where {T, U <: AbstractRandomElement{T}} = T

mutable struct IndependentRandomElement{T,D <: Distribution} <: AbstractRandomElement{T}
    dist::D
end

IndependentRandomElement( dist::D ) where {D <: Distribution} = IndependentRandomElement{eltype(dist),D}( dist )

struct TransformedRandomElement{T,U <: AbstractRandomElement{T}} <: AbstractRandomElement{T}
    op::Function
    # for now, all elements must be of the same type
    args::Vector{U}
end

const NRE = Union{AbstractRandomElement,Number}

for op in [:+,:*,:/,:-]
    e1 = :( Base.$op( x::NRE, y::NRE ) = $op( promote( x, y )... ) )
    eval( e1 )
    
    e2 = quote
        Base.$op( x::AbstractRandomElement{T}, y::AbstractRandomElement{T} ) where T =
            TransformedRandomElement( $op, AbstractRandomElement{T}[x,y] )
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
    tre::TransformedRandomElement{T,U};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
) where {T,U} =
    memoize( assigned, tre, () -> tre.op( rand.( rng, tre.args, assigned=assigned )... ) )

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

Base.rand(
    rng::AbstractRNG,
    sp::RandomElementSampler,
) where {N, T<:AbstractRandomElement} =
    rand.( rng, sp.re, assigned=Dict{AbstractRandomElement,Any}() )

end # module

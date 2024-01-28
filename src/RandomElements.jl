module RandomElements

using Distributions
using Random

export IndependentRandomVariable

abstract type AbstractRandomElement{T} end

Base.eltype( ::Type{U} ) where {T, U <: AbstractRandomElement{T}} = T

abstract type AbstractRandomVariable{T <: Number} <: AbstractRandomElement{T} end

mutable struct IndependentRandomVariable{T} <: AbstractRandomVariable{T}
    dist::Distribution
end

IndependentRandomVariable( dist::Distribution ) = IndependentRandomVariable{eltype(dist)}( dist )

struct AbstractRandomVector{T,V <: AbstractVector{T}} <: AbstractRandomElement{T}
    v::V
end


struct TransformedRandomVariable{T,U <: AbstractRandomVariable{T}} <: AbstractRandomVariable{T}
    op::Function
    # for now, all elements must be of the same type
    args::Vector{U}
end


const NRV = Union{AbstractRandomVariable,Number}

for op in [:+,:*,:/,:-]
    e1 = :( Base.$op( x::NRV, y::Number ) = $op( promote( x, y )... ) )
    eval( e1 )
    
    e2 = :( Base.$op( x::N, y::NRV ) = $op( promote( x, y )... ) )
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
    trv::TransformedRandomVariable{T,U};
    assigned::Dict{AbstractRandomVariable,Any} = Dict{AbstractRandomVariable,Any}(),
) where {T,U} =
    memoize( assigned, trv, () -> trv.op( rand.( rng, trv.args, assigned=assigned )... ) )

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

Base.rand(
    rng::AbstractRNG,
    sp::RandomVariableSampler,
) where {N, T<:AbstractRandomVariable} =
    rand.( rng, sp.rv, assigned=Dict{AbstractRandomVariable,Any}() )

end # module

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

struct TransformedRandomElement{O,T,U <: AbstractRandomElement{T}} <: AbstractRandomElement{T}
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

function memoize( d::Dict, k, f::Function )
    if !haskey( d, k )
        d[k] = f()
    end
    return d[k]
end

Base.rand(
    rng::AbstractRNG,
    irv::IndependentRandomElement{T};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
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
    sp::Random.SamplerTrivial{U},
) where {T <: Number, U <: AbstractRandomElement{T}} =
    rand( rng, sp[], assigned=Dict{AbstractRandomElement,Any}() )

struct RandomElementSampler{T} <: Random.Sampler{T}
    re::T
end

Random.gentype( ::Type{Vector{U}} ) where {T <: Number, U <: AbstractRandomElement{T}} = Vector{T}

Random.Sampler( ::Type{<:AbstractRNG}, vre::Vector{<:AbstractRandomElement}, repetition::Random.Repetition ) =
    RandomElementSampler( vre )

Base.rand( rng::AbstractRNG, sp::RandomElementSampler ) = rand.( rng, sp.re, assigned=Dict{AbstractRandomElement,Any}() )

abstract type AbstractSequence{T}
end

const AbstractTimeSeries{T} = AbstractRandomElement{AbstractSequence{T}}

struct Time
    lag::Int
end

Time() = Time(0)

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

struct LaggedTimeSeries{T} <: AbstractRandomElement{AbstractSequence{T}}
    base::AbstractTimeSeries{T}
    t::Time
end

Base.getindex( ts::AbstractTimeSeries{T}, t::Time ) where {T} = LaggedTimeSeries( ts, t )

struct IID{T} <: AbstractTimeSeries{T}
    dist::Distribution
end

IID( dist::Distribution{Univariate} ) = IID{Float64}( dist )

Base.getindex( ts::AbstractTimeSeries{T}, indices::AbstractVector{Int} ) where {T} = IndexedTimeSeries( indices, ts )

struct Node{F,T} <: AbstractSequence{T}
    calc::F
    lagged_vars::Vector{Node}
    cache::Vector{T}
end

function Base.getindex( node::Node{F,T}, i::Int ) where {T,F}
    j = length(node.cache)
    while i > j
        j += 1
        push!( node.cache, node.calc( getindex.( node.lagged_vars, j-1 )... ) )
    end
    return node.cache[i]
end

Expr( ts::IID{T}, lagged_vars::Dict{AbstractTimeSeries,Symbol} ) where {T} = Expr( :call, [:rand, ts.dist] )

function Expr( ts::LaggedTimeSeries{T}, lagged_vars::Dict{AbstractTimeSeries,Symbol} ) where {T}
    if !haskey( lagged_vars, ts )
        lagged_vars[ts] = Symbol("x" + string(length(lagged_vars)))
    end
    return lagged_vars[ts]
end

function Expr( ts::TimeSeries{T}, lagged_vars::Dict{AbstractTimeSeries,Symbol} ) where {T}
end

Base.rand(
    rng::AbstractRNG,
    ts::IID{T};
    assigned::Dict{AbstractRandomElement,Any} = Dict{AbstractRandomElement,Any}(),
) where {T} =
    memoize( assigned, ts, () -> Node( () -> rand( ts.dist ), Node[], T[] ) )

end # module

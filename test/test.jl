using RandomElements
using Random
using Distributions
using LinearAlgebra

Random.seed!(1)
X = IndependentRandomVariable( Normal() )
N = 1_000
xs = rand( X, N );
@assert( abs(mean(xs)) < 3/sqrt(N) )

Y = X + 1
ys = rand( Y, N );
@assert( abs(mean(ys) - 1) < 3/sqrt(N) )

xys=rand( [X,Y], 10 )
@assert( maximum(abs.(map( a->-(a...), xys ) .+ 1)) < 1e-6 )

X = IndependentRandomVariable( Normal() )
Y = IndependentRandomVariable( Normal() )
@assert( abs(-(rand( [X,Y] )...)) > 1e-6 )

@assert( abs(-(rand( [X+1, X+1] )...)) < 1e-6 )
@assert( abs(-(rand( [X+1, Y+1] )...)) > 1e-6 )


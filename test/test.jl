using RandomElements
using Random
using Distributions

Random.seed!(1)
X = IndependentRandomElement( Normal() )
N = 1_000
xs = rand( X, N );
@assert( abs(mean(xs)) < 3/sqrt(N) )

Y = X + 1
ys = rand( Y, N );
@assert( abs(mean(ys) - 1) < 3/sqrt(N) )

rand( [X,Y], 10 )

X = IndependentRandomElement( Normal() )
Y = IndependentRandomElement( Normal() )
@assert( abs(-(rand( [X,Y] )...)) > 1e-6 )


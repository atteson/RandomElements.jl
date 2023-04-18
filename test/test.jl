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

Z = IndependentRandomElement( MvNormal( [0,0], I(2) ) )
rand( [X, Z] )

W = X + 3 * Y + 1
@time w = rand( W, 1_000_000 );
@assert( abs(mean(w) - 1) < 0.01 )
@assert( abs( std(w) - sqrt(10) ) < 0.01 )

X = TimeSeries()
X[] = lag( X ) + IID( Normal() )

e = :( x -> x + 1 )
e.head
e.args
e.args[2].args

X = IID( Normal() )
s = rand( X )

N = 1_000
S = [s[i] for i in 1:N];
@assert( abs(sum(S)/sqrt(N)) < 3 )


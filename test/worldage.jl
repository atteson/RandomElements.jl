using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

struct S{F}
    f
end

function g( i )
    s = S( @RuntimeGeneratedFunction( :( (j) -> j + $i ) ) )
    s.f(1)
end

g(1)
g(2)

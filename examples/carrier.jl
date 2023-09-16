using Plots
using LinearAlgebra
using ApproxFun
using ApproxFunNewton

d = Chebyshev(-1..1)
x = Fun(d)
u0 = 2(x^2-1)*(1-2/(1+20x^2))

N(u,x=Fun(d)) = [u(-1); u(1); 0.001u'' + 2(1-x^2)u + u^2 - 1]

u = ApproxFunNewton.newton(N, u0; θ̄=1, λmin=1e-12, maxiterations=25, tolerance=1e-15, damped=true, verbose=true)

plot(u, title="Carrier Problem", xlabel="x", ylabel="u(x)", legend=false)

savefig("images/carrier.png")

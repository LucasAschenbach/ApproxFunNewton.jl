using Plots
using LinearAlgebra
using ApproxFun
using ApproxFunNewton

R = 1e4 # Reynolds number
Pₑ = 0.7*R # Peclet number

d = Chebyshev(0..1)

L = (R,f,h,θ,A) -> [f(0); f(1)-1;
                    f'(0); f'(1);
                    h(0); h(1);
                    θ(0); θ(1)-1;
                    f'''-R*((f')^2-f*f''-A);
                    h''+R*f*h'+1;
                    θ''+0.7*R*f*θ';
                    A']

x = Fun(d)
f0 = -2x^3+3x^2; h0 = 0*x; θ0 = -(x-1)^2+1; A0 = 0*x

# DIRECT SOLVE

f, h, θ, A = ApproxFunNewton.newton((f,h,θ,A) -> L(R,f,h,θ,A), [f0,h0,θ0,A0], θ̄=1e15, damped=false, verbose=true)
u = [f,h,θ,A]

plts = []
for (i,fun) in enumerate(["f(x)", "h(x)", "θ(x)", "A"])
    plt = plot(u[i], xlabel="x", ylabel=fun, legend=false)
    push!(plts, plt)
end
plt = plot(plts..., layout=4, plot_title="Fluid Injection (R=$R, Pₑ=$Pₑ)", size=(800,800))

savefig(plt, "images/fluidinjection.png")

# CONTINUATION

f0, h0, θ0, A0 = ApproxFunNewton.newton((f,h,θ,A) -> L(10.,f,h,θ,A), [f0,h0,θ0,A0], θ̄=1e15, damped=false, verbose=true)
vlist = continuation(L, [f0,h0,θ0,A0], a=10., b=1000., λ0=10, verbose=true)

pltscont = []
for (i,fun) in enumerate(["f(x)", "h(x)", "θ(x)", "A"])
    plt = plot(vlist, vars=Tuple(i), xlabel="x", ylabel=fun, legend=false)
    push!(pltscont, plt)
end
pltcont = plot(pltscont..., layout=4, plot_title="Fluid Injection (R=$R, Pₑ=$Pₑ)", size=(800,800))

savefig(pltcont, "images/fluidinjection_cont.png")

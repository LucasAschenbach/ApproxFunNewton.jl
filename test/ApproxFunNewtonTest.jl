module ApproxFunNewtonTest

using ApproxFun
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "testutils.jl"))
include("../src/newton.jl")

@verbose @testset "Newton" begin
    @testset "Single unknown" begin
        x = Fun(Chebyshev(0..1))

        # Test a nonlinear ODE with nonlinear BCs
        u = 0.5*one(x)

        N = u -> [u(0)*u(0)*u(0) - 0.125;
                  u' - u*u]
        u = newton(N, u)

        u_exact = -1 / (x - 2)
        @test norm(u - u_exact) < 1e-15
    end

    @testset "Multiple unknowns" begin
        x = Fun(Chebyshev(0..1))

        # Test two coupled systems of nonlinear ODEs with nonlinear BCs
        u1 = 0.5*one(x)
        u2 = 0.5*one(x)

        N_dep = (u1,u2) -> [
                u1(0)*u1(0)*u1(0) - 0.125;
                u2(0) - 1;
                u1' - u1*u1;
                u2' + u1*u1*u2;
            ]
        u1, u2 = newton(N_dep, [u1,u2])

        u1_exact = -1 / (x - 2)
        u2_exact = exp(1 / (x - 2) + 1/2)
        @test norm(u2 - u2_exact) < 1e-15
        @test norm(u1 - u1_exact) < 1e-15

        # Test two independent systems of nonlinear ODEs with nonlinear BCs
        u1 = 0.5*one(x)
        u2 = 0.5*one(x)

        N_ind = (u1,u2) -> [
                u1(0)*u1(0)*u1(0) - 0.125;
                u2(0)*u2(0)*u2(0) + 0.125;
                u1' - u1*u1;
                u2' - u2*u2;
            ]

        # note: disable monotonicity test
        u1, u2 = newton(N_ind, [u1,u2], θ̄=Inf)

        u1_exact = -1 / (x - 2)
        u2_exact = -1 / (x + 2)
        @test norm(u2 - u2_exact) < 1e-15
        @test norm(u1 - u1_exact) < 1e-15

        # Compare to original newton for one equation and verify all solutions correct
        N_single = u -> [u(0)*u(0)*u(0) - 0.125;
                        u' - u*u]

        u = 0.5*one(x)
        u = newton(N_single, [u])
        @test norm(u - u1) < 1e-15
    end

    @testset "Continuation" begin
        x = Fun(Chebyshev(-1..1))
        u0 = 0*x
    
        N = (λ,u,x=Fun()) -> [
                u(-1.)-1.;
                u(1.)+0.5;
                0.001u'' + 6(1-x^2)u' + λ*u^2 - 1
            ]

        vlist = continuation(N, [u0])
        λ, u = last(vlist)
        r = N(λ, u)
        r = collect(Fun, r)

        @test λ == 1.0
        @test maximum([abs(u(-1.)-1), abs(u(1.)+0.5), norm(r[3])]) < 1e-14
    end
end

end # module NewtonTest

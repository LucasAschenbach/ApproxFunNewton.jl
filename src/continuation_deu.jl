# Alternative continuation method using Deuflhard's Adaptive Stepsize Control

export continuation_deu, plot

using ApproxFun
import ApproxFun: DualFun, Fun, chop, jacobian
import DomainSets: AbstractInterval
import LinearAlgebra: norm
import Plots: plot, plot!
using DualNumbers
using Printf
using Plots

# Combines a list of `Fun`s with a parameter to behave like a vector of the
# parameter and the coefficients concatenated
struct ParamFun{S<:Number, T<:Fun}
    λ::S
    u::Array{T,1}
end

ParamFun() = ParamFun(0., [Fun(0)])
ParamFun(λ::Number, u::Fun) = ParamFun(λ, [u])

# Basic operations
Base.@propagate_inbounds function Base.iterate(pf::ParamFun, state=1)
    state > length(pf) && return nothing
    (state == 1 ? pf.λ : pf.u[state-1], state + 1)
end
Base.length(pf::ParamFun) = length(pf.u) + 1
Base.show(io::IO, pf::ParamFun) = print(io, "[$(pf.λ), $(pf.u)]")

# Broadcasting
Base.@propagate_inbounds function Base.getindex(pf::ParamFun, i::Integer)
    i == 1 && return pf.λ
    pf.u[i-1]
end

Base.broadcast_axes(::Type{<:ParamFun}, A) = axes(A)
Base.broadcastable(pf::ParamFun) = pf

struct ParamFunStyle <: Base.BroadcastStyle end

Base.BroadcastStyle(::Type{<:ParamFun}) = ParamFunStyle()
Base.BroadcastStyle(::Type{<:ParamFun}, ::Type{<:Any}) = ParamFunStyle()


# Arithmetic

# component-wise operations
for OP in (:(==), :<, :<=, :>, :>=)
    @eval Base.$OP(z::ParamFun, w::ParamFun) = $OP(z.λ, w.λ) && $OP(z.u, w.u)
end

for OP in (:+, :-)
    @eval Base.$OP(z::ParamFun, w::ParamFun) = begin
        @assert length(z.u) == length(w.u) "ParamFun must have same length"
        ParamFun($OP(z.λ, w.λ), $OP(z.u, w.u))
    end
end

# scalar operations
for OP in (:*, :/, :+, :-)
    @eval Base.$OP(z::Number, w::ParamFun) = ParamFun($OP(z, w.λ), $OP(z, w.u))
    @eval Base.$OP(z::ParamFun, w::Number) = ParamFun($OP(z.λ, w), $OP(z.u, w))
end
Base.:-(z::ParamFun) = ParamFun(-z.λ, -z.u)

# vector operations
LinearAlgebra.norm(z::ParamFun, p::Real=2) = norm([z.λ, maximum(norm.(z.u, p))], p)
function LinearAlgebra.dot(z::ParamFun, w::ParamFun)
    @assert length(z.u) == length(w.u) "ParamFun must have same length"
    cfs_sum = 0
    for i in 1:length(z.u)
        cfs_u, cfs_w = coefficients(z.u[i]), coefficients(w.u[i])
        if length(cfs_u) == length(cfs_w)
            cfs_sum += dot(cfs_u, cfs_w)
        elseif length(cfs_u) > length(cfs_w)
            cfs_sum += dot(cfs_u, [cfs_w; zeros(eltype(cfs_w), length(cfs_u)-length(cfs_w))])
        else
            cfs_sum += dot([cfs_u; zeros(eltype(cfs_u), length(cfs_w)-length(cfs_u))], cfs_w)
        end
    end
    return dot(z.λ,w.λ) + cfs_sum
end

# ApproxFun overrides
ApproxFunBase.setdomain(f::ParamFun, d::Domain) = ParamFun(f.λ, [Fun(setdomain(space(u), d), u.coefficients) for u in f.u])

function tangent(N, v::ParamFun{Float64}, t₀::ParamFun{Float64})
    λ, u = v.λ, v.u
    # jacobian
    Nλ = N(Dual(λ,1), u...)
    Jλ = map(d -> dualpart.(coefficients(d)), Nλ)
    Jus = Array{Any,2}(undef,1,length(u)) # jacobians
    for i = 1:length(u)
        @inbounds Jus[i] = jacobian.(N(λ, [ (j == i ? DualFun(u[j]) : u[j]) for j = 1:length(u) ]...))
    end
    Ju = Base.typed_hcat(Operator, Jus...)
    # kernel
    tu = -Ju\Jλ
    t = ParamFun(1, collect(Fun, tu))
    t = t/norm(t)
    # make sure angle is acute
    if dot(t₀,t) < 0
        t = -t
    end
    return t
end

function topath(N, u0::Array{T,1}; θ̄=2, maxiterations=15, tolerance=1e-15, verbose=false) where T <: Fun
    u, θ₁, δ, ε, k, err, info = copy(u0), tolerance, Inf, Inf, 0, Inf, Nothing
    numf = length(u0) # number of functions
    if numf == 0
        error("u0 must contain at least 1 function")
    end
    Js = Array{Any}(undef,numf) # jacobians
    while true
        @stop(:MAXIT, k >= maxiterations)
        k += 1

        F = N([ (j == 1 ? DualFun(u[j]) : u[j]) for j = 1:numf ]...)
        @inbounds Js[1] = jacobian.(F)
        F = map(d -> typeof(d) <: DualFun ? d.f : d, F)
        for i = 2:numf
            @inbounds Js[i] = jacobian.(N([ (j == i ? DualFun(u[j]) : u[j]) for j = 1:numf ]...))
        end
        J = Base.typed_hcat(Operator, Js...); J = qr(J)

        Δu = J\F
        Δu_norm = err = maximum(norm.(components(Δu)))
        u = u .- Δu
        if k == 1
            δ = Δu_norm
        end
        if Δu_norm <= 10*tolerance
            ε = maximum(norm.(u .- u0))
            @stop(:CONV, true)
        end
        u = map(q -> chop(q, tolerance), u)
        
        if θ̄ == Inf
            verbose && @printf("-\n")
            continue
        end
        F = N(u...)
        Δū = J\F
        Δū_norm = maximum(norm.(components(Δū)))
        if Δū_norm <= tolerance && Δu_norm <= sqrt(10*tolerance)
            u = u .- Δū
            ε = maximum(norm.(u .- u0))
            @stop(:CONV, true)
        end
        θ = Δū_norm/Δu_norm
        if k == 1
            θ₁ = θ
            δ = Δu_norm
        end
        @stop(:DIVERGE, θ > θ̄) # Monotonicity Test
    end
    return u, θ₁, δ, ε, k, err, info
end

function continuation_deu(N, u0::Union{Fun,Array{T,1}}; a=0., b=1., direction=1.,
                                                        maxiterations=5000, tolerance=1e-15,
                                                        s0=Inf, stepmin=1E-6, stepmax=Inf, β=1/sqrt(2.),
                                                        pathmonitor=(y)->true, verbose=false) where T <: Fun
    vlist, k, err, info = _continuation_deu(N, u0; a=a, b=b, direction=direction,
                                                   maxiterations=maxiterations, tolerance=tolerance,
                                                   s0=s0, stepmin=stepmin, stepmax=stepmax, β=β,
                                                   pathmonitor=pathmonitor, verbose=verbose)
    if info == :MAXIT
        @warn "Continuation method reached maximum number of iterations $(k) with error $(err) and θ₁ $(θ₁)"
    elseif info == :DIVERGE
        @warn "Continuation method diverged at iteration $(k) with error $(err) and θ₁ $(θ₁)"
    elseif info == :MONITOR
        @warn "Continuation method stopped by monitor at iteration $(k) with error $(err) and θ₁ $(θ₁)"
    end
    return vlist
end

"""
`continuation` solves the equation `N(λ,u) = 0` with **one** unknown using a
continuation method.

N is a function of two variables, `λ` and `u`, where `λ` is a parameter ranging
from `a` to `b`. `u0` is the initial guess for the solution at `λ = a`.
"""
function _continuation_deu(N, u0::Fun; a=0., b=1., direction=1.,
                                       maxiterations=5000, tolerance=1e-15,
                                       s0=Inf, stepmin=1E-6, stepmax=Inf, β=1/sqrt(2.),
                                       pathmonitor=(y)->true, verbose=false)
    _continuation_deu(N, [u0]; a=a, b=b, direction=direction,
                               maxiterations=maxiterations, tolerance=tolerance,
                               s0=s0, stepmin=stepmin, stepmax=stepmax, β=β,
                               pathmonitor=pathmonitor, verbose=verbose)
end

"""
`continuation` solves the equation `N(λ,u₁,u₂,⋯) = 0` with **multiple** unknowns
using a continuation method.

N is a function of two variables, `λ` and `u`, where `λ` is a parameter ranging
from `a` to `b`. `u0` is the array of initial function guesses for the solution
at `λ = a`.
"""
function _continuation_deu(N, u0::Array{T,1}; a=0., b=1., direction=1.,
                                              maxiterations=5000, tolerance=1e-15,
                                              σ0=Inf, stepmin=1E-6, stepmax=Inf,
                                              pathmonitor=(y)->true, verbose=false) where T <: Fun
    @assert length(u0) > 0 "u0 must be a vector of length > 0"
    @assert b > a "b=$b must be greater than a=$a"
    p = 1 # tangetial predictor behaves here as order 1 predictor
    v = ParamFun(a, u0)
    vlist = [v]
    t = ParamFun(direction, [Fun(0) for i in 1:length(u0)])
    k, err, info = 0, Inf, :CONV
    σ, θ₁, δ, ε = min(σ0, b-a, stepmax), Inf, Inf, Inf
    while v.λ < b*(1-eps()) && info == :CONV
        @stop(:MAXIT, k >= maxiterations)
        t = tangent(N, v, t)
        verbose && @printf("λ = %.5f  |  step      θ₁        iter   err       info\n", v.λ)
        if k > 1
            if θ₁ > 0 && ε > 0
                σ = (δ/ε * (sqrt(2)-1)/(2*θ₁))^(1/p) * σ
            else
                σ = Inf
            end
            σ = max(stepmin, min(σ, stepmax))
        end
        vj = v
        while true
            if t.λ > 0
                σ = min(σ, (b-v.λ)/t.λ)
            end
            vj = v + σ*t
            verbose && @printf(("             |  %.2e  "), σ)
            uj, θ₁j, δj, εj, k_topath, err, info = topath((u...) -> N(vj.λ, u...), vj.u; θ̄=0.25, maxiterations=15, tolerance=tolerance)
            vj = ParamFun(vj.λ, uj)
            verbose && @printf("%.2e  %5d  %.2e  %s\n", θ₁j, k_topath, err, info)
            k += k_topath
            if !pathmonitor(v)
                info = :MONITOR
            end
            if info == :CONV
                v, θ₁, δ, ε = vj, θ₁j, δj, εj
                push!(vlist, v)
                break
            elseif σ == stepmin
                @stop(info, true)
            else
                σ = ((sqrt(2)-1)/(sqrt(1+4θ₁j)-1))^(1/p) * σ
                σ = max(stepmin, min(σ, stepmax))
            end
        end
    end
    return vlist, k, err, info
end

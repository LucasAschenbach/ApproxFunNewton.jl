export newton, continuation, plot

using ApproxFun
import ApproxFun: DualFun, Fun, chop, jacobian
import DomainSets: AbstractInterval
import LinearAlgebra: norm
import Plots: plot, plot!
using DualNumbers
using Printf
using Plots

function newton(N, u0::Union{Fun,Array{T,1}}; damped=false, λ0=1, λmin=1e-6, θ̄=nothing,
                                              maxiterations=15, tolerance=1E-15, verbose=false) where T <: Fun
    u, θ₁, k, err, info = nothing, Inf, 0, Inf, nothing
    if damped
        isnothing(θ̄) && (θ̄ = 1)
        u, θ₁, k, err, info = _newton_damped(N, u0; λ0=λ0, λmin=λmin, θ̄=θ̄, maxiterations=maxiterations, tolerance=tolerance, verbose=verbose)
    else
        isnothing(θ̄) && (θ̄ = 2)
        u, θ₁, k, err, info = _newton(N, u0, θ̄=θ̄, maxiterations=maxiterations, tolerance=tolerance, verbose=verbose)
    end
    if info == :MAXIT
        @warn "Newton's method reached maximum number of iterations $(k) with error $(err) and θ₁ $(θ₁)"
    elseif info == :DIVERGE
        @warn "Newton's method diverged at iteration $(k) with error $(err) and θ₁ $(θ₁)"
    end
    return u
end

macro stop(name, test)
    esc(:(if ($test) info = $name; break; end))
end

"""
`_newton` solves a system of equations `N(u) = 0` with **one** unknown `u` using
Newton's method.

`N` is a function that takes one `Fun` and returns a `Fun`.
`u0` is the initial function guess, and `θ̄` is the emergency stop tolerance.
"""
function _newton(N, u0::Fun; θ̄=2, maxiterations=15, tolerance=1E-15, verbose=false)
    u, θ₁, k, err, info = u0, Inf, 0, Inf, Nothing
    verbose && @printf("iter     |  F         err       θ\n")
    while true
        @stop(:MAXIT, k >= maxiterations)
        k += 1
        verbose && @printf("k = %3d  |  ", k)

        DF = N(DualFun(u))
        J = map(jacobian, DF); J = qr(J)
        F = map(d-> typeof(d) <: DualFun ? d.f : d, DF)
        verbose && @printf("%.2e  ", norm(coefficients.(collect(F))))
        Δu = J\F
        Δu_norm = err = norm(Δu)
        verbose && @printf("%.2e  ", err)
        u = u - Δu # Newton Update
        @stop(:CONV, Δu_norm <= 10*tolerance)
        u = chop(u, tolerance)

        if θ̄ == Inf
            verbose && @printf("-\n")
            continue
        end
        F = N(u)
        Δū = J\F
        Δū_norm = norm(Δū)
        if Δū_norm <= tolerance && Δu_norm <= sqrt(10*tolerance)
            u = u - Δū # Simplified Newton Update
            @stop(:CONV, true)
        end
        θ = Δū_norm/Δu_norm
        verbose && @printf("%.2e\n", θ)
        if k == 1
            θ₁ = θ
        end
        @stop(:DIVERGE, θ > θ̄) # Adaptive Monotonicity Test
    end
    return u, θ₁, k, err, info
end

"""
`_newton` solves a system of equation `N(u₁,u₂,⋯) = 0` with **multiple** unknowns
`u₁,u₂,⋯` using Newton's method.

`N` is a function that takes any number of
`Fun`s and returns a `Fun`. `u0` is the array of initial function guesses, and
`θ̄` is the emergency stop tolerance.
"""
function _newton(N, u0::Array{T,1}; θ̄=2, maxiterations=15, tolerance=1E-15, verbose=false) where T <: Fun
    u, θ₁, k, err, info = copy(u0), tolerance, 0, Inf, Nothing
    numf = length(u0) # number of functions
    if numf == 0
        error("u0 must contain at least 1 function")
    end
    Js = Array{Any}(undef,numf) # jacobians
    verbose && @printf("iter     |  F         err       θ\n")
    while true
        @stop(:MAXIT, k >= maxiterations)
        k += 1
        verbose && @printf("k = %3d  |  ", k)

        F = N([ (j == 1 ? DualFun(u[j]) : u[j]) for j = 1:numf ]...)
        @inbounds Js[1] = jacobian.(F)
        F = map(d -> typeof(d) <: DualFun ? d.f : d, F)
        verbose && @printf("%.2e  ", norm(coefficients.(collect(F))))
        for i = 2:numf
            @inbounds Js[i] = jacobian.(N([ (j == i ? DualFun(u[j]) : u[j]) for j = 1:numf ]...))
        end
        J = Base.typed_hcat(Operator, Js...); J = qr(J)

        Δu = J\F
        Δu_norm = err = maximum(norm.(components(Δu)))
        verbose && @printf("%.2e  ", err)
        u = u .- Δu
        @stop(:CONV, Δu_norm <= 10*tolerance)
        u = map(q -> chop(q, tolerance), u)
        
        if θ̄ == Inf
            verbose && @printf("-\n")
            continue
        end
        F = N(u...)
        Δū = J\F
        Δū_norm = maximum(norm.(components(Δū)))
        if Δū_norm <= tolerance && Δu_norm <= sqrt(10*tolerance)
            u = u .- Δū # Simplified Newton Step
            @stop(:CONV, true)
        end
        θ = Δū_norm/Δu_norm
        verbose && @printf("%.2e\n", θ)
        if k == 1
            θ₁ = θ
        end
        @stop(:DIVERGE, θ > θ̄) # Adaptive Monotonicity Test
    end
    return u, θ₁, k, err, info
end

"""
`_newton_damped` solves a system of equations `N(u) = 0` with **one** unknown `u` using
Newton's method with damping.

`N` is a function that takes one `Fun` and returns a `Fun`.
`u0` is the initial function guess, and `θ̄` is the monotonicity tolerance.
"""
function _newton_damped(N, u0::Fun; λ0=1, λmin=1e-6, θ̄=1, maxiterations=15, tolerance=1E-15, verbose=false)
    u, θ₁, k, err, info = u0, tolerance, 0, Inf, Nothing
    λ = max(λ0, λmin)
    Δu_norm_prev = Inf
    Δū = nothing
    Δū_norm = Inf
    θ = Inf
    verbose && @printf("iter      |  F         err       λ         θ\n")
    while true
        @stop(:MAXIT, k >= maxiterations)
        k += 1
        verbose && @printf("k = %4d  |  ", k)

        DF = N(DualFun(u))
        J = map(jacobian, DF); J = qr(J)
        F = map(d->d.f, DF)
        verbose && @printf("%.2e  ", norm(coefficients.(collect(F))))
        Δu = J\F
        Δu_norm = err = norm(Δu)
        verbose && @printf("%.2e  ", err)
        if Δu_norm <= tolerance
            u = u - Δu
            @stop(:CONV, true)
        end
        if k > 1
            # predictor (3.49)
            μ = Δu_norm_prev*Δū_norm/(norm(Δū - Δu)*Δu_norm * λ)
            λ = min(1, μ)
        end
        verbose && @printf("%.2e  ", λ)
        @stop(:DIVERGE, λ < λmin)
        j = 0
        while true
            j += 1
            if verbose && j > 1
                @printf("          |                      %.2e  ", λ)
            end
            v = u - λ*Δu
            v = chop(v, tolerance)

            F = N(v)
            Δū = J\F
            Δū_norm = norm(Δū)
            θ = Δū_norm/Δu_norm
            verbose && @printf("%.2e      \n", θ)
            if j == 1
                if k == 1
                    θ₁ = θ
                end
                if Δū_norm <= tolerance && Δu_norm <= sqrt(10*tolerance) && λ == 1
                    u = v - Δū # Simplified Newton Step
                    info = :CONV
                    # TODO: refactor, write with @stop
                    return u, θ₁, k, err, info
                end
            end

            # corrector (3.48)
            μ = Δu_norm/(2*norm(Δū-(1-λ)*Δu)) * λ^2
            λj = min(1, μ)

            if θ < θ̄ # Adaptive Monotonicity Test
                λ = λj
                Δu_norm_prev = Δu_norm
                u = v
                break
            elseif λ == λmin
                info = :DIVERGE
                return u, θ₁, k, err, info
            else
                λ = min(1/2*λ, λj)
                λ = max(λ, λmin)
            end
        end
    end
    return u, θ₁, k, err, info
end

"""
`_newton_damped` solves a system of equation `N(u₁,u₂,⋯) = 0` with **multiple** unknowns
`u₁,u₂,⋯` using Newton's method with damping.

`N` is a function that takes any number of
`Fun`s and returns a `Fun`. `u0` is the array of initial function guesses, and
`θ̄` is the monotonicity tolerance.
"""
function _newton_damped(N, u0::Array{T,1}; λ0=1, λmin=1e-6, θ̄=1, maxiterations=15, tolerance=1E-15, verbose=false) where T <: Fun
    u, θ₁, k, err, info = copy(u0), tolerance, 0, Inf, Nothing
    numf = length(u0) # number of functions
    @assert numf >= 1 "u0 must contain at least 1 function"
    λ = max(λ0, λmin)
    Δu_norm_prev = Inf
    Δū = nothing
    Δū_norm = Inf
    θ = Inf
    Js = Array{Any}(undef,numf) # jacobians
    verbose && @printf("iter      |  F         err       λ         θ\n")
    while true
        @stop(:MAXIT, k >= maxiterations)
        k += 1
        verbose && @printf("k = %4d  |  ", k)

        F = N([ (j == 1 ? DualFun(u[j]) : u[j]) for j = 1:numf ]...)
        @inbounds Js[1] = jacobian.(F)
        F = map(d -> typeof(d) <: DualFun ? d.f : d, F)
        verbose && @printf("%.2e  ", norm(coefficients.(collect(F))))
        for i = 2:numf
            @inbounds Js[i] = jacobian.(N([ (j == i ? DualFun(u[j]) : u[j]) for j = 1:numf ]...))
        end
        J = Base.typed_hcat(Operator, Js...); J = qr(J)

        Δu = J\F
        Δu_norm = err = maximum(norm.(Array(Δu)))
        verbose && @printf("%.2e  ", err)
        if Δu_norm <= tolerance
            u = u .- Δu
            @stop(:CONV, true)
        end

        if k > 1
            # predictor (3.49)
            μ = Δu_norm_prev*Δū_norm/(norm(Δū - Δu)*Δu_norm * λ)
            λ = min(1, μ)
        end
        verbose && @printf("%.2e  ", λ)
        @stop(:DIVERGE, λ < λmin)
        j = 0
        while true
            j += 1
            (verbose && j > 1) && @printf("          |                      %.2e  ", λ)
            v = u .- λ*Δu
            v = map(q -> chop(q, tolerance), v)

            F = N(v...)
            Δū = J\F
            Δū_norm = maximum(norm.(Array(Δū)))
            θ = Δū_norm/Δu_norm
            verbose && @printf("%.2e\n", θ)
            if j == 1
                if k == 1
                    θ₁ = θ
                end
                if Δū_norm <= tolerance && Δu_norm <= sqrt(10*tolerance) && λ == 1
                    u = v .- Δū
                    info = :CONV
                    return u, θ₁, k, err, info
                end
            end

            # corrector (3.48)
            μ = Δu_norm/(2*norm(Δū-(1-λ)*Δu)) * λ^2
            λj = min(1, μ)

            if θ < θ̄ # Natural Monotonicity Test
                λ = λj
                Δu_norm_prev = Δu_norm
                u = v
                break
            elseif λ == λmin
                info = :DIVERGE
                return u, θ₁, k, err, info
            else
                λ = min(1/2*λ, λj)
                λ = max(λ, λmin)
            end
        end
    end
    return u, θ₁, k, err, info
end

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
    Jus = Array{Any,1}(undef,length(u)) # jacobians
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

function topath(N, v::ParamFun{Float64}; θ̄=0.25, maxiterations=25, tolerance=1E-15)
    λ, u0 = v.λ, v.u
    u, θ₁, k, err, info = _newton((u...) -> N(λ, u...), u0; θ̄=θ̄, maxiterations=maxiterations, tolerance=tolerance, verbose=false)
    return ParamFun(λ, u), θ₁, k, err, info
end

function continuation(N, u0::Union{Fun,Array{T,1}}; a=0., b=1., direction=1.,
                                                    maxiterations=5000, tolerance=1E-15,
                                                    λ0=Inf, stepmin=1E-6, stepmax=Inf, β=1/sqrt(2.),
                                                    pathmonitor=(y)->true, verbose=false) where T <: Fun
    vlist, k, err, info = _continuation(N, u0; a=a, b=b, direction=direction,
                                               maxiterations=maxiterations, tolerance=tolerance,
                                               λ0=λ0, stepmin=stepmin, stepmax=stepmax, β=β,
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
function _continuation(N, u0::Fun; a=0., b=1., direction=1.,
                                   maxiterations=5000, tolerance=1E-15,
                                   λ0=Inf, stepmin=1E-6, stepmax=Inf, β=1/sqrt(2.),
                                   pathmonitor=(y)->true, verbose=false)
    _continuation(N, [u0]; a=a, b=b, direction=direction,
                           maxiterations=maxiterations, tolerance=tolerance,
                           λ0=λ0, stepmin=stepmin, stepmax=stepmax, β=β,
                           pathmonitor=pathmonitor, verbose=verbose)
end

"""
`continuation` solves the equation `N(λ,u₁,u₂,⋯) = 0` with **multiple** unknowns
using a continuation method.

N is a function of two variables, `λ` and `u`, where `λ` is a parameter ranging
from `a` to `b`. `u0` is the array of initial function guesses for the solution
at `λ = a`.
"""
function _continuation(N, u0::Array{T,1}; a=0., b=1., direction=1.,
                                          maxiterations=5000, tolerance=1E-15,
                                          λ0=Inf, stepmin=1E-6, stepmax=Inf, β=1/sqrt(2.),
                                          pathmonitor=(y)->true, verbose=false) where T <: Fun
    @assert length(u0) > 0 "u0 must be a vector of length > 0"
    @assert b > a "b=$b must be greater than a=$a"
    v = ParamFun(a, u0)
    vlist = [v]
    t₀ = ParamFun(direction, [Fun(0) for i in 1:length(u0)])
    s, k, err, info = min(λ0, b-a, stepmax), 0, Inf, :CONV
    while v.λ < b*(1-eps()) && info == :CONV
        @stop(:MAXIT, k >= maxiterations)
        t = tangent(N, v, t₀)
        v₀, t₀ = v, t
        verbose && @printf("λ = %.5f  |  step      θ₁        iter   err       info\n", v.λ)
        while true
            if t.λ > 0
                s = min(s, (b-v₀.λ)/t.λ)
            end
            v = v₀ + s*t
            verbose && @printf(("             |  %.6f  "), s)
            v, θ₁, k_topath, err, info = topath(N, v; θ̄=0.25, maxiterations=15, tolerance=tolerance)
            verbose && @printf("%.2e  %5d  %.2e  %s\n", θ₁, k_topath, err, info)
            k += k_topath
            if !pathmonitor(v)
                info = :MONITOR
            end
            if info == :CONV
                if θ₁ < 1/8
                    s = s/β # increase stepsize
                end
                s = min(s, stepmax)
                push!(vlist, v)
                break
            else
                s = β*s # decrease stepsize
                @stop(info, s < stepmin)
            end
        end
    end
    return vlist, k, err, info
end

function Plots.plot(vlist::Array{T,1}; vars::Union{Tuple{Int},Nothing}=nothing, kwargs...) where T <: ParamFun
    plt = plot(; kwargs...)
    dim = length(vlist[1].u)
    a, b = vlist[1].λ, vlist[end].λ
    palette = Plots.palette(:default)
    if vars === nothing
        vars = (1:dim)
    else
        for i in vars
            @assert 1 <= i <= dim "vars must be a tuple of integers between 1 and $(dim)"
        end
    end
    for v in vlist[1:end-1]
        λ, u = v.λ, v.u
        for i in vars
            ui = u[i]
            d = domain(ui)
            l, r = 0., 0.
            if isa(d, AbstractInterval)
                l, r = endpoints(d)
            else
                error("domain of type $(typeof(d)) not supported")
            end
            alpha = 0.5 + 0.25*(λ-a)/(b-a) # rescale to [0.5, 0.75]
            γ = 2.2 # gamma correction, sRGB space
            naturalAlpha = alpha^γ
            plot!(plt, ui, l, r, alpha=naturalAlpha, color=palette[i], label="λ=$(round(λ, digits=3))")
        end
    end
    v = vlist[end]
    λ, u = v.λ, v.u
    for i in vars
        ui = u[i]
        plot!(plt, ui, alpha=1, color=palette[i], label="λ=$(round(λ, digits=3))")
    end
    return plt
end

export plotpath

using Plots
using LaTeXStrings
using Interpolations
import ApproxFunNewton: ParamFun

function plotpath(vlist::AbstractArray{T,1}; vars::Tuple, kwargs...) where T <: ParamFun
    @assert length(vars) == 2 "vars must be a tuple of length 2"
    plt = plot(; kwargs...)
    λs = [v.λ for v in vlist]
    A = Array{Float64}(undef, length(vlist), 2)
    label = Array{String}(undef, 2)
    for j in 1:2
        if vars[j] == :λ
            label[j] = L"\lambda"
            A[:,j] = λs
        elseif vars[j] isa Tuple
            @assert length(vars[j]) == 2 "vars[$j] tuple must be of length 2"
            label[j] = L"u_{%$(vars[j][1])}[%$(vars[j][2])]"
            format(vals, index) = length(vals) >= index ? vals[index] : 0
            A[:,j] = [format(coefficients(v.u[vars[j][1]]), vars[j][2]) for v in vlist]
        else
            error("vars[$j] must be :λ or a tuple of length 2")
        end
    end
    x, y = A[:,1], A[:,2]

    itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), range(λs[1],λs[end],length(λs)), 1:2)
    λsfine = λs[1]:0.01:λs[end]
    xs, ys = [itp(λ,1) for λ in λsfine], [itp(λ,2) for λ in λsfine]
    
    plot!(plt, xs, ys; xlabel=label[1], ylabel=label[2])
    scatter!(plt, x, y)
    return plt
end

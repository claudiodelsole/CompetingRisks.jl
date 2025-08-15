# export functions
export summary_dishes, summary_theta, summary_kernelpars, summary_coefficients, summary_logevidence

"""
    summary_dishes(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 1000)

Traceplot and histogram of posterior samples for the number of dishes, i.e. occupied locations of the random measures. First argument `params::Parameters` is the output of [`posterior_sampling`](@ref).

See also [`summary_theta`](@ref), [`summary_kernelpars`](@ref), [`summary_coefficients`](@ref), [`summary_logevidence`](@ref).
"""
function summary_dishes(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 1000)

    # create Chains
    chn = Chains(hcat(params.dishes_number, params.accept_dishes), [:k, :accept])
    chn_post = chn[burn_in+thin:thin:end]

    # describe chains
    describe(chn_post)

    # traceplot
    pltrace = plot(chn[:k][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:k])[thin:thin:end]])
    hline!(pltrace, [mean(chn_post[:k])], linecolor = 2)
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "\$k\$", xtickfontcolor = :white)

    # histogram and density
    plhist = histogram(chn_post[:k], normalize = :probability, bins = range(1, maximum(chn_post[:k])))
    # density!(plhist, chn_post[:k], linewidth = 2, linecolor = 2)
    plot!(plhist, legend = false, xlabel = "\$k\$")

    # combine plots
    return (pltrace, plhist)

end # summary_dishes

"""
    summary_theta(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 1000)

Traceplot and histogram of posterior samples for the concentration parameter θ. First argument `params::Parameters` is the output of [`posterior_sampling`](@ref).

See also: [`summary_dishes`](@ref), [`summary_kernelpars`](@ref), [`summary_coefficients`](@ref), [`summary_logevidence`](@ref).
"""
function summary_theta(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 1000)

    # create Chains
    chn = Chains(params.theta, [:θ])
    chn_post = chn[burn_in+thin:thin:end]

    # describe chain
    describe(chn_post)

    # traceplot
    pltrace = plot(chn[:θ][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:θ])[thin:thin:end]])
    hline!(pltrace, [mean(chn_post[:θ])], linecolor = 2)
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "\$θ\$", xtickfontcolor = :white)

    # histogram
    plhist = histogram(chn_post[:θ], normalize = :pdf)
    density!(plhist, chn_post[:θ], linewidth = 2, linecolor = 2)
    plot!(plhist, legend = false, xlabel = "\$θ\$")

    # combine plots
    return (pltrace, plhist)

end # summary_theta

"""
    summary_kernelpars(params::Parameters, name::Symbol; thin::Int64 = 10, burn_in::Int64 = 1000)

Traceplot and histogram of posterior samples for the kernel parameter `name`. First argument `params::Parameters` is the output of [`posterior_sampling`](@ref).

See also [`summary_dishes`](@ref), [`summary_theta`](@ref), [`summary_coefficients`](@ref), [`summary_logevidence`](@ref).
"""
function summary_kernelpars(params::Parameters, name::Symbol; thin::Int64 = 10, burn_in::Int64 = 1000)

    # create Chains
    auxchn = Chains(reshape(params.kernelpars, length(params.accept_kernelpars), :), collect(fieldnames(params.KernelType)))
    chn = Chains(hcat(Array(auxchn), log.(Array(auxchn)), params.accept_kernelpars), vcat(names(auxchn), Symbol.("log", String.(names(auxchn))), :accept))
    chn_post = chn[burn_in+thin:thin:end]

    # describe chains
    describe(chn_post)

    # traceplot
    logname = Symbol("log", String(name))
    pltrace = plot(chn[logname][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[logname])[thin:thin:end]])
    hline!(pltrace, [mean(chn_post[logname])], linecolor = 2)
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "\$\\log($(String(name)))\$", xtickfontcolor = :white)

    # histogram
    plhist = histogram(chn_post[logname], normalize = :pdf)
    density!(plhist, chn_post[logname], linewidth = 2, linecolor = 2)
    plot!(plhist, legend = false, xlabel = "\$\\log($(String(name)))\$")

    # combine plots
    return (pltrace, plhist)

end # summary_kernelpars

"""
    summary_coefficients(params::Parameters, l::Int64; thin::Int64 = 10, burn_in::Int64 = 1000, hazard_ratio::Bool = false)

Traceplot and histogram of posterior samples for the Cox regression coefficient of level `l`. First argument `params::Parameters` is the output of [`posterior_sampling`](@ref). The flag `hazard_ratio` indicates whether consider the regression coefficient or the hazard ratio.

See also [`summary_dishes`](@ref), [`summary_theta`](@ref), [`summary_kernelpars`](@ref), [`summary_logevidence`](@ref).
"""
function summary_coefficients(params::Parameters, l::Int64; thin::Int64 = 10, burn_in::Int64 = 1000, hazard_ratio::Bool = false)

    # retrieve traces
    traces = cat(reshape(params.eta, length(params.dishes_number), :), reshape(exp.(params.eta), length(params.dishes_number), :), dims = 3)
    accept = reshape(params.accept_coeffs, length(params.dishes_number), :)

    # create Chains
    chn = Chains(hcat(traces[:,l,1], traces[:,l,2], accept[:,l]), [:η, :expη, :accept])
    chn_post = chn[burn_in+thin:thin:end]

    # describe chains
    describe(chn_post)

    # traceplot
    pltrace = plot(chn[hazard_ratio ? :expη : :η][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:accept])[thin:thin:end]])
    hline!(pltrace, [mean(chn_post[hazard_ratio ? :expη : :η])], linecolor = 2)
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = hazard_ratio ? "\$\\exp η\$" : "\$η\$", xtickfontcolor = :white)

    # histogram
    plhist = histogram(chn_post[hazard_ratio ? :expη : :η], normalize = :pdf)
    density!(plhist, chn_post[hazard_ratio ? :expη : :η], linewidth = 2, linecolor = 2)
    plot!(plhist, legend = false, xlabel = hazard_ratio ? "\$\\exp η\$" : "\$η\$")

    # combine plots
    return (pltrace, plhist)

end # summary_coefficient

"""
    summary_logevidence(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 1000)

Traceplot for the model logevidence. First argument `params::Parameters` is the output of [`posterior_sampling`](@ref).

See also [`summary_dishes`](@ref), [`summary_theta`](@ref), [`summary_kernelpars`](@ref), [`summary_coefficients`](@ref).
"""
function summary_logevidence(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 1000)

    # create Chains
    chn = Chains(params.logevidence, [:logevidence])

    # describe chain
    describe(chn[burn_in+thin:thin:end])

    # traceplot
    pltrace = plot(chn[:logevidence][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:loglik])[thin:thin:end]])
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "logevidence", xtickfontcolor = :white)

    # combine plots
    return pltrace

end # summary_logevidence

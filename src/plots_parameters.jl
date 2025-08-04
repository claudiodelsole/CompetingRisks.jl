# export functions
export summary_dishes, summary_theta, summary_gamma, summary_kappa, summary_coefficients, summary_loglikelihood

"""
    summary_dishes(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

"""
function summary_dishes(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

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
    summary_theta(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

"""
function summary_theta(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

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
    summary_gamma(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

"""
function summary_gamma(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

    # create Chains
    chn = Chains(hcat(params.logalpha, exp.(params.logalpha), params.accept_alpha), [:logγ, :γ, :accept])
    chn_post = chn[burn_in+thin:thin:end]

    # describe chains
    describe(chn_post)

    # traceplot
    pltrace = plot(chn[:logγ][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:logγ])[thin:thin:end]])
    hline!(pltrace, [mean(chn_post[:logγ])], linecolor = 2)
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "\$\\log(γ)\$", xtickfontcolor = :white)

    # histogram
    plhist = histogram(chn_post[:logγ], normalize = :pdf)
    density!(plhist, chn_post[:logγ], linewidth = 2, linecolor = 2)
    plot!(plhist, legend = false, xlabel = "\$\\log(γ)\$")

    # combine plots
    return (pltrace, plhist)

end # summary_gamma

"""
    summary_kappa(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

"""
function summary_kappa(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

    # create Chains
    chn = Chains(hcat(params.logkappa, exp.(params.logkappa), params.accept_kappa), [:logκ, :κ, :accept])
    chn_post = chn[burn_in+thin:thin:end]

    # describe chains
    describe(chn_post)

    # traceplot
    pltrace = plot(chn[:logκ][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:logκ])[thin:thin:end]])
    hline!(pltrace, [mean(chn_post[:logκ])], linecolor = 2)
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "\$\\log(κ)\$", xtickfontcolor = :white)

    # histogram
    plhist = histogram(chn_post[:logκ], normalize = :pdf)
    density!(plhist, chn_post[:logκ], linewidth = 2, linecolor = 2)
    plot!(plhist, legend = false, xlabel = "\$\\log(κ)\$")

    # combine plots
    return (pltrace, plhist)

end # summary_kappa

"""
    summary_coefficient(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100, l::Int64 = 1, hazard_ratio::Bool = false)

"""
function summary_coefficients(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100, l::Int64 = 1, hazard_ratio::Bool = false)

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
    summary_loglikelihood(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

"""
function summary_loglikelihood(params::Parameters; thin::Int64 = 10, burn_in::Int64 = 100)

    # create Chains
    chn = Chains(params.loglik, [:loglik])

    # describe chain
    describe(chn[burn_in+thin:thin:end])

    # traceplot
    pltrace = plot(chn[:loglik][thin:thin:end], linecolor = [i < burn_in ? :gray : 1 for i in eachindex(chn[:loglik])[thin:thin:end]])
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "loglikelihood", xtickfontcolor = :white)

    # combine plots
    return pltrace

end # summary_loglikelihood

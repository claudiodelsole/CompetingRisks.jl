# export functions
export traceplot_survival, traceplot_incidence, traceplot_proportions

"""
    traceplot_survival(estimator::Estimator, time::Float64; l::Int64 = 0)

Traceplot and empirical autocorrelation function for the estimator of the survival function at a given timepoint `time` and level `l` of the categorical predictor. First argument `estimator` is the output of [`posterior_sampling`](@ref).

See also [`traceplot_incidence`](@ref), [`traceplot_proportions`](@ref).
"""
function traceplot_survival(estimator::Estimator, time::Float64; l::Int64 = 0)

    # retrieve trace
    trace = reshape(estimator.survival_samples, length(estimator.times), estimator.L + 1, :)

    # create Chains
    t = sum(time .>= estimator.times)
    chn = Chains(vec(trace[t,l+1,:]), [:survival])

    # describe chain
    describe(chn)

    # traceplot
    pltrace = plot(chn[:survival])
    plot!(pltrace, legend = false, xlabel = "samples", ylabel = "\$S(t)\$", xtickfontcolor = :white)

    # autocorrelation plot
    plcor = plot(autocor(chn[:survival]), linecolor = 2)
    hline!(plcor, [0.0], linecolor = :black, linestyle = :dash)
    plot!(plcor, legend = false, ylims = (-0.1, 1.0), xlabel = "lag", ylabel = "ACF")

    # combine plots
    return (pltrace, plcor)

end # traceplot_survival

"""
    traceplot_incidence(estimator::Estimator, time::Float64; l::Int64 = 0)

Traceplots and empirical autocorrelation functions for the estimators of the incidence functions at a given timepoint `time` and level `l` of the categorical predictor. First argument `estimator` is the output of [`posterior_sampling`](@ref).

See also [`traceplot_survival`](@ref), [`traceplot_proportions`](@ref).
"""
function traceplot_incidence(estimator::Estimator, time::Float64; l::Int64 = 0)

    # labels and colors
    mycolors = reshape([d for d in range(1, estimator.D)], 1, :)
    mylabels = reshape(["cause " * string(d) for d in range(1, estimator.D)], 1, :)

    # reshape samples
    trace = reshape_samples(estimator.survival_samples, estimator.hazard_samples, length(estimator.times), estimator.L, estimator.D)

    # retrieve trace
    t = sum(time .>= estimator.times)
    chn = Chains(Matrix{Float64}(transpose(trace[t,l+1,:,:])), vec(mylabels))

    # describe chain
    describe(chn)

    # traceplot
    pltrace = plot(Array(chn), linecolor = mycolors, label = mylabels)
    plot!(pltrace, xlabel = "samples", ylabel = "\$f_δ(t)\$", xtickfontcolor = :white)

    # autocorrelation plot
    plcor = plot(autocor(Array(chn)), linecolor = mycolors, label = mylabels)
    hline!(plcor, [0.0], linecolor = :black, linestyle = :dash, label = false)
    plot!(plcor, ylims = (-0.1, 1.0), xlabel = "lag", ylabel = "ACF")

    # combine plots
    return (pltrace, plcor)

end # traceplot_incidence

"""
    traceplot_proportions(estimator::Estimator, time::Float64)

Traceplots and empirical autocorrelation functions for the estimators of the prediction curves (or relative hazards) at a given timepoint `time`. First argument `estimator` is the output of [`posterior_sampling`](@ref).

See also [`traceplot_survival`](@ref), [`traceplot_incidence`](@ref).
"""
function traceplot_proportions(estimator::Estimator, time::Float64)

    # labels and colors
    mycolors = reshape([d for d in range(1, estimator.D)], 1, :)
    mylabels = reshape(["cause " * string(d) for d in range(1, estimator.D)], 1, :)

    # reshape samples
    trace = reshape(copy(estimator.hazard_samples), length(estimator.times), estimator.L + 1, estimator.D, :)

    # normalize samples
    trace = mapslices(p -> p / sum(p), trace, dims = 3)

    # retrieve trace
    t = sum(time .>= estimator.times)
    chn = Chains(Matrix{Float64}(transpose(trace[t,1,:,:])), vec(mylabels))

    # describe chain
    describe(chn)

    # traceplot
    pltrace = plot(Array(chn), linecolor = mycolors, label = mylabels)
    plot!(pltrace, xlabel = "samples", ylabel = "\$p_n(δ \\vert t)\$", xtickfontcolor = :white)

    # autocorrelation plot
    plcor = plot(autocor(Array(chn)), linecolor = mycolors, label = mylabels)
    hline!(plcor, [0.0], linecolor = :black, linestyle = :dash, label = false)
    plot!(plcor, ylims = (-0.1, 1.0), xlabel = "lag", ylabel = "ACF")

    # combine plots
    return (pltrace, plcor)

end # traceplot_proportions

# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: Weibull
import StatsBase: counts
import Plots: plot, plot!, histogram

# set seed for reproducibility
seed!(24)

# include auxiliary files
include("../aux_code/create_datasets.jl")
include("../aux_code/functions.jl")
include("../aux_code/rbart.jl")
include("../aux_code/errors.jl")

##########
# Synthetic independent dataset
##########

# dimensions
N = 100     # number of patients
D = 2       # number of diseases

# define models
true_models = [Weibull(1.2), Weibull(2.4)]

# models summary
summary_models(true_models)

# create synthetic dataset
data = independent_dataset(N, true_models)

# times vector
times = collect(0.0:0.01:2.0)

# true survival function
survival_true = [prod([survival(model, t) for model in true_models]) for t in times]

# true incidence functions
incidence_true = [hazard(model, t) * prod([survival(model, t) for model in true_models]) for t in times, model in true_models]
cumincidence_true = mapslices(f -> integrate_trapz(f, times, cum = true), incidence_true, dims = 1)

# true proportions
proportions_true = [hazard(model, t) / sum([hazard(model, t) for model in true_models]) for t in times, model in true_models]

# true density function
density_true = vec(sum(incidence_true, dims = 2))

# data summary
println("Subjects per competing event: ", counts(data.Delta, D))
println("Maximum survival time: ", maximum(data.T))

# histogram for data summary
begin
    histogram(data.T, normalize = :pdf, bins = 20)
    plot!(times, density_true, linewidth = 2.0)
    plot!(xlim = (0.0, 1.8), xlabel = "\$t\$", ylabel = "\$f(t)\$", legend = false, size = (480,360))
end

##########
# Setup R
##########

# setup RCall
using RCall

# import required libraries
R"library(survival)"
R"library(cmprsk)"

# import data
@rput data
@rput times

##########
# Posterior sampling algorithm
##########

# kernel choice
CompetingRisks.kernel(x::Float64, t::Float64, _::Nothing, kappa::Float64) = CompetingRisks.kernel_DL(x, t, nothing, kappa)
CompetingRisks.KernelInt(x::Float64, t::Float64, _::Nothing, kappa::Float64) = CompetingRisks.KernelInt_DL(x, t, nothing, kappa)

# resampling step
CompetingRisks.resample_kappa(_::Restaurants) = (0.0, false)

# create Restaurants
rf = Restaurants(data, sigma = 0.25, sigma0 = 0.25)

# chain parameters
nsamples = 2000

# setup acceptance rates
CompetingRisks.stdev_dishes[] = 0.5

# run chain
marginal_estimator, conditional_estimator, params = posterior_sampling(rf, nothing, nsamples, times = times, burn_in = 5000)

##########
# Diagnostics
##########

# number of dishes
(pltrace, plhist) = summary_dishes(params, burn_in = 5000)

# base measure mass
(pltrace, plhist) = summary_theta(params, burn_in = 5000)

# kernel parameter
(pltrace, plhist) = summary_gamma(params, burn_in = 5000)

##########
# Posterior estimates: survival function
##########

# frequentist estimate
R"fit <- survfit(Surv(T, Delta > 0) ~ 1, data = data)"
R"km <- summary(fit, times)$surv"

# retrieve estimate
survival_freq = vcat(rcopy(R"km"), zeros(Float64, length(times) - length(rcopy(R"km"))))

# posterior estimates
(survival_post, _, _) = estimate_survival(marginal_estimator)
(_, survival_lower, survival_upper) = estimate_survival(conditional_estimator)

# plots 
plot_survival(times, survival_post, survival_true = survival_true, kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)
plot!(xlim = (0.0,1.3), size = (480,360))

##########
# Posterior estimates: cumulative incidence functions
##########

# frequentist estimate
R"fit <- cuminc(data$T, data$Delta)"
R"aj = timepoints(fit, times)$est"

# retrieve estimate
cumincidence_freq = Matrix(transpose(rcopy(R"aj")))
cumincidence_freq = mapslices(values -> map(x -> ismissing(x) ? maximum(skipmissing(values)) : x, values), cumincidence_freq, dims = 1)

# posterior estimates
(cumincidence_post, _, _) = estimate_incidence(marginal_estimator, cum = true)
(_, cumincidence_lower, cumincidence_upper) = estimate_incidence(conditional_estimator, cum = true)

# plots
plot_incidence(times, cumincidence_post, cum = true, incidence_true = cumincidence_true, aalen_johansen = cumincidence_freq, lower = cumincidence_lower, upper = cumincidence_upper)
plot!(size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.65))

##########
# BART for competing risks
##########

# import BART library
R"library(BART)"

# discretize times
R"data$Tr <- pmax(0.01, round(data$T, digits = 2))"

# BART competing risks model
R"post <- crisk.bart(times = data$Tr, delta = data$Delta)"

##########
# Posterior estimates: survival function
##########

# posterior estimate
(survival_post_bart, survival_lower_bart, survival_upper_bart) = estimate_survival_bart(rcopy(R"post$times"), rcopy(R"post$surv.test"), times)

# plots
plot_survival(times, survival_post_bart, survival_true = survival_true, kaplan_meier = survival_freq, lower = survival_lower_bart, upper = survival_upper_bart)
plot!(xlim = (0.0, 1.3), size = (480,360))

##########
# Posterior estimates: cumulative incidence functions
##########

# posterior estimates
(cumincidence_post_bart, cumincidence_lower_bart, cumincidence_upper_bart) = estimate_incidence_bart(rcopy(R"post$times"), rcopy(R"post$cif.test"), rcopy(R"post$cif.test2"), times)

# plots
plot_incidence(times, cumincidence_post_bart, cum = true, incidence_true = cumincidence_true, aalen_johansen = cumincidence_freq, lower = cumincidence_lower_bart, upper = cumincidence_upper_bart)
plot!(size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.65))

##########
# Comparison: survival function
##########

# errors
println("frequentist:\t", error_survival(survival_freq, survival_true))
println("BNP marginal:\t", error_survival(survival_post, survival_true))
println("BART:\t", error_survival(survival_post_bart, survival_true))

# plots
begin
    
    # initialize plot
    pl = plot(ylim = (0.0, 1.0), xlabel = "\$t\$", ylabel = "\$S(t)\$") 
    plot!(xlim = (0.0, 1.3), size = (480,360))

    # plot frequentist estimate
    plot!(pl, times, survival_true, linecolor = 3, label = "true")
    plot!(pl, times, survival_freq, linecolor = :black, label = "freq")

    # plot hCRM estimate
    plot!(pl, times, survival_post, linecolor = 1, label = "hCRM")
    plot!(pl, times, survival_lower, fillrange = survival_upper, linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(pl, times, survival_post_bart, linecolor = 2, label = "BART")
    plot!(pl, times, survival_lower_bart, fillrange = survival_upper_bart, linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

end

##########
# Comparison: cumulative incidence functions
##########

# errors
println("frequentist:\t", error_cumincidence(cumincidence_freq, cumincidence_true))
println("BNP marginal:\t", error_cumincidence(cumincidence_post, cumincidence_true))
println("BART:\t", error_cumincidence(cumincidence_post_bart, cumincidence_true))

# plots
begin

    # initialize plot
    pl = plot(xlabel = "\$t\$", ylabel = "\$F_\\delta(t)\$")
    plot!(size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.65))

    # plot frequentist estimate
    plot!(pl, times, cumincidence_true, linecolor = 3, linestyle = [:solid :dash], label = ["true" false])
    plot!(pl, times, cumincidence_freq, linecolor = :black, linestyle = [:solid :dash], label = ["freq" false])

    # plot hCRM estimate
    plot!(pl, times, cumincidence_post, linecolor = 1, linestyle = [:solid :dash], label = ["hCRM" false])
    plot!(pl, times, cumincidence_lower, fillrange = cumincidence_upper, linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(pl, times, cumincidence_post_bart, linecolor = 2, linestyle = [:solid :dash], label = ["BART" false])
    plot!(pl, times, cumincidence_lower_bart, fillrange = cumincidence_upper_bart, linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

end

# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: Weibull
import StatsBase: counts
import Statistics: mean, quantile
import Plots: plot, plot!, vline!, histogram, savefig

# set seed for reproducibility
seed!(24)

# include auxiliary files
include("../aux_code/functions.jl")
include("../aux_code/create_datasets.jl")
include("../aux_code/errors.jl")

##########
# Synthetic independent dataset
##########

# dimensions
N = 300     # number of patients
D = 3       # number of diseases

# define models
true_models = [Weibull(1.2), Weibull(1.6), Weibull(2.4)]

# models summary
(plincidence, plprop) = summary_models(true_models)
plot!(plincidence, size = (480,360))
plot!(plprop, size = (480,360))

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
println("# Data summary")
println("Subjects per competing event: ", counts(data.Delta, D))
println("Maximum survival time: ", maximum(data.T))

# histogram for data summary
histogram(data.T, normalize = :pdf, bins = 20)
plot!(times, density_true, linewidth = 2.0)
plot!(xlim = (0.0, 1.8), xlabel = "\$t\$", ylabel = "\$f(t)\$", legend = false, size = (480,360))
savefig("figures_supp/histogram.svg")

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

# create CompetingRisksModel
# cmprsk = CompetingRisksModel(DykstraLaudKernel)
cmprsk = CompetingRisksModel(DykstraLaudKernel, sigma = 0.25, sigma0 = 0.25)

# setup acceptance rates
stdevs(dishes = 0.5)

# run chain
# @profview posterior_sampling(data, cmprsk, nsamples = 2000, times = times, burn_in = 5000)
marginal_estimator, posterior_estimator, params = posterior_sampling(data, cmprsk, nsamples = 2000, times = times, burn_in = 5000)

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
(_, survival_lower, survival_upper) = estimate_survival(posterior_estimator)

# errors
println("# Estimation errors")
println("frequentist:\t", error_survival(survival_freq, survival_true))
println("BNP marginal:\t", error_survival(survival_post, survival_true))

# plots
plot_survival(times, survival_post, survival_true = survival_true, kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)
plot!(xlim = (0.0,1.3), size = (480,360))
savefig("figures/survival.svg")

##########
# Posterior estimates: incidence functions
##########

# posterior estimates
(incidence_post, _, _) = estimate_incidence(marginal_estimator)
(_, incidence_lower, incidence_upper) = estimate_incidence(posterior_estimator)

# plots
plot_incidence(times, incidence_post, incidence_true = incidence_true, lower = incidence_lower, upper = incidence_upper)
plot!(size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.9))
savefig("figures/incidence.svg")

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
(_, cumincidence_lower, cumincidence_upper) = estimate_incidence(posterior_estimator, cum = true)

# plots
plot_incidence(times, cumincidence_post, cum = true, incidence_true = cumincidence_true, aalen_johansen = cumincidence_freq, lower = cumincidence_lower, upper = cumincidence_upper)
plot!(size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.47))
savefig("figures/cumincidence.svg")

##########
# Posterior estimates: prediction curves
##########

# posterior estimates
(proportions_post, proportions_lower, proportions_upper) = estimate_proportions(marginal_estimator)

# plots
plot_proportions(times, proportions_post, proportions_true = proportions_true, lower = proportions_lower, upper = proportions_upper)
plot!(size = (480,360), xlim = (0.0,1.3), ylim = (0.0,1.0))
vline!([quantile(data.T, 0.95)], linestyle = :dashdot, linecolor = :black, linealpha = 0.5, label = false)
savefig("figures/prediction.svg")

##########
# Diagnostics
##########

println("# Model hyperparameters")

# number of dishes
(pltrace, plhist) = summary_dishes(params, burn_in = 5000)
plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (0.0, 50.0))
savefig("figures_supp/illustration_dishes_t.svg")
plot(plhist, size = (480,360), xlim = (0.0, 50.0))
savefig("figures_supp/illustration_dishes_p.svg")

# base measure mass
(pltrace, plhist) = summary_theta(params, burn_in = 5000)
plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (0.0, 24.0))
savefig("figures_supp/illustration_theta_t.svg")
plot(plhist, size = (480,360), xlim = (0.0, 24.0))
savefig("figures_supp/illustration_theta_p.svg")

# kernel parameter
(pltrace, plhist) = summary_kernelpars(params, :γ, burn_in = 5000)
plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (-3.0, 4.0))
savefig("figures_supp/illustration_gamma_t.svg")
plot(plhist, size = (480,360), xlim = (-3.0, 4.0))
savefig("figures_supp/illustration_gamma_p.svg")

println("# Functionals")

# traceplots for survival function
(pltrace, plcor) = traceplot_survival(marginal_estimator, 0.4)
# (pltrace, plcor) = traceplot_survival(posterior_estimator, 0.4)
plot(pltrace, size = (480,360), xlim = (0,2000), ylim = (0.47, 0.57))
savefig("figures_supp/illustration_survival_t.svg")
plot(plcor, size = (480,360), xlim = (0.0,33.0))
savefig("figures_supp/illustration_survival_ac.svg")

# traceplots for incidence functions
(pltrace, plcor) = traceplot_incidence(marginal_estimator, 0.4)
# (pltrace, plcor) = traceplot_incidence(posterior_estimator, 0.4)
plot(pltrace, size = (480,360), xlim = (0,2000), ylim = (0.0, 0.66))
savefig("figures_supp/illustration_incidence_t.svg")
plot(plcor, size = (480,360), xlim = (0.0,33.0))
savefig("figures_supp/illustration_incidence_ac.svg")

# traceplots for prediction curves
# (pltrace, plcor) = traceplot_proportions(marginal_estimator, 0.4)
# (pltrace, plcor) = traceplot_proportions(posterior_estimator, 0.4)
# plot(pltrace, size = (480,360), xlim = (0,2000), ylim = (0.0, 1.0))
# plot(plcor, size = (480,360), xlim = (0.0,33.0))

# effective sample size
times_grid = collect(0.1:0.1:2.0)
ess_grid = [ess_survival(marginal_estimator, time) for time in times_grid]
println("# Effective sample sizes")
println("ESS on grid: ", string(mean(ess_grid)))

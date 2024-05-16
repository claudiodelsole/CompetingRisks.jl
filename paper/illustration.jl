# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: Weibull
import StatsBase: quantile
import Plots: plot!, vline!

# set seed for reproducibility
seed!(16)

##########
# Synthetic independent dataset
##########

# dimensions
N = 100     # number of patients
D = 2       # number of diseases

# define models
true_models = [Weibull(1.5), Weibull(2.5)]

# models summary
summary_models(true_models)

# create synthetic dataset
crd = independent_dataset(N, true_models)

# data summary
summary_data(crd)

# times vector
upper_time = 2.0
times = Vector{Float64}(0.0:0.01:upper_time)

# true hazard functions
hazard_true = [hazard(model, t) for t in times, model in true_models]
cumhazard_true = mapslices(f -> integrate_trapz(f, times; cum = true), hazard_true; dims = 1)

# true survival function
survival_true = [prod([survival(model, t) for model in true_models]) for t in times]

# true incidence functions
incidence_true = [hazard(model, t) * prod([survival(model, t) for model in true_models]) for t in times, model in true_models]
cumincidence_true = mapslices(f -> integrate_trapz(f, times; cum = true), incidence_true; dims = 1)

# true proportions
proportion_true = mapslices(p -> p / sum(p), hazard_true; dims = 2)

# true density function
density_true = vec(sum(incidence_true, dims = 2))

##########
# Setup kernel choice   # !! TO BE IMPROVED !!
##########

# kernel choice
CompetingRisks.kernel(x::Float64, t::Float64, eta::Float64) = CompetingRisks.kernel_DL(x, t, eta)
CompetingRisks.kernel(x::Float64, t::Float64, cp::Float64, eta::Float64) = CompetingRisks.kernel_DL(x, t, cp, eta)

# KernelInt choice
CompetingRisks.KernelInt(x::Float64, t::Float64, eta::Float64) = CompetingRisks.KernelInt_DL(x, t, eta)
CompetingRisks.KernelInt(x::Float64, t::Float64, cp::Float64, eta::Float64) = CompetingRisks.KernelInt_DL(x, t, cp, eta)

# resampling step
CompetingRisks.resample_eta(_::Union{RestaurantFranchise,RestaurantArray}) = (0.0, false)

##########
# Gibbs sampling algorithm
##########

# create RestaurantFranchise
# rf = RestaurantFranchise(crd)
rf = RestaurantFranchise(crd; sigma = 0.25, sigma0 = 0.25)

# initialize marginal estimators
survival_est = SurvivalMarginal(rf, times)
incidence_est = IncidenceMarginal(rf, times)

# initialize conditional estimators
hazard_est_ = HazardConditional(times, D)
survival_est_ = SurvivalConditional(times)

# estimators vector
marginal_estimators = [survival_est, incidence_est]
conditional_estimators = [hazard_est_, survival_est_]

# chain parameters
num_samples = (n = 5000, m = 100)
burn_in = 50000

# run chain
dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in)

##########
# Diagnostics
##########

# diagnostics output
summary_dishes(dgn, burn_in)
summary_theta(dgn, burn_in)
summary_alpha(dgn, burn_in)
summary_loglikelihood(dgn, burn_in)

# traceplots (marginal)
traceplots(survival_est, 1.0)
traceplots(incidence_est, survival_est, 1.0)

# effective sample size
for t in 0.0:0.1:2.0
    traceplots(survival_est, t)
end

##########
# Posterior estimates: survival function
##########

# frequentist estimate
survival_freq = kaplan_meier(crd, times).estimate
# (survival_freq, survival_freq_lower, survival_freq_upper) = kaplan_meier(crd, times)

# posterior estimate (marginal)
survival_post = estimate(survival_est)

# credible intervals (marginal)
(survival_lower, survival_upper) = credible_intervals(survival_est; lower = 0.05, upper = 0.95)

# plots (marginal)
survival_plot(times, survival_post; survival_true = survival_true, kaplan_meier = survival_freq)
plot!(title = "", xlim = (0.0, 1.6), size = (480,360))

# errors (marginal)
survival_errorplot(times, survival_post, survival_true, density_true; kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)

# posterior estimate (conditional)
survival_post = estimate(survival_est_)

# credible intervals (conditional)
(survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)

# plots (conditional)
survival_plot(times, survival_post; survival_true = survival_true, kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.6))

# errors (conditional)
survival_errorplot(times, survival_post, survival_true, density_true; kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)

##########
# Posterior estimates: incidence functions
##########

# frequentist estimate
cumincidence_freq = aalen_johansen(crd, times)

# posterior estimates (conditional)
incidence_post = estimate(hazard_est_, survival_est_)
cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)

# credible intervals (conditional)
(incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
(cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)

# plots (incidence)
incidence_plot(times, incidence_post; incidence_true = incidence_true, lower = incidence_lower, upper = incidence_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.6), ylim = (0.0,1.0))

# plots (cumincidence)
incidence_plot(times, cumincidence_post;  cum = true, incidence_true = cumincidence_true, aalen_johansen = cumincidence_freq, lower = cumincidence_lower, upper = cumincidence_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.6), ylim = (0.0,0.65))

##########
# Posterior estimates: proportions
##########

# posterior estimates (conditional)
proportion_post = estimate(hazard_est_, prop = true)

# credible intervals (conditional)
(proportion_lower, proportion_upper) = credible_intervals(hazard_est_; prop = true, lower = 0.05, upper = 0.95)

# plots
proportion_plot(times, proportion_post; proportion_true = proportion_true, lower = proportion_lower, upper = proportion_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.6), ylim = (0.0,1.0))
vline!([maximum(crd.T)], linestyle = :dashdot, linecolor = :black, linealpha = 0.5, label = false)

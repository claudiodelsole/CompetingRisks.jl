# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: Weibull
import Statistics: mean, quantile
import Plots: plot, plot!, vline!, histogram!

# set seed for reproducibility
seed!(24)

##########
# Synthetic independent dataset
##########

# dimensions
N = 300     # number of patients
D = 3       # number of diseases

# define models
true_models = [Weibull(1.2), Weibull(1.6), Weibull(2.4)]

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
# hazard_true = [hazard(model, t) for t in times, model in true_models]
# cumhazard_true = mapslices(f -> integrate_trapz(f, times; cum = true), hazard_true; dims = 1)

# true survival function
survival_true = [prod([survival(model, t) for model in true_models]) for t in times]

# true incidence functions
incidence_true = [hazard(model, t) * prod([survival(model, t) for model in true_models]) for t in times, model in true_models]
cumincidence_true = mapslices(f -> integrate_trapz(f, times; cum = true), incidence_true; dims = 1)

# true proportions
proportion_true = [hazard(model, t) / sum([hazard(model, t) for model in true_models]) for t in times, model in true_models]

# true density function
density_true = vec(sum(incidence_true, dims = 2))

# histogram for data summary
plot(legend = false, size = (480,360))
histogram!(crd.T, normalize = :pdf, bins = 20)
plot!(times, density_true, linewidth = 2.0)
plot!(xlim = (0.0, 1.8))

##########
# Setup kernel choice
##########

# kernel choice
CompetingRisks.kernel(x::Float64, t::Float64, eta::Float64) = CompetingRisks.kernel_DL(x, t, eta)

# KernelInt choice
CompetingRisks.KernelInt(x::Float64, t::Float64, eta::Float64) = CompetingRisks.KernelInt_DL(x, t, eta)

# resampling step
CompetingRisks.resample_eta(_::Union{RestaurantFranchise,RestaurantArray}) = (0.0, false)

##########
# Gibbs sampling algorithm
##########

# create RestaurantFranchise
rf = RestaurantFranchise(crd; sigma = 0.25, sigma0 = 0.25)

# initialize marginal estimators
survival_est = SurvivalMarginal(rf, times)
incidence_est = IncidenceMarginal(rf, times)
marginal_estimators = [survival_est, incidence_est]

# initialize conditional estimators
hazard_est_ = HazardConditional(times, D)
survival_est_ = SurvivalConditional(times)
conditional_estimators = [hazard_est_, survival_est_]

# chain parameters
num_samples = (n = 5000, m = 20)
burn_in = 25000

# setup acceptance rates
CompetingRisks.mhdev_dishes[] = 0.5

# run chain
dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in)

##########
# Diagnostics
##########

# number of dishes
plot(summary_dishes(dgn)[1], title = "", size = (480,360), xlim = (0,75000), ylim = (0.0, 55.0), 
    xlabel = "samples", ylabel = "k", xtickfontcolor=:white)
plot(summary_dishes(dgn)[2], title = "", size = (480,360), xlim = (0.0, 55.0), xlabel = "k")

# base measure mass
plot(summary_theta(dgn)[1], title = "", size = (480,360), xlim = (0,75000), ylim = (0.0, 40.0), 
    xlabel = "samples", ylabel = "θ", xtickfontcolor=:white)
plot(summary_theta(dgn)[2], title = "", size = (480,360), xlim = (0.0, 40.0), xlabel = "θ")

# kernel parameter
plot(summary_alpha(dgn)[1], title = "", size = (480,360), xlim = (0,75000), ylim = (-2.5, 3.5), 
    xlabel = "samples", ylabel = "log(γ)", xtickfontcolor=:white)
plot(summary_alpha(dgn)[2], title = "", size = (480,360), xlim = (-2.5, 3.5), xlabel = "log(γ)")

# traceplots (marginal) for survival function
plot(traceplots(survival_est, 0.4)[1], title = "", size = (480,360), xlim = (0,5000), ylim = (0.47, 0.57), 
    xlabel = "samples", xtickfontcolor=:white)
plot(traceplots(survival_est, 0.4)[2], title = "", size = (480,360), xlim = (0.0,33.0), xlabel = "lag")

# traceplots (marginal) for incidence functions
plot(traceplots(incidence_est, survival_est, 0.4)[1], title = "", size = (480,360), xlim = (0,5000), ylim = (0.05, 0.75), 
    xlabel = "samples", xtickfontcolor=:white)
plot(traceplots(incidence_est, survival_est, 0.4)[2], title = "", size = (480,360), xlim = (0.0,33.0), xlabel = "lag")

# effective sample size
times_grid = 0.1:0.1:2.0
ess_grid = zeros(length(times_grid))
for (t, time) in enumerate(times_grid)
    ess_grid[t] = ess(survival_est, time)
end
println("ESS on grid: ", string(mean(ess_grid)))

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
plot!(title = "", xlim = (0.0,1.3), size = (480,360))

# errors (marginal)
survival_errorplot(times, survival_post, survival_true, density_true; kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)

# posterior estimate (conditional)
survival_post = estimate(survival_est_)

# credible intervals (conditional)
(survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)

# plots (conditional)
survival_plot(times, survival_post; survival_true = survival_true, kaplan_meier = survival_freq, lower = survival_lower, upper = survival_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.3))

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
plot!(title = "", size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.9))

# plots (cumincidence)
incidence_plot(times, cumincidence_post; cum = true, incidence_true = cumincidence_true, aalen_johansen = cumincidence_freq, lower = cumincidence_lower, upper = cumincidence_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.3), ylim = (0.0,0.45))

# errors (conditional)
incidence_errorplot(times, incidence_post, incidence_true; lower = incidence_lower, upper = incidence_upper)
incidence_errorplot(times, cumincidence_post, cumincidence_true; cum = true, aalen_johansen = cumincidence_freq, lower = cumincidence_lower, upper = cumincidence_upper)

##########
# Posterior estimates: proportions
##########

# posterior estimates (marginal)
proportion_post = estimate(incidence_est)

# credible intervals (marginal)
(proportion_lower, proportion_upper) = credible_intervals(incidence_est; lower = 0.05, upper = 0.95)

# plots
proportion_plot(times, proportion_post; proportion_true = proportion_true, lower = proportion_lower, upper = proportion_upper)
plot!(title = "", size = (480,360), xlim = (0.0,1.3), ylim = (0.0,1.0))
vline!([quantile(crd.T,0.95)], linestyle = :dashdot, linecolor = :black, linealpha = 0.5, label = false)

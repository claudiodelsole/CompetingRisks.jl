# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: quantile
import Statistics: mean, quantile
import CSV: CSV
import Plots: plot, plot!, vline!

##########
# Melanoma survival dataset
##########

# set seed for reproducibility
seed!(42)

# load dataset
data = CSV.File("paper/melanoma.txt")

# retrieve observed variables
T = Float64.(data.days)         # observations, time-to-event
Delta = data.status             # event causes
predictors = data.sex           # categorical predictors

# map event causes codes
Delta[Delta.==2] .= 0       # censored observations
Delta[Delta.==3] .= 2       # deaths from other causes

# convert days to years
T = T./365.25

# create synthetic dataset
crd = CompetingRisksDataset(T, Delta, predictors)

# data summary
summary_data(crd)

# times vector
# upper_time = quantile(crd.T, 0.99)
upper_time = 12.0
times = Vector{Float64}(0.0:0.02:upper_time)

##########
# Setup kernel choice 
##########

# kernel choice
CompetingRisks.kernel(x::Float64, t::Float64, cp::Float64, eta::Float64) = CompetingRisks.kernel_OU(x, t, cp, eta)

# KernelInt choice
CompetingRisks.KernelInt(x::Float64, t::Float64, cp::Float64, eta::Float64) = CompetingRisks.KernelInt_OU(x, t, cp, eta)

# resampling step
CompetingRisks.resample_alpha(_::Union{RestaurantFranchise,RestaurantArray}) = (0.0, false)

##########
# Gibbs sampling algorithm
##########

# create RestaurantFranchise
# rf = RestaurantFranchise(crd)
rf = RestaurantFranchise(crd; sigma = 0.25, sigma0 = 0.25)

# create RestaurantArray
# rf = RestaurantArray(crd)
# rf = RestaurantArray(crd; sigma = 0.25)

# create CoxModel
cm = CoxModel(crd)

# initialize marginal estimators
survival_est = SurvivalMarginal(rf, cm, times)
incidence_est = IncidenceMarginal(rf, cm, times)
marginal_estimators = [survival_est, incidence_est]

# initialize conditional estimators
hazard_est_ = HazardConditional(times, crd.D; cm.L)
survival_est_ = SurvivalConditional(times; cm.L)
conditional_estimators = [hazard_est_, survival_est_]

# chain parameters
num_samples = (n = 5000, m = 20)
burn_in = 50000

# setup acceptance rates
CompetingRisks.mhdev_dishes[] = 0.5

# run chain
dgn = Gibbs(rf, cm, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in, thin = 20)

##########
# Diagnostics
##########

# number of dishes
plot(summary_dishes(dgn)[2], title = "", size = (480,360), xlim = (0.0, 13.0), xlabel = "k")

# base measure mass
plot(summary_theta(dgn)[2], title = "", size = (480,360), xlim = (0.0, 6.5), xlabel = "θ")

# kernel shape parameter
plot(summary_eta(dgn)[2], title = "", size = (480,360), xlim = (-8.5, 1.5), xlabel = "log(κ)")

# regression coefficient
plot(summary_coefficients(dgn)[2], title = "", size = (480,360), xlim = (-0.7, 1.8), xlabel = "η")

##########
# Posterior estimates: hazard rate ratio
##########

# retrieve samples
coeffs_samples = dgn.xi[burn_in+1:end]
coeffs_samples = exp.(coeffs_samples)

# posterior estimate
println("--- Hazard rate ratio ---")
println("Posterior mean: ", string(mean(coeffs_samples)))
println("Credible bands: ", string([quantile(coeffs_samples, 0.05), quantile(coeffs_samples, 0.95)]))

##########
# Posterior estimates: survival function
##########

# frequentist estimate
survival_freq = kaplan_meier(crd, times).estimate
# (survival_freq, survival_freq_lower, survival_freq_upper) = kaplan_meier(crd, times)

# posterior estimate (conditional)
survival_post = estimate(survival_est_)

# credible intervals (conditional)
(survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)

# plots
survival_plot(times, survival_post[:,1]; kaplan_meier = survival_freq[:,1], lower = survival_lower[:,1], upper = survival_upper[:,1])
survival_plot(times, survival_post[:,2]; kaplan_meier = survival_freq[:,2], lower = survival_lower[:,2], upper = survival_upper[:,2])
plot!(title = "", xlim = (0.0, 10.5), size = (480,360))

##########
# Posterior estimates: incidence functions
##########

# frequentist estimate
cumincidence_freq = aalen_johansen(crd, times)

# posterior estimates (conditional)
# incidence_post = estimate(hazard_est_, survival_est_)
cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)

# credible intervals (conditional)
# (incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
(cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)

# plots (cumincidence)
incidence_plot(times, cumincidence_post[:,1,:]; cum = true, aalen_johansen = cumincidence_freq[:,1,:], lower = cumincidence_lower[:,1,:], upper = cumincidence_upper[:,1,:])
incidence_plot(times, cumincidence_post[:,2,:]; cum = true, aalen_johansen = cumincidence_freq[:,2,:], lower = cumincidence_lower[:,2,:], upper = cumincidence_upper[:,2,:])
plot!(title = "", size = (480,360), xlim = (0.0,10.5), ylim = (0.0,0.55))

##########
# Posterior estimates: prediction curves
##########

# posterior estimates (marginal)
proportion_post = estimate(incidence_est)

# credible intervals (marginal)
(proportion_lower, proportion_upper) = credible_intervals(incidence_est; lower = 0.05, upper = 0.95)

# plots
proportion_plot(times, proportion_post[:,1,:]; lower = proportion_lower[:,1,:], upper = proportion_upper[:,1,:])
plot!(title = "", size = (480,360), xlim = (0.0,10.5), ylim = (0.0,1.0))
vline!([maximum(crd.T[crd.Delta.!=0])], linestyle = :dashdot, linecolor = :black, linealpha = 0.5, label = false)

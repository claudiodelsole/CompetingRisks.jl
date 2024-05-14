# load module
using CompetingRisks

# imports
import Distributions: quantile
import CSV: CSV

##########
# Melanoma survival dataset
##########

# set seed for reproducibility
seed!(42)

# load dataset
data = CSV.File("melanoma.txt")

# retrieve observed variables
T = Float64.(data.days)         # observations, time-to-event
Delta = data.status             # event causes
predictors = data.sex           # categorical predictors

# remap event causes codes
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
hazard_est = HazardMarginal(rf, cm, times)
survival_est = SurvivalMarginal(rf, cm, times)
incidence_est = HazardMarginal(rf, cm, times; incidence = true)

# initialize conditional estimators
hazard_est_ = HazardConditional(times, crd.D; cm.L)
survival_est_ = SurvivalConditional(times; cm.L)

# estimators vector
marginal_estimators = [hazard_est, survival_est, incidence_est]
conditional_estimators = [hazard_est_, survival_est_]

# chain parameters
num_samples = (n = 4000, m = 25)
burn_in = 100000

# run chain
dgn = Gibbs(rf, cm, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in, thin = 25)

##########
# Diagnostics
##########

# diagnostics output
summary_dishes(dgn, burn_in)
summary_theta(dgn, burn_in)
summary_alpha(dgn, burn_in)
summary_eta(dgn, burn_in)
summary_coefficients(dgn, burn_in)
summary_loglikelihood(dgn, burn_in)

# traceplots (marginal)
traceplots(survival_est, 5.0)
traceplots(incidence_est, survival_est, 5.0)

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

# posterior estimate (conditional)
survival_post = estimate(survival_est_)

# credible intervals (conditional)
(survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)

# plots
survival_plot(times, survival_post[:,1]; kaplan_meier = survival_freq[:,1], lower = survival_lower[:,1], upper = survival_upper[:,1])
survival_plot(times, survival_post[:,2]; kaplan_meier = survival_freq[:,2], lower = survival_lower[:,2], upper = survival_upper[:,2])

##########
# Posterior estimates: incidence functions
##########

# frequentist estimate
cumincidence_freq = aalen_johansen(crd, times)

# posterior estimates (marginal) 
incidence_post = estimate(incidence_est, survival_est)
cumincidence_post = estimate(incidence_est, survival_est; cum = true)

# credible intervals (marginal)
(incidence_lower, incidence_upper) = credible_intervals(incidence_est, survival_est; lower = 0.05, upper = 0.95)
(cumincidence_lower, cumincidence_upper) = credible_intervals(incidence_est, survival_est; cum = true, lower = 0.05, upper = 0.95)

# posterior estimates (conditional)
incidence_post = estimate(hazard_est_, survival_est_)
cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)

# credible intervals (conditional)
(incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
(cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)

# plots (incidence)
incidence_plot(times, incidence_post[:,1,:]; lower = incidence_lower[:,1,:], upper = incidence_upper[:,1,:])
incidence_plot(times, incidence_post[:,2,:]; lower = incidence_lower[:,2,:], upper = incidence_upper[:,2,:])

# plots (incidence)
incidence_plot(times, cumincidence_post[:,1,:]; cum = true, aalen_johansen = cumincidence_freq[:,1,:], lower = cumincidence_lower[:,1,:], upper = cumincidence_upper[:,1,:])
incidence_plot(times, cumincidence_post[:,2,:]; cum = true, aalen_johansen = cumincidence_freq[:,2,:], lower = cumincidence_lower[:,2,:], upper = cumincidence_upper[:,2,:])
plot!(title = "", size = (480,360), xlim = (0.0,10.5), ylim = (0.0,0.55))

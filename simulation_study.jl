# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: MixtureModel, LocationScale, Weibull
import Statistics: mean

# include functions
include("create_datasets.jl")

##########
# Simulation study setup
##########

# number of tests
num_tests = 100

# dimensions
N = 100     # number of patients
D = 3       # number of diseases

# common distribution
common = MixtureModel([Weibull(1.2), LocationScale(1.0, 1.0, Weibull(3.0))])

# define models
true_models = [MixtureModel([Weibull(1.5), common]), MixtureModel([Weibull(2.0), common]), MixtureModel([Weibull(2.5), common])]

# models summary
summary_models(true_models)

# censoring distribution
# censoring = Gamma(1.0, 6.0)

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

# true density function
density_true = vec(sum(incidence_true, dims = 2))

##### models summary plots #####

# labels and colors
mycolors = reshape(1:length(true_models), 1, :)
mylabels = reshape(["cause " * string(d) for d in 1:length(true_models)], 1, :)

# hazard functions
plot(title = "", legend = :topleft, size = (480,360), xlim = (0.0, 1.65), ylim = (0.0, 1.6))
plot!(times, hazard_true, linecolor = mycolors, label = mylabels)

# incidence functions
plot(title = "", legend = :topright, size = (480,360), xlim = (0.0, 1.65), ylim = (0.0, 0.48))
plot!(times, incidence_true, linecolor = mycolors, label = mylabels)

##########
# Run tests: frequentist estimators
##########

# initialize error vectors
test_survival_errors = zeros(2, num_tests)
test_cumincidence_errors = zeros(D, 2, num_tests)

# loop on tests
for test in 1:num_tests

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    crd = independent_dataset(N, true_models)
    # crd = independent_dataset(N, true_models; censoring)

    # posterior survival function
    survival_freq = kaplan_meier(crd, times).estimate
    (max_error, int_error) = survival_error(times, survival_freq, survival_true, density_true)
    test_survival_errors[:,test] = [max_error, int_error]

    # posterior cumulative incidence functions
    cumincidence_freq = aalen_johansen(crd, times)
    (max_error, int_error) = incidence_error(times, cumincidence_freq, cumincidence_true, incidence_true)
    test_cumincidence_errors[:,:,test] = [max_error int_error]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: frequentist estimators
##########

println("--- Frequentist estimators ---")
println()

# survival errors
println("--- Survival function ---")
test_survival_errors = vec(mean(test_survival_errors, dims = 2))
println("MAX: ", string(test_survival_errors[1]))
println("INT: ", string(test_survival_errors[2]))
println()

# cumincidence errors
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = dropdims(mean(test_cumincidence_errors, dims = 3), dims = 3)
println("MAX: ", string(test_cumincidence_errors[:,1]))
println("INT: ", string(test_cumincidence_errors[:,2]))
println()

##########
# Run tests: smoothed frequentist estimators
##########

# initialize error vectors
test_survival_errors = zeros(2, num_tests)
test_cumincidence_errors = zeros(D, 2, num_tests)

# loop on tests
for test in 1:num_tests

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    crd = independent_dataset(N, true_models)
    # crd = independent_dataset(N, true_models; censoring)

    # posterior survival function
    survival_freq = kaplan_meier(crd, times).estimate
    survival_freq = smoother(survival_freq, times; left = t -> 2 - kaplan_meier(crd, -t).estimate, right = t -> kaplan_meier(crd, t).estimate)
    (max_error, int_error) = survival_error(times, survival_freq, survival_true, density_true)
    test_survival_errors[:,test] = [max_error, int_error]

    # posterior cumulative incidence functions
    cumincidence_freq = aalen_johansen(crd, times)
    cumincidence_freq = smoother(cumincidence_freq, times, right = t -> aalen_johansen(crd, t))
    (max_error, int_error) = incidence_error(times, cumincidence_freq, cumincidence_true, incidence_true)
    test_cumincidence_errors[:,:,test] = [max_error int_error]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: smoothed frequentist estimators
##########

println("--- Smoothed frequentist estimators ---")
println()

# survival errors
println("--- Survival function ---")
test_survival_errors = vec(mean(test_survival_errors, dims = 2))
println("MAX: ", string(test_survival_errors[1]))
println("INT: ", string(test_survival_errors[2]))
println()

# cumincidence errors
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = dropdims(mean(test_cumincidence_errors, dims = 3), dims = 3)
println("MAX: ", string(test_cumincidence_errors[:,1]))
println("INT: ", string(test_cumincidence_errors[:,2]))
println()

##########
# Run tests: RestaurantFranchise estimators
##########

# initialize diagnostics vectors
test_dishes = zeros(num_tests)
test_theta = zeros(num_tests)
test_logalpha = zeros(num_tests)

# initialize error vectors
test_survival_errors = zeros(2, num_tests)
test_incidence_errors = zeros(D, num_tests)
test_cumincidence_errors = zeros(D, 2, num_tests)

# initialize credible bands inclusion
isinbands_survival = zeros(num_tests)
isinbands_incidence = zeros(D, num_tests)
isinbands_cumincidence = zeros(D, num_tests)

# initialize credible bands width
bandwidth_survival = zeros(num_tests)
bandwidth_incidence = zeros(D, num_tests)
bandwidth_cumincidence = zeros(D, num_tests)

# loop on tests
for test in 1:num_tests

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    crd = independent_dataset(N, true_models)
    # crd = independent_dataset(N, true_models; censoring)

    # create RestaurantFranchise
    # rf = RestaurantFranchise(crd)
    rf = RestaurantFranchise(crd; sigma = 0.25, sigma0 = 0.25)
    
    # initialize estimators
    hazard_est = HazardMarginal(rf, times)
    survival_est = SurvivalMarginal(rf, times)
    incidence_est = HazardMarginal(rf, times; incidence = true)

    # initialize conditional estimators
    hazard_est_ = HazardConditional(times, D)
    survival_est_ = SurvivalConditional(times)
    
    # estimators vector
    marginal_estimators = [hazard_est, survival_est, incidence_est]
    conditional_estimators = [hazard_est_, survival_est_]
    
    # chain parameters (marginal)
    # num_samples = (n = 2500, m = 0)
    # burn_in = 25000

    # chain parameters (conditional)
    num_samples = (n = 1000, m = 25)
    burn_in = 25000
    
    # run chain
    # dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in)
    dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in, thin = 25)

    # diagnostics
    test_dishes[test] = mean(dgn.dishes_number[burn_in+1:end])
    test_theta[test] = mean(dgn.theta[burn_in+1:end])
    test_logalpha[test] = mean(dgn.logalpha[burn_in+1:end])

    # posterior survival function
    # survival_post = estimate(survival_est)
    survival_post = estimate(survival_est_)
    (max_error, int_error) = survival_error(times, survival_post, survival_true, density_true)
    test_survival_errors[:,test] = [max_error, int_error]

    # credible bands
    # (survival_lower, survival_upper) = credible_intervals(survival_est; lower = 0.05, upper = 0.95)
    (survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)
    isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    bandwidth_survival[test] = maximum(survival_upper .- survival_lower)
    
    # posterior incidence function
    # incidence_post = estimate(incidence_est, survival_est)
    incidence_post = estimate(hazard_est_, survival_est_)
    test_incidence_errors[:,test] = incidence_error(times, incidence_post, incidence_true)

    # credible bands
    # (incidence_lower, incidence_upper) = credible_intervals(incidence_est, survival_est; lower = 0.05, upper = 0.95)
    (incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
    isinbands_incidence[:, test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_incidence[:, test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in 1:D]

    # posterior cumulative incidence functions
    # cumincidence_post = estimate(incidence_est, survival_est; cum = true)
    cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)
    (max_error, int_error) = incidence_error(times, cumincidence_post, cumincidence_true, incidence_true)
    test_cumincidence_errors[:,:,test] = [max_error int_error]

    # credible bands
    # (cumincidence_lower, cumincidence_upper) = credible_intervals(incidence_est, survival_est; cum = true, lower = 0.05, upper = 0.95)
    (cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)
    isinbands_cumincidence[:, test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_cumincidence[:, test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in 1:D]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: RestaurantFranchise estimators
##########

println("--- RestaurantFranchise estimators ---")
println()

# diagnostics
println("--- Diagnostics ---")
println("Dishes: ", mean(test_dishes))
println("Theta: ", mean(test_theta))
println("Alpha: ", mean(exp.(test_logalpha)))
println()

# survival
println("--- Survival function ---")
test_survival_errors = vec(mean(test_survival_errors, dims = 2))
println("MAX: ", string(test_survival_errors[1]))
println("INT: ", string(test_survival_errors[2]))
isinbands_survival = mean(isinbands_survival)
println("inband: ", string(isinbands_survival))
bandwidth_survival = mean(bandwidth_survival)
println("bandwidth: ", string(bandwidth_survival))
println()

# incidence
println("--- Incidence functions ---")
test_incidence_errors = vec(mean(test_incidence_errors, dims = 2))
println("INT: ", string(test_incidence_errors))
isinbands_incidence = vec(mean(isinbands_incidence, dims = 2))
println("inband: ", string(isinbands_incidence))
bandwidth_incidence = vec(mean(bandwidth_incidence, dims = 2))
println("bandwidth: ", string(bandwidth_incidence))
println()

# cumincidence
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = dropdims(mean(test_cumincidence_errors, dims = 3), dims = 3)
println("MAX: ", string(test_cumincidence_errors[:,1]))
println("INT: ", string(test_cumincidence_errors[:,2]))
isinbands_cumincidence = vec(mean(isinbands_cumincidence, dims = 2))
println("inband: ", string(isinbands_cumincidence))
bandwidth_cumincidence = vec(mean(bandwidth_cumincidence, dims = 2))
println("bandwidth: ", string(bandwidth_cumincidence))
println()

##########
# Run tests: RestaurantArray estimators
##########

# initialize diagnostics vectors
test_dishes = zeros(num_tests)
test_theta = zeros(num_tests)
test_logalpha = zeros(num_tests)

# initialize error vectors
test_survival_errors = zeros(2, num_tests)
test_incidence_errors = zeros(D, num_tests)
test_cumincidence_errors = zeros(D, 2, num_tests)

# initialize credible bands inclusion
isinbands_survival = zeros(num_tests)
isinbands_incidence = zeros(D, num_tests)
isinbands_cumincidence = zeros(D, num_tests)

# initialize credible bands width
bandwidth_survival = zeros(num_tests)
bandwidth_incidence = zeros(D, num_tests)
bandwidth_cumincidence = zeros(D, num_tests)

# loop on tests
for test in 1:num_tests

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    crd = independent_dataset(N, true_models)
    # crd = independent_dataset(N, true_models; censoring)

    # create RestaurantArray
    # rf = RestaurantArray(crd)
    rf = RestaurantArray(crd; sigma = 0.25)
    
    # initialize estimators
    hazard_est = HazardMarginal(rf, times)
    survival_est = SurvivalMarginal(rf, times)
    incidence_est = HazardMarginal(rf, times; incidence = true)

    # initialize conditional estimators
    hazard_est_ = HazardConditional(times, D)
    survival_est_ = SurvivalConditional(times)
    
    # estimators vector
    marginal_estimators = [hazard_est, survival_est, incidence_est]
    conditional_estimators = [hazard_est_, survival_est_]
    
    # chain parameters (marginal)
    # num_samples = (n = 2500, m = 0)
    # burn_in = 25000

    # chain parameters (conditional)
    num_samples = (n = 1000, m = 25)
    burn_in = 25000
    
    # run chain
    # dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in)
    dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in, thin = 25)

    # diagnostics
    test_dishes[test] = mean(dgn.dishes_number[burn_in+1:end])
    test_theta[test] = mean(dgn.theta[burn_in+1:end])
    test_logalpha[test] = mean(dgn.logalpha[burn_in+1:end])

    # posterior survival function
    # survival_post = estimate(survival_est)
    survival_post = estimate(survival_est_)
    (max_error, int_error) = survival_error(times, survival_post, survival_true, density_true)
    test_survival_errors[:,test] = [max_error, int_error]

    # credible bands
    # (survival_lower, survival_upper) = credible_intervals(survival_est; lower = 0.05, upper = 0.95)
    (survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)
    isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    bandwidth_survival[test] = maximum(survival_upper .- survival_lower)

    # posterior incidence function
    # incidence_post = estimate(incidence_est, survival_est)
    incidence_post = estimate(hazard_est_, survival_est_)
    test_incidence_errors[:,test] = incidence_error(times, incidence_post, incidence_true)

    # credible bands
    # (incidence_lower, incidence_upper) = credible_intervals(incidence_est, survival_est; lower = 0.05, upper = 0.95)
    (incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
    isinbands_incidence[:, test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_incidence[:, test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in 1:D]

    # posterior cumulative incidence functions
    # cumincidence_post = estimate(incidence_est, survival_est; cum = true)
    cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)
    (max_error, int_error) = incidence_error(times, cumincidence_post, cumincidence_true, incidence_true)
    test_cumincidence_errors[:,:,test] = [max_error int_error]

    # credible bands
    # (cumincidence_lower, cumincidence_upper) = credible_intervals(incidence_est, survival_est; cum = true, lower = 0.05, upper = 0.95)
    (cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)
    isinbands_cumincidence[:, test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_cumincidence[:, test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in 1:D]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: RestaurantArray estimators
##########

println("--- RestaurantArray estimators ---")
println()

# diagnostics
println("--- Diagnostics ---")
println("Dishes: ", mean(test_dishes))
println("Theta: ", mean(test_theta))
println("Alpha: ", mean(exp.(test_logalpha)))
println()

# survival
println("--- Survival function ---")
test_survival_errors = vec(mean(test_survival_errors, dims = 2))
println("MAX: ", string(test_survival_errors[1]))
println("INT: ", string(test_survival_errors[2]))
isinbands_survival = mean(isinbands_survival)
println("inband: ", string(isinbands_survival))
bandwidth_survival = mean(bandwidth_survival)
println("bandwidth: ", string(bandwidth_survival))
println()

# incidence
println("--- Incidence functions ---")
test_incidence_errors = vec(mean(test_incidence_errors, dims = 2))
println("INT: ", string(test_incidence_errors))
isinbands_incidence = vec(mean(isinbands_incidence, dims = 2))
println("inband: ", string(isinbands_incidence))
bandwidth_incidence = vec(mean(bandwidth_incidence, dims = 2))
println("bandwidth: ", string(bandwidth_incidence))
println()

# cumincidence
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = dropdims(mean(test_cumincidence_errors, dims = 3), dims = 3)
println("MAX: ", string(test_cumincidence_errors[:,1]))
println("INT: ", string(test_cumincidence_errors[:,2]))
isinbands_cumincidence = vec(mean(isinbands_cumincidence, dims = 2))
println("inband: ", string(isinbands_cumincidence))
bandwidth_cumincidence = vec(mean(bandwidth_cumincidence, dims = 2))
println("bandwidth: ", string(bandwidth_cumincidence))
println()

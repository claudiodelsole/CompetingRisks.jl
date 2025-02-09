# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: MixtureModel, LocationScale, Weibull
import Statistics: mean
import Plots: plot, plot!

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
# common = Weibull(1.2)

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

##########
# Model summary plots
##########

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
# Setup kernel choice
##########

# kernel choice
CompetingRisks.kernel(x::Float64, t::Float64, eta::Float64) = CompetingRisks.kernel_DL(x, t, eta)

# KernelInt choice
CompetingRisks.KernelInt(x::Float64, t::Float64, eta::Float64) = CompetingRisks.KernelInt_DL(x, t, eta)

# resampling step
CompetingRisks.resample_eta(_::Union{RestaurantFranchise,RestaurantArray}) = (0.0, false)

##########
# Run tests: frequentist estimators
##########

# initialize counting vectors
test_counts = zeros(D, num_tests)

# initialize error vectors
test_survival_errors = zeros(num_tests)
test_cumincidence_errors = zeros(D, num_tests)

# loop on tests
for test in 1:num_tests

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    crd = independent_dataset(N, true_models)
    # crd = independent_dataset(N, true_models; censoring)

    # observation counts
    for d in 1:D
        test_counts[d,test] = sum(crd.Delta .== d)
    end

    # posterior survival function
    survival_freq = kaplan_meier(crd, times).estimate
    test_survival_errors[test] = survival_error(survival_freq, survival_true)

    # posterior cumulative incidence functions
    cumincidence_freq = aalen_johansen(crd, times)
    test_cumincidence_errors[:,test] = cumincidence_error(cumincidence_freq, cumincidence_true)

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: frequentist estimators
##########

println("### Frequentist estimators ###")
println()

# observations counts
println("--- Observations counts ---")
test_counts = vec(mean(test_counts, dims = 2))
println("counts: ", string(test_counts))
println()

# print output
println("--- Estimation errors ---")

# survival errors
println("survival: ", string(mean(test_survival_errors)))

# cumincidence errors
test_cumincidence_errors = vec(mean(test_cumincidence_errors, dims = 2))
println("cumincidence: ", string(test_cumincidence_errors))
println()

##########
# Run tests: RestaurantFranchise estimators - marginal method
##########

# initialize diagnostics vectors
test_dishes = zeros(num_tests)
test_theta = zeros(num_tests)
test_logalpha = zeros(num_tests)

# initialize error vectors
test_survival_errors = zeros(num_tests)
test_incidence_errors = zeros(D, num_tests)
test_cumincidence_errors = zeros(D, num_tests)

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
    
    # initialize marginal estimators
    survival_est = SurvivalMarginal(rf, times)
    incidence_est = IncidenceMarginal(rf, times)
    
    # estimators vector (marginal)
    marginal_estimators = [survival_est, incidence_est]
    conditional_estimators = Array{ConditionalEstimator}(undef, 0)
    
    # chain parameters (marginal)
    num_samples = (n = 2500, m = 0)
    burn_in = 10000

    # run chain
    dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in)

    # diagnostics
    test_dishes[test] = mean(dgn.dishes_number[burn_in+1:end])
    test_theta[test] = mean(dgn.theta[burn_in+1:end])
    test_logalpha[test] = mean(dgn.logalpha[burn_in+1:end])

    # posterior survival function
    survival_post = estimate(survival_est)
    test_survival_errors[test] = survival_error(survival_post, survival_true)

    # credible bands
    (survival_lower, survival_upper) = credible_intervals(survival_est; lower = 0.05, upper = 0.95)
    isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    bandwidth_survival[test] = maximum(survival_upper .- survival_lower)
    
    # posterior incidence function
    incidence_post = estimate(incidence_est, survival_est)
    test_incidence_errors[:,test] = incidence_error(incidence_post, incidence_true, times)

    # credible bands
    (incidence_lower, incidence_upper) = credible_intervals(incidence_est, survival_est; lower = 0.05, upper = 0.95)
    isinbands_incidence[:,test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_incidence[:,test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in 1:D]

    # posterior cumulative incidence functions
    cumincidence_post = estimate(incidence_est, survival_est; cum = true)
    test_cumincidence_errors[:,test] = cumincidence_error(cumincidence_post, cumincidence_true)

    # credible bands
    (cumincidence_lower, cumincidence_upper) = credible_intervals(incidence_est, survival_est; cum = true, lower = 0.05, upper = 0.95)
    isinbands_cumincidence[:,test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_cumincidence[:,test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in 1:D]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: RestaurantFranchise estimators - marginal method
##########

println("### RestaurantFranchise estimators - marginal method ###")
println()

# diagnostics
println("--- Diagnostics ---")
println("Dishes: ", mean(test_dishes))
println("Theta: ", mean(test_theta))
println("Alpha: ", mean(exp.(test_logalpha)))
println()

# survival
println("--- Survival function ---")
println("error: ", string(mean(test_survival_errors)))
println("inband: ", string(mean(isinbands_survival)))
println("bandwidth: ", string(mean(bandwidth_survival)))
println()

# incidence
println("--- Incidence functions ---")
test_incidence_errors = vec(mean(test_incidence_errors, dims = 2))
println("error: ", string(test_incidence_errors))
println("inband: ", string(mean(isinbands_incidence)))
println("bandwidth: ", string(mean(bandwidth_incidence)))
println()

# cumincidence
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = vec(mean(test_cumincidence_errors, dims = 2))
println("error: ", string(test_cumincidence_errors))
println("inband: ", string(mean(isinbands_cumincidence)))
println("bandwidth: ", string(mean(bandwidth_cumincidence)))
println()

##########
# Run tests: RestaurantFranchise estimators - conditional method
##########

# initialize diagnostics vectors
test_dishes = zeros(num_tests)
test_theta = zeros(num_tests)
test_logalpha = zeros(num_tests)

# initialize error vectors
test_survival_errors = zeros(num_tests)
test_incidence_errors = zeros(D, num_tests)
test_cumincidence_errors = zeros(D, num_tests)

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

    # initialize conditional estimators
    hazard_est_ = HazardConditional(times, D)
    survival_est_ = SurvivalConditional(times)

    # estimators vector (conditional)
    marginal_estimators = Array{MarginalEstimator}(undef, 0)
    conditional_estimators = [hazard_est_, survival_est_]

    # chain parameters (conditional)
    num_samples = (n = 1000, m = 20)
    burn_in = 10000
    
    # run chain
    dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in, thin = 25)

    # diagnostics
    test_dishes[test] = mean(dgn.dishes_number[burn_in+1:end])
    test_theta[test] = mean(dgn.theta[burn_in+1:end])
    test_logalpha[test] = mean(dgn.logalpha[burn_in+1:end])

    # posterior survival function
    survival_post = estimate(survival_est_)
    test_survival_errors[test] = survival_error(survival_post, survival_true)

    # credible bands
    (survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)
    isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    bandwidth_survival[test] = maximum(survival_upper .- survival_lower)
    
    # posterior incidence function
    incidence_post = estimate(hazard_est_, survival_est_)
    test_incidence_errors[:,test] = incidence_error(incidence_post, incidence_true, times)

    # credible bands
    (incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
    isinbands_incidence[:,test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_incidence[:,test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in 1:D]

    # posterior cumulative incidence functions
    cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)
    test_cumincidence_errors[:,test] = cumincidence_error(cumincidence_post, cumincidence_true)

    # credible bands
    (cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)
    isinbands_cumincidence[:,test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_cumincidence[:,test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in 1:D]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: RestaurantFranchise estimators - conditional method
##########

println("### RestaurantFranchise estimators - conditional method ###")
println()

# diagnostics
println("--- Diagnostics ---")
println("Dishes: ", mean(test_dishes))
println("Theta: ", mean(test_theta))
println("Alpha: ", mean(exp.(test_logalpha)))
println()

# survival
println("--- Survival function ---")
println("error: ", string(mean(test_survival_errors)))
println("inband: ", string(mean(isinbands_survival)))
println("bandwidth: ", string(mean(bandwidth_survival)))
println()

# incidence
println("--- Incidence functions ---")
test_incidence_errors = vec(mean(test_incidence_errors, dims = 2))
println("error: ", string(test_incidence_errors))
println("inband: ", string(mean(isinbands_incidence)))
println("bandwidth: ", string(mean(bandwidth_incidence)))
println()

# cumincidence
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = vec(mean(test_cumincidence_errors, dims = 2))
println("error: ", string(test_cumincidence_errors))
println("inband: ", string(mean(isinbands_cumincidence)))
println("bandwidth: ", string(mean(bandwidth_cumincidence)))
println()

##########
# Run tests: RestaurantArray estimators - marginal method
##########

# initialize diagnostics vectors
test_dishes = zeros(num_tests)
test_theta = zeros(num_tests)
test_logalpha = zeros(num_tests)

# initialize error vectors
test_survival_errors = zeros(num_tests)
test_incidence_errors = zeros(D, num_tests)
test_cumincidence_errors = zeros(D, num_tests)

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
    
    # initialize marginal estimators
    survival_est = SurvivalMarginal(rf, times)
    incidence_est = IncidenceMarginal(rf, times)
    
    # estimators vector (marginal)
    marginal_estimators = [survival_est, incidence_est]
    conditional_estimators = Array{ConditionalEstimator}(undef, 0)
    
    # chain parameters (marginal)
    num_samples = (n = 2500, m = 0)
    burn_in = 10000
    
    # run chain
    dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in)

    # diagnostics
    test_dishes[test] = mean(dgn.dishes_number[burn_in+1:end])
    test_theta[test] = mean(dgn.theta[burn_in+1:end])
    test_logalpha[test] = mean(dgn.logalpha[burn_in+1:end])

    # posterior survival function
    survival_post = estimate(survival_est)
    test_survival_errors[test] = survival_error(survival_post, survival_true)

    # credible bands
    (survival_lower, survival_upper) = credible_intervals(survival_est; lower = 0.05, upper = 0.95)
    isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    bandwidth_survival[test] = maximum(survival_upper .- survival_lower)

    # posterior incidence function
    incidence_post = estimate(incidence_est, survival_est)
    test_incidence_errors[:,test] = incidence_error(incidence_post, incidence_true, times)

    # credible bands
    (incidence_lower, incidence_upper) = credible_intervals(incidence_est, survival_est; lower = 0.05, upper = 0.95)
    isinbands_incidence[:,test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_incidence[:,test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in 1:D]

    # posterior cumulative incidence functions
    cumincidence_post = estimate(incidence_est, survival_est; cum = true)
    test_cumincidence_errors[:,test] = cumincidence_error(cumincidence_post, cumincidence_true)

    # credible bands
    (cumincidence_lower, cumincidence_upper) = credible_intervals(incidence_est, survival_est; cum = true, lower = 0.05, upper = 0.95)
    isinbands_cumincidence[:,test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_cumincidence[:,test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in 1:D]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: RestaurantArray estimators - marginal method
##########

println("### RestaurantArray estimators - marginal method ###")
println()

# diagnostics
println("--- Diagnostics ---")
println("Dishes: ", mean(test_dishes))
println("Theta: ", mean(test_theta))
println("Alpha: ", mean(exp.(test_logalpha)))
println()

# survival
println("--- Survival function ---")
println("error: ", string(mean(test_survival_errors)))
println("inband: ", string(mean(isinbands_survival)))
println("bandwidth: ", string(mean(bandwidth_survival)))
println()

# incidence
println("--- Incidence functions ---")
test_incidence_errors = vec(mean(test_incidence_errors, dims = 2))
println("error: ", string(test_incidence_errors))
println("inband: ", string(mean(isinbands_incidence)))
println("bandwidth: ", string(mean(bandwidth_incidence)))
println()

# cumincidence
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = vec(mean(test_cumincidence_errors, dims = 2))
println("error: ", string(test_cumincidence_errors))
println("inband: ", string(mean(isinbands_cumincidence)))
println("bandwidth: ", string(mean(bandwidth_cumincidence)))
println()

##########
# Run tests: RestaurantArray estimators - conditional method
##########

# initialize diagnostics vectors
test_dishes = zeros(num_tests)
test_theta = zeros(num_tests)
test_logalpha = zeros(num_tests)

# initialize error vectors
test_survival_errors = zeros(num_tests)
test_incidence_errors = zeros(D, num_tests)
test_cumincidence_errors = zeros(D, num_tests)

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

    # initialize conditional estimators
    hazard_est_ = HazardConditional(times, D)
    survival_est_ = SurvivalConditional(times)

    # estimators vector (conditional)
    marginal_estimators = Array{MarginalEstimator}(undef, 0)
    conditional_estimators = [hazard_est_, survival_est_]

    # chain parameters (conditional)
    num_samples = (n = 1000, m = 20)
    burn_in = 10000
    
    # run chain
    dgn = Gibbs(rf, marginal_estimators, conditional_estimators, num_samples; burn_in = burn_in, thin = 25)

    # diagnostics
    test_dishes[test] = mean(dgn.dishes_number[burn_in+1:end])
    test_theta[test] = mean(dgn.theta[burn_in+1:end])
    test_logalpha[test] = mean(dgn.logalpha[burn_in+1:end])

    # posterior survival function
    survival_post = estimate(survival_est_)
    test_survival_errors[test] = survival_error( survival_post, survival_true)

    # credible bands
    (survival_lower, survival_upper) = credible_intervals(survival_est_; lower = 0.05, upper = 0.95)
    isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    bandwidth_survival[test] = maximum(survival_upper .- survival_lower)

    # posterior incidence function
    incidence_post = estimate(hazard_est_, survival_est_)
    test_incidence_errors[:,test] = incidence_error(incidence_post, incidence_true, times)

    # credible bands
    (incidence_lower, incidence_upper) = credible_intervals(hazard_est_, survival_est_; lower = 0.05, upper = 0.95)
    isinbands_incidence[:,test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_incidence[:,test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in 1:D]

    # posterior cumulative incidence functions
    cumincidence_post = estimate(hazard_est_, survival_est_; cum = true)
    test_cumincidence_errors[:,test] = cumincidence_error(cumincidence_post, cumincidence_true)

    # credible bands
    (cumincidence_lower, cumincidence_upper) = credible_intervals(hazard_est_, survival_est_; cum = true, lower = 0.05, upper = 0.95)
    isinbands_cumincidence[:,test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in 1:D] ./ length(times)
    bandwidth_cumincidence[:,test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in 1:D]

    # print output
    println("Completed test ", string(test))

end

##########
# Tests results: RestaurantArray estimators - conditional method
##########

println("### RestaurantArray estimators - conditional method ###")
println()

# diagnostics
println("--- Diagnostics ---")
println("Dishes: ", mean(test_dishes))
println("Theta: ", mean(test_theta))
println("Alpha: ", mean(exp.(test_logalpha)))
println()

# survival
println("--- Survival function ---")
println("error: ", string(mean(test_survival_errors)))
println("inband: ", string(mean(isinbands_survival)))
println("bandwidth: ", string(mean(bandwidth_survival)))
println()

# incidence
println("--- Incidence functions ---")
test_incidence_errors = vec(mean(test_incidence_errors, dims = 2))
println("error: ", string(test_incidence_errors))
println("inband: ", string(mean(isinbands_incidence)))
println("bandwidth: ", string(mean(bandwidth_incidence)))
println()

# cumincidence
println("--- Cumulative incidence functions ---")
test_cumincidence_errors = vec(mean(test_cumincidence_errors, dims = 2))
println("error: ", string(test_cumincidence_errors))
println("inband: ", string(mean(isinbands_cumincidence)))
println("bandwidth: ", string(mean(bandwidth_cumincidence)))
println()

# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: MixtureModel, LocationScale, Weibull
import StatsBase: counts
import Statistics: mean, std
import Plots: plot, plot!, savefig

# include auxiliary files
include("../aux_code/functions.jl")
include("../aux_code/create_datasets.jl")
include("../aux_code/errors.jl")

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
# (plincidence, plprop) = summary_models(true_models)

# times vector
times = collect(0.0:0.01:2.0)

# true hazard functions
hazard_true = [hazard(model, t) for t in times, model in true_models]

# true survival function
survival_true = [prod([survival(model, t) for model in true_models]) for t in times]

# true incidence functions
incidence_true = [hazard(model, t) * prod([survival(model, t) for model in true_models]) for t in times, model in true_models]
cumincidence_true = mapslices(f -> integrate_trapz(f, times, cum = true), incidence_true, dims = 1)

# true density function
density_true = vec(sum(incidence_true, dims = 2))

##########
# Model summary plots
##########

# labels and colors
mycolors = reshape(range(1, length(true_models)), 1, :)
mylabels = reshape(["cause " * string(d) for d in range(1, length(true_models))], 1, :)

# hazard functions
plot(legend = :topleft, size = (480,360), xlim = (0.0, 1.65), ylim = (0.0, 1.6), xlabel = "\$t\$", ylabel = "\$h_δ(t)\$")
plot!(times, hazard_true, linecolor = mycolors, label = mylabels)
savefig("figures_supp/true_hazard.pdf")

# incidence functions
plot(legend = :topright, size = (480,360), xlim = (0.0, 1.65), ylim = (0.0, 0.48), xlabel = "\$t\$", ylabel = "\$f_δ(t)\$")
plot!(times, incidence_true, linecolor = mycolors, label = mylabels)
savefig("figures_supp/true_incidence.pdf")

##########
# Setup R
##########

# setup RCall
using RCall

# import required libraries
R"library(survival)"
R"library(cmprsk)"

# import times
@rput times

##########
# Run tests: frequentist estimators
##########

# initialize counting vectors
causes_counts = zeros(D, num_tests)

# initialize error vectors
errors_survival = zeros(num_tests)
errors_cumincidence = zeros(D, num_tests)

# loop on tests
for test in range(1, num_tests)

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    data = independent_dataset(N, true_models)

    # observation counts
    causes_counts[:,test] = counts(data.Delta, D)

    # import data
    @rput data

    # posterior survival function
    R"fit <- survfit(Surv(T, Delta > 0) ~ 1, data = data)"
    survival_freq = vcat(rcopy(R"summary(fit, times)$surv"), zeros(Float64, length(times) - length(rcopy(R"summary(fit, times)$surv"))))
    errors_survival[test] = error_survival(survival_freq, survival_true)

    # posterior cumulative incidence functions
    R"fit <- cuminc(data$T, data$Delta)"
    cumincidence_freq = mapslices(values -> map(x -> ismissing(x) ? maximum(skipmissing(values)) : x, values), Matrix(transpose(rcopy(R"timepoints(fit, times)$est"))), dims = 1)
    errors_cumincidence[:, test] = error_cumincidence(cumincidence_freq, cumincidence_true)

    # print output
    println("Completed test ", string(test))

end

# compute means and standard deviations
mean_std(samples::Array{Float64}; dims::Int64 = 1) = (mu = dropdims(mean(samples, dims = dims), dims = dims), sd = dropdims(std(samples, dims = dims), dims = dims) / sqrt(num_tests))

# compute means and standard deviations
causes_counts = vec(mean(causes_counts, dims = 2))
errors_survival = (mu = mean(errors_survival), sd = std(errors_survival) / sqrt(num_tests))
errors_cumincidence = mean_std(errors_cumincidence, dims = 2)

##########
# Tests results: frequentist estimators
##########

println("# Frequentist estimators")
println()

# observations counts
println("**Observations counts**")
println("counts:\t", string(causes_counts))
println()

# print output
println("**Estimation errors**")
println("survival:\t", string(errors_survival.mu), "\t(", string(errors_survival.sd), ")")
println("cumincidence:\t", string(errors_cumincidence.mu), "\t(", string(errors_cumincidence.sd), ")")
println()

##########
# Run tests: restaurant franchise estimators
##########

# initialize diagnostics vectors
dishes = zeros(num_tests)
theta = zeros(num_tests)
gamma = zeros(num_tests)

# initialize error vectors
errors_survival = zeros(num_tests, 2)
errors_incidence = zeros(D, num_tests, 2)
errors_cumincidence = zeros(D, num_tests, 2)

# initialize credible bands inclusion
# isinbands_survival = zeros(num_tests)
# isinbands_incidence = zeros(D, num_tests)
# isinbands_cumincidence = zeros(D, num_tests)

# initialize credible bands width
# bandwidth_survival = zeros(num_tests)
# bandwidth_incidence = zeros(D, num_tests)
# bandwidth_cumincidence = zeros(D, num_tests)

# create CompetingRisksModel
# cmprsk = CompetingRisksModel(DykstraLaudKernel)
cmprsk = CompetingRisksModel(DykstraLaudKernel, sigma = 0.25, sigma0 = 0.25)

# chain parameters
burn_in, thin = 2500, 10

# loop on tests
for test in range(1, num_tests)

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    data = independent_dataset(N, true_models)

    # run chain
    marginal_estimator, conditional_estimator, params = posterior_sampling(data, cmprsk, times = times, burn_in = burn_in, nsamples_crms = 10)

    # diagnostics
    dishes[test] = mean(params.dishes_number[burn_in+thin:thin:end])
    theta[test] = mean(params.theta[burn_in+thin:thin:end])
    gamma[test] = mean(params.kernelpars[burn_in+thin:thin:end])

    ### marginal estimator

    # posterior survival function
    (survival_post, survival_lower, survival_upper) = estimate_survival(marginal_estimator)
    errors_survival[test, 1] = error_survival(survival_post, survival_true)
    
    # posterior incidence function
    (incidence_post, incidence_lower, incidence_upper) = estimate_incidence(marginal_estimator)
    errors_incidence[:, test, 1] = error_incidence(incidence_post, incidence_true, times)

    # posterior cumulative incidence functions
    (cumincidence_post, cumincidence_lower, cumincidence_upper) = estimate_incidence(marginal_estimator, cum = true)
    errors_cumincidence[:, test, 1] = error_cumincidence(cumincidence_post, cumincidence_true)

    ### conditional estimator

    # posterior survival function
    (survival_post, survival_lower, survival_upper) = estimate_survival(conditional_estimator)
    errors_survival[test, 2] = error_survival(survival_post, survival_true)
    # isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    # bandwidth_survival[test] = maximum(survival_upper .- survival_lower)
    
    # posterior incidence function
    (incidence_post, incidence_lower, incidence_upper) = estimate_incidence(conditional_estimator)
    errors_incidence[:, test, 2] = error_incidence(incidence_post, incidence_true, times)
    # isinbands_incidence[:, test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in range(1, D)] ./ length(times)
    # bandwidth_incidence[:, test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in range(1, D)]

    # posterior cumulative incidence functions
    (cumincidence_post, cumincidence_lower, cumincidence_upper) = estimate_incidence(conditional_estimator, cum = true)
    errors_cumincidence[:, test, 2] = error_cumincidence(cumincidence_post, cumincidence_true)
    # isinbands_cumincidence[:, test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in range(1, D)] ./ length(times)
    # bandwidth_cumincidence[:, test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in range(1, D)]

    # print output
    println("Completed test ", string(test))

end

# compute means and standard deviations
errors_survival = mean_std(errors_survival)
errors_incidence = mean_std(errors_incidence, dims = 2)
errors_cumincidence = mean_std(errors_cumincidence, dims = 2)

# compute means and standard deviations
# isinbands_survival = mean_std(isinbands_survival)
# isinbands_incidence = mean_std(isinbands_incidence, dims = 2)
# isinbands_cumincidence = mean_std(isinbands_cumincidence, dims = 2)

# compute means and standard deviations
# bandwidth_survival = mean_std(bandwidth_survival)
# bandwidth_incidence = mean_std(bandwidth_incidence, dims = 2)
# bandwidth_cumincidence = mean_std(bandwidth_cumincidence, dims = 2)

##########
# Tests results: restaurant franchise estimators
##########

println("# Restaurant franchise estimators")
println()

# parameters
println("**Parameters**")
println("k:\t", mean(dishes), "\t(", string(std(dishes)), ")")
println("θ:\t", mean(theta), "\t(", string(std(theta)), ")")
println("γ:\t", mean(gamma), "\t(", string(std(gamma)), ")")
println()

# marginal method
println("**Marginal method**")
println("survival:\t", string(errors_survival.mu[1]), "\t(", string(errors_survival.sd[1]), ")")
println("incidence:\t", string(errors_incidence.mu[:,1]), "\t(", string(errors_incidence.sd[:,1]), ")")
println("cumincidence:\t", string(errors_cumincidence.mu[:,1]), "\t(", string(errors_cumincidence.sd[:,1]), ")")
println()

# conditional method
println("**Conditional method**")
println("survival:\t", string(errors_survival.mu[2]), "\t(", string(errors_survival.sd[2]), ")")
println("incidence:\t", string(errors_incidence.mu[:,2]), "\t(", string(errors_incidence.sd[:,2]), ")")
println("cumincidence:\t", string(errors_cumincidence.mu[:,2]), "\t(", string(errors_cumincidence.sd[:,2]), ")")
println()

##########
# Run tests: restaurant array estimators
##########

# initialize diagnostics vectors
dishes = zeros(num_tests)
theta = zeros(num_tests)
gamma = zeros(num_tests)

# initialize error vectors
errors_survival = zeros(num_tests, 2)
errors_incidence = zeros(D, num_tests, 2)
errors_cumincidence = zeros(D, num_tests, 2)

# initialize credible bands inclusion
# isinbands_survival = zeros(num_tests)
# isinbands_incidence = zeros(D, num_tests)
# isinbands_cumincidence = zeros(D, num_tests)

# initialize credible bands width
# bandwidth_survival = zeros(num_tests)
# bandwidth_incidence = zeros(D, num_tests)
# bandwidth_cumincidence = zeros(D, num_tests)

# create CompetingRisksModel
# cmprsk = CompetingRisksModel(DykstraLaudKernel)
cmprsk = CompetingRisksModel(DykstraLaudKernel, sigma = 0.25, sigma0 = 0.25, hierarchical = false)

# chain parameters
burn_in, thin = 2500, 10

# loop on tests
for test in range(1, num_tests)

    # set seed for reproducibility
    seed!(test)

    # create synthetic dataset
    data = independent_dataset(N, true_models)

    # run chain
    marginal_estimator, conditional_estimator, params = posterior_sampling(data, cmprsk, times = times, burn_in = burn_in, nsamples_crms = 10)

    # diagnostics
    dishes[test] = mean(params.dishes_number[burn_in+thin:thin:end])
    theta[test] = mean(params.theta[burn_in+thin:thin:end])
    gamma[test] = mean(params.kernelpars[burn_in+thin:thin:end])

    ### marginal estimator

    # posterior survival function
    (survival_post, survival_lower, survival_upper) = estimate_survival(marginal_estimator)
    errors_survival[test, 1] = error_survival(survival_post, survival_true)
    
    # posterior incidence function
    (incidence_post, incidence_lower, incidence_upper) = estimate_incidence(marginal_estimator)
    errors_incidence[:, test, 1] = error_incidence(incidence_post, incidence_true, times)

    # posterior cumulative incidence functions
    (cumincidence_post, cumincidence_lower, cumincidence_upper) = estimate_incidence(marginal_estimator, cum = true)
    errors_cumincidence[:, test, 1] = error_cumincidence(cumincidence_post, cumincidence_true)

    ### conditional estimator

    # posterior survival function
    (survival_post, survival_lower, survival_upper) = estimate_survival(conditional_estimator)
    errors_survival[test, 2] = error_survival(survival_post, survival_true)
    # isinbands_survival[test] = sum(survival_lower .<= survival_true .<= survival_upper) / length(times)
    # bandwidth_survival[test] = maximum(survival_upper .- survival_lower)
    
    # posterior incidence function
    (incidence_post, incidence_lower, incidence_upper) = estimate_incidence(conditional_estimator)
    errors_incidence[:, test, 2] = error_incidence(incidence_post, incidence_true, times)
    # isinbands_incidence[:, test] = [sum(incidence_lower[:,d] .<= incidence_true[:,d] .<= incidence_upper[:,d]) for d in range(1, D)] ./ length(times)
    # bandwidth_incidence[:, test] = [maximum(incidence_upper[:,d] .- incidence_lower[:,d]) for d in range(1, D)]

    # posterior cumulative incidence functions
    (cumincidence_post, cumincidence_lower, cumincidence_upper) = estimate_incidence(conditional_estimator, cum = true)
    errors_cumincidence[:, test, 2] = error_cumincidence(cumincidence_post, cumincidence_true)
    # isinbands_cumincidence[:, test] = [sum(cumincidence_lower[:,d] .<= cumincidence_true[:,d] .<= cumincidence_upper[:,d]) for d in range(1, D)] ./ length(times)
    # bandwidth_cumincidence[:, test] = [maximum(cumincidence_upper[:,d] .- cumincidence_lower[:,d]) for d in range(1, D)]

    # print output
    println("Completed test ", string(test))

end

# compute means and standard deviations
errors_survival = mean_std(errors_survival)
errors_incidence = mean_std(errors_incidence, dims = 2)
errors_cumincidence = mean_std(errors_cumincidence, dims = 2)

# compute means and standard deviations
# isinbands_survival = mean_std(isinbands_survival)
# isinbands_incidence = mean_std(isinbands_incidence, dims = 2)
# isinbands_cumincidence = mean_std(isinbands_cumincidence, dims = 2)

# compute means and standard deviations
# bandwidth_survival = mean_std(bandwidth_survival)
# bandwidth_incidence = mean_std(bandwidth_incidence, dims = 2)
# bandwidth_cumincidence = mean_std(bandwidth_cumincidence, dims = 2)

##########
# Tests results: restaurant array estimators
##########

println("# Restaurant array estimators")
println()

# parameters
println("**Parameters**")
println("k:\t", mean(dishes), "\t(", string(std(dishes)), ")")
println("θ:\t", mean(theta), "\t(", string(std(theta)), ")")
println("γ:\t", mean(gamma), "\t(", string(std(gamma)), ")")
println()

# marginal method
println("**Marginal method**")
println("survival:\t", string(errors_survival.mu[1]), "\t(", string(errors_survival.sd[1]), ")")
println("incidence:\t", string(errors_incidence.mu[:,1]), "\t(", string(errors_incidence.sd[:,1]), ")")
println("cumincidence:\t", string(errors_cumincidence.mu[:,1]), "\t(", string(errors_cumincidence.sd[:,1]), ")")
println()

# conditional method
println("**Conditional method**")
println("survival:\t", string(errors_survival.mu[2]), "\t(", string(errors_survival.sd[2]), ")")
println("incidence:\t", string(errors_incidence.mu[:,2]), "\t(", string(errors_incidence.sd[:,2]), ")")
println("cumincidence:\t", string(errors_cumincidence.mu[:,2]), "\t(", string(errors_cumincidence.sd[:,2]), ")")

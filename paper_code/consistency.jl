# load module
using CompetingRisks

# imports
import Random: seed!
import Distributions: Weibull, Normal
import StatsBase: counts
import Statistics: mean, quantile, std
import Plots: plot, plot!, vline!, histogram, savefig

# include auxiliary files
include("../aux_code/functions.jl")
include("../aux_code/create_datasets.jl")
include("../aux_code/errors.jl")

##########
# Synthetic independent dataset
##########

# define models
true_models = [Weibull(1.2), Weibull(1.6), Weibull(2.4)]

# times vector
times = collect(0.0:0.02:2.5)

# true survival function
survival_true = [prod([survival(model, t) for model in true_models]) for t in times]

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
# Simulation study
##########

# number of tests
num_tests = 50

# dimensions
N = [30, 60, 120, 240, 480]
D = 3   # number of diseases

# initialize error vectors
errors_freq = zeros(length(N), num_tests)
errors_post = zeros(length(N), num_tests)

# create CompetingRisksModel
# cmprsk = CompetingRisksModel(DykstraLaudKernel)
cmprsk = CompetingRisksModel(DykstraLaudKernel, sigma = 0.25, sigma0 = 0.25)

# loop on dimension
for (id, n) in enumerate(N)

    # chain parameters
    burn_in, thin = 10 * n, Int64(n / 30)

    # loop on tests
    for test in range(1, num_tests)

        # set seed for reproducibility
        seed!(test)

        # create synthetic dataset
        data = independent_dataset(n, true_models)

        # run chain
        marginal_estimator, _, params = posterior_sampling(data, cmprsk, times = times, burn_in = burn_in, thin = thin, nsamples_crms = 0)

        # posterior estimate
        (survival_post, survival_lower, survival_upper) = estimate_survival(marginal_estimator)
        errors_post[id, test] = error_survival(survival_post, survival_true)

        # import data
        @rput data

        # frequentist estimate
        R"fit <- survfit(Surv(T, Delta > 0) ~ 1, data = data)"
        R"km <- summary(fit, times)$surv"
        survival_freq = vcat(rcopy(R"km"), zeros(Float64, length(times) - length(rcopy(R"km"))))
        errors_freq[id, test] = error_survival(survival_freq, survival_true)

        # print output
        println("Completed test ", string(test))

    end

    # print output
    println("Completed tests for N = ", string(n))
    println()

end

##########
# Results
##########

# compute means and standard deviations
mean_std(samples::Array{Float64}; dims::Int64 = 1) = (mu = dropdims(mean(samples, dims = dims), dims = dims), sd = dropdims(std(samples, dims = dims), dims = dims) / sqrt(num_tests))
qnorm = quantile(Normal(), 0.975)

# compute means and standard deviations
errors_post = mean_std(errors_post, dims = 2)
errors_freq = mean_std(errors_freq, dims = 2)

# plot
begin 

    # initialize plot
    plot(size = (480,360), ylim = (-4.2, -1.7), xlabel = "\$ \\log (n)\$", ylabel = "\$\\log (d_K)\$")

    # posterior estimates
    plot!(log.(N), log.(errors_post.mu), linecolor = 1, label = "posterior")
    plot!(log.(N), log.(errors_post.mu - qnorm * errors_post.sd), fillrange = log.(errors_post.mu + qnorm * errors_post.sd), linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.3, primary = false)

    # frequentist estimates
    plot!(log.(N), log.(errors_freq.mu), linecolor = 3, label = "Kaplan-Meier")
    plot!(log.(N), log.(errors_freq.mu - qnorm * errors_freq.sd), fillrange = log.(errors_freq.mu + qnorm * errors_freq.sd), linecolor = 3, linealpha = 0.0, fillcolor = 3, fillalpha = 0.3, primary = false)

    # save figure
    savefig("figures_supp/consistency.pdf")

end

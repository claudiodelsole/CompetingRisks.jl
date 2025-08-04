# imports
import Statistics: mean, quantile

"""
    estimate_survival_bart(times::Vector{Float64}, survival_samples::Matrix{Float64}, evaltimes::Vector{Float64}; lower::Float64 = 0.025, upper::Float64 = 0.975)

"""
function estimate_survival_bart(times::Vector{Float64}, survival_samples::Matrix{Float64}, evaltimes::Vector{Float64}; lower::Float64 = 0.025, upper::Float64 = 0.975)

    # reshape samples
    post_samples = reshape(copy(survival_samples), size(survival_samples, 1), length(times), :)

    # evalute at evaltimes
    post_samples = mapslices(values -> timepoints(times, values, evaltimes, survival = true), post_samples, dims = 2)
    
    # compute mean
    post_mean = mean(post_samples, dims = 1)

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples, dims = 1)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples, dims = 1)

    # find singular dimensions
    todrop = tuple(findall(size(post_mean) .== 1)...)

    return (dropdims(post_mean, dims = todrop), dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # estimate_survival

"""
    estimate_incidence_bart(times::Vector{Float64}, cumincidence_samples1::Matrix{Float64}, cumincidence_samples2::Matrix{Float64}, evaltimes::Vector{Float64}; lower::Float64 = 0.025, upper::Float64 = 0.975)

"""
function estimate_incidence_bart(times::Vector{Float64}, cumincidence_samples1::Matrix{Float64}, cumincidence_samples2::Matrix{Float64}, evaltimes::Vector{Float64}; lower::Float64 = 0.025, upper::Float64 = 0.975)

    # reshape samples
    post_samples = cat(reshape(copy(cumincidence_samples1), size(cumincidence_samples1, 1), length(times), :),
                       reshape(copy(cumincidence_samples2), size(cumincidence_samples2, 1), length(times), :), dims = 4)

    # evalute at evaltimes
    post_samples = mapslices(values -> timepoints(times, values, evaltimes), post_samples, dims = 2)

    # compute mean
    post_mean = mean(post_samples, dims = 1)

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples, dims = 1)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples, dims = 1)

    # find singular dimensions
    todrop = tuple(findall(size(post_mean) .== 1)...)

    return (dropdims(post_mean, dims = todrop), dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # estimate_incidence_bart

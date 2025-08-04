# export functions
export estimate_survival, estimate_incidence, estimate_proportions

"""
    reshape_samples(survival_samples::Vector{Float64}, hazard_samples::Vector{Float64}, T::Int64, L::Int64, D::Int64)

"""
function reshape_samples(survival_samples::Vector{Float64}, hazard_samples::Vector{Float64}, T::Int64, L::Int64, D::Int64)

    # compute post samples
    post_samples = reshape(copy(hazard_samples), T, L + 1, D, :)
    for d in axes(post_samples, 3)
        post_samples[:,:,d,:] .*= reshape(survival_samples, T, L + 1, :)
    end
    
    return post_samples

end # reshape_samples

"""
    estimate_survival(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975)

"""
function estimate_survival(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975)

    # reshape samples
    post_samples = reshape(estimator.survival_samples, length(estimator.times), estimator.L + 1, :)
    
    # compute mean
    post_mean = mean(post_samples, dims = 3)

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples, dims = 3)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples, dims = 3)

    # find singular dimensions
    todrop = tuple(findall(size(post_mean) .== 1)...)

    return (dropdims(post_mean, dims = todrop), dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # estimate_survival

"""
    estimate_incidence(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975, cum::Bool = false)

"""
function estimate_incidence(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975, cum::Bool = false)

    # reshape samples
    post_samples = reshape_samples(estimator.survival_samples, estimator.hazard_samples, length(estimator.times), estimator.L, estimator.D)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, estimator.times, cum = true), post_samples, dims = 1)
    end

    # compute mean
    post_mean = mean(post_samples, dims = 4)

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples, dims = 4)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples, dims = 4)

    # find singular dimensions
    todrop = tuple(findall(size(post_mean) .== 1)...)

    return (dropdims(post_mean, dims = todrop), dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # estimate_incidence

"""
    estimate_proportions(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975)

"""
function estimate_proportions(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975)

    # reshape samples
    post_samples = reshape(copy(estimator.hazard_samples), length(estimator.times), estimator.L + 1, estimator.D, :)

    # normalize
    post_samples ./= sum(post_samples, dims = 3)

    # compute estimate
    post_mean = mapslices(samples -> mean(filter(!isnan, samples)), post_samples, dims = 4)

    # compute quantiles
    post_lower = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), lower), post_samples; dims = 4)
    post_upper = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), upper), post_samples; dims = 4)
    
    # find singular dimensions
    todrop = tuple(findall(size(post_mean) .== 1)...)

    return (dropdims(post_mean, dims = todrop), dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # estimate_proportions

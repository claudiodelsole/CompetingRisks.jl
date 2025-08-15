# export functions
export estimate_survival, estimate_incidence, estimate_proportions

"""
    estimate_survival(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975)

Compute estimate and pointwise credible bands of the survival function at timepoints in `estimator.times` for each level of the categorical predictor.

The `estimator` argument is the output of [`posterior_sampling`](@ref), `lower` and `upper` are quantiles for pointwise credible bands.

# Outputs:
- `survival_post::Array{Float64}`: posterior estimates of survival function
- `survival_lower::Array{Float64}`: lower limit of pointwise credible bands
- `survival_upper::Array{Float64}`: upper limit of pointwise credible bands
The index order of the outputs is `(time, level of predictor, sample)`.

See also [`estimate_incidence`](@ref), [`estimate_proportions`](@ref).
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

Compute estimates and pointwise credible bands of the incidence or cumulative incidence functions at timepoints in `estimator.times` for each level of the categorical predictor.

The `estimator` argument is the output of [`posterior_sampling`](@ref), `lower` and `upper` are quantiles for pointwise credible bands.

# Outputs:
- `incidence_post::Array{Float64}`: posterior estimates of incidence or cumulative incidence functions
- `incidence_lower::Array{Float64}`: lower limit of pointwise credible bands
- `incidence_upper::Array{Float64}`: upper limit of pointwise credible bands
The index order of the outputs is `(time, level of predictor, disease, sample)`.

See also [`estimate_survival`](@ref), [`estimate_proportions`](@ref).
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

Compute estimates and pointwise credible bands of the prediction curves of relative hazard functions at timepoints in `estimator.times` for each level of the categorical predictor.

The `estimator` argument is the output of [`posterior_sampling`](@ref). Estimates of prediction curves are obtained if `estimator` is `marginal_estimator`, posterior estimates of relative hazard functions are obtained if `estimator` is `posterior_estimator`.Parameters `lower` and `upper` are quantiles for pointwise credible bands.

# Outputs:
- `incidence_post::Array{Float64}`: estimates of prediction curves or posterior estimates of relative hazard functions
- `incidence_lower::Array{Float64}`: lower limit of pointwise credible bands
- `incidence_upper::Array{Float64}`: upper limit of pointwise credible bands
The index order of the outputs is `(time, disease, sample)`

See also [`estimate_survival`](@ref), [`estimate_incidence`](@ref).
"""
function estimate_proportions(estimator::Estimator; lower::Float64 = 0.025, upper::Float64 = 0.975)

    # reshape samples
    post_samples = reshape(copy(estimator.hazard_samples), length(estimator.times), estimator.L + 1, estimator.D, :)[:,1,:,:]

    # normalize
    post_samples ./= sum(post_samples, dims = 2)

    # compute estimate
    post_mean = mapslices(samples -> mean(filter(!isnan, samples)), post_samples, dims = 3)

    # compute quantiles
    post_lower = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), lower), post_samples; dims = 3)
    post_upper = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), upper), post_samples; dims = 3)
    
    # find singular dimensions
    todrop = tuple(findall(size(post_mean) .== 1)...)

    return (dropdims(post_mean, dims = todrop), dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # estimate_proportions

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

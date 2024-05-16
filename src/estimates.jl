# export functions
export estimate, credible_intervals

"""
    reshape_samples(hazard_estimator::Union{IncidenceMarginal,HazardConditional}, 
            survival_estimator::Union{SurvivalMarginal,SurvivalConditional})

"""
function reshape_samples(hazard_estimator::Union{IncidenceMarginal,HazardConditional}, 
        survival_estimator::Union{SurvivalMarginal,SurvivalConditional})

    # times vector length
    num_times = length(hazard_estimator.times)

    # reshape terms
    hazard_post_samples = reshape(hazard_estimator.post_samples, num_times, hazard_estimator.L + 1, hazard_estimator.D, :)
    survival_post_samples = reshape(survival_estimator.post_samples, num_times, hazard_estimator.L + 1, :)

    # compute post samples
    post_samples = copy(hazard_post_samples)
    for d in axes(post_samples, 3)
        post_samples[:,:,d,:] .*= survival_post_samples
    end
    
    return post_samples

end # reshape_samples

"""
    estimate(estimator::HazardMarginal; cum::Bool = false)

"""
function estimate(estimator::HazardMarginal; cum::Bool = false)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, estimator.D, :)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, estimator.times; cum = true), post_samples; dims = 1)
    end

    # compute estimate
    post_estimate = mean(post_samples; dims = 4)
    
    # find singular dimensions
    todrop = tuple(findall(size(post_estimate) .== 1)...)

    return dropdims(post_estimate, dims = todrop)

end # estimate

"""
    estimate(estimator::HazardConditional; cum::Bool = false, prop::Bool = false)

"""
function estimate(estimator::HazardConditional; cum::Bool = false, prop::Bool = false)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, estimator.D, :)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, estimator.times; cum = true), post_samples; dims = 1)
    end

    if prop == true # normalize
        post_samples ./= sum(post_samples; dims = 3)
    end

    # compute estimate
    post_estimate = mapslices(samples -> mean(filter(!isnan, samples)), post_samples; dims = 4)
    
    # find singular dimensions
    todrop = tuple(findall(size(post_estimate) .== 1)...)

    return dropdims(post_estimate, dims = todrop)

end # estimate

"""
    estimate(estimator::Union{SurvivalMarginal,SurvivalConditional})

"""
function estimate(estimator::Union{SurvivalMarginal,SurvivalConditional})

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, :)
    
    # compute estimate
    post_estimate = mean(post_samples, dims = 3)

    # find singular dimensions
    todrop = tuple(findall(size(post_estimate) .== 1)...)

    return dropdims(post_estimate, dims = todrop)

end # estimate

"""
    estimate(hazard_estimator::Union{IncidenceMarginal,HazardConditional}, 
            survival_estimator::Union{SurvivalMarginal,SurvivalConditional}; cum::Bool = false)

"""
function estimate(hazard_estimator::Union{IncidenceMarginal,HazardConditional}, 
        survival_estimator::Union{SurvivalMarginal,SurvivalConditional}; cum::Bool = false)

    # reshape samples
    post_samples = reshape_samples(hazard_estimator, survival_estimator)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, hazard_estimator.times; cum = true), post_samples; dims = 1)
    end

    # compute estimate
    post_estimate = mean(post_samples, dims = 4)

    # find singular dimensions
    todrop = tuple(findall(size(post_estimate) .== 1)...)

    return dropdims(post_estimate, dims = todrop)

end # estimate

"""
    estimate(estimator::IncidenceMarginal)

"""
function estimate(estimator::IncidenceMarginal)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, estimator.D, :)

    # normalize
    post_samples ./= sum(post_samples; dims = 3)

    # compute estimate
    post_estimate = mapslices(samples -> mean(filter(!isnan, samples)), post_samples; dims = 4)
    
    # find singular dimensions
    todrop = tuple(findall(size(post_estimate) .== 1)...)

    return dropdims(post_estimate, dims = todrop)

end # estimate

"""
    credible_intervals(estimator::HazardMarginal; 
            cum::Bool = false, lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function credible_intervals(estimator::HazardMarginal; 
        cum::Bool = false, lower::Float64 = 0.05, upper::Float64 = 0.95)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, estimator.D, :)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, estimator.times; cum = true), post_samples; dims = 1)
    end

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples; dims = 4)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples; dims = 4)

    # find singular dimensions
    todrop = tuple(findall(size(post_lower) .== 1)...)

    return (dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # credible_intervals

"""
    credible_intervals(estimator::HazardConditional; 
            cum::Bool = false, prop::Bool = false, lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function credible_intervals(estimator::HazardConditional; 
        cum::Bool = false, prop::Bool = false, lower::Float64 = 0.05, upper::Float64 = 0.95)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, estimator.D, :)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, estimator.times; cum = true), post_samples; dims = 1)
    end

    if prop == true # normalize
        post_samples ./= sum(post_samples; dims = 3)
    end

    # compute quantiles
    post_lower = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), lower), post_samples; dims = 4)
    post_upper = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), upper), post_samples; dims = 4)

    # find singular dimensions
    todrop = tuple(findall(size(post_lower) .== 1)...)

    return (dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # credible_intervals

"""
    credible_intervals(estimator::Union{SurvivalMarginal,SurvivalConditional}; lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function credible_intervals(estimator::Union{SurvivalMarginal,SurvivalConditional}; lower::Float64 = 0.05, upper::Float64 = 0.95)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, :)

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples; dims = 3)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples; dims = 3)

    # find singular dimensions
    todrop = tuple(findall(size(post_lower) .== 1)...)

    return (dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # credible_intervals

"""
    credible_intervals(hazard_estimator::Union{IncidenceMarginal,HazardConditional}, 
            survival_estimator::Union{SurvivalMarginal,SurvivalConditional}; 
            cum::Bool = false, lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function credible_intervals(hazard_estimator::Union{IncidenceMarginal,HazardConditional}, 
        survival_estimator::Union{SurvivalMarginal,SurvivalConditional}; 
        cum::Bool = false, lower::Float64 = 0.05, upper::Float64 = 0.95)

    # reshape samples
    post_samples = reshape_samples(hazard_estimator, survival_estimator)

    if cum == true  # integrate over times
        post_samples = mapslices(f -> integrate_trapz(f, hazard_estimator.times; cum = true), post_samples; dims = 1)
    end

    # compute quantiles
    post_lower = mapslices(samples -> quantile(samples, lower), post_samples; dims = 4)
    post_upper = mapslices(samples -> quantile(samples, upper), post_samples; dims = 4)

    # find singular dimensions
    todrop = tuple(findall(size(post_lower) .== 1)...)

    return (dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # credible_intervals

"""
    credible_intervals(estimator::IncidenceMarginal; lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function credible_intervals(estimator::IncidenceMarginal; lower::Float64 = 0.05, upper::Float64 = 0.95)

    # reshape samples
    post_samples = reshape(estimator.post_samples, length(estimator.times), estimator.L + 1, estimator.D, :)

    # normalize
    post_samples ./= sum(post_samples; dims = 3)

    # compute quantiles
    post_lower = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), lower), post_samples; dims = 4)
    post_upper = mapslices(samples -> isempty(filter(!isnan, samples)) ? NaN : quantile(filter(!isnan, samples), upper), post_samples; dims = 4)

    # find singular dimensions
    todrop = tuple(findall(size(post_lower) .== 1)...)

    return (dropdims(post_lower, dims = todrop), dropdims(post_upper, dims = todrop))

end # credible_intervals

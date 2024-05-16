# export functions
export kaplan_meier, nelson_aalen, aalen_johansen

"""
    frequentist_setup(crd::CompetingRisksDataset; l::Int64 = 0)

"""
function frequentist_setup(crd::CompetingRisksDataset; l::Int64 = 0)

    # initialize vectors
    event_times = Array{Float64}(undef, 0)
    deaths = Array{Int64}(undef, 0)
    atrisk = Array{Int64}(undef, 0)

    # initialize auxiliary variables
    last_time = 0.0
    last_atrisk = sum(crd.predictors .== l)

    # loop on times-to-event
    for idx in sortperm(crd.T)

        # retrieve time and cause
        time = crd.T[idx]
        cause = crd.Delta[idx]
        predictor = crd.predictors[idx]

        if predictor != l   # ignore observtion
            continue
        end

        if time > last_time    # new event time

            # append to vectors
            append!(event_times, time)
            append!(deaths, fill(0, crd.D))
            append!(atrisk, last_atrisk)
            
        end

        # register new death
        if cause != 0
            deaths[end - crd.D + cause] += 1
        end

        # update auxiliary variables
        last_time = time
        last_atrisk -= 1

    end

    # reshape deaths
    deaths = transpose(reshape(deaths, crd.D, :))

    return (event_times, deaths, atrisk)

end # frequentist_setup

"""
    kaplan_meier(crd::CompetingRisksDataset, t::Float64; lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function kaplan_meier(crd::CompetingRisksDataset, t::Float64; lower::Float64 = 0.05, upper::Float64 = 0.95)

    # number of categorical levels
    L = maximum(crd.predictors)

    # initialize estimate and variance
    logestimate = zeros(Float64, L+1)
    logvariance = zeros(Float64, L+1)

    # initialize confidence intervals
    loglower = zeros(Float64, L+1)
    logupper = zeros(Float64, L+1)

    # loop on categorical predictors
    for l in 0:L

        # retrieve vectors
        (event_times, deaths, atrisk) = frequentist_setup(crd; l = l)

        # cumulate deaths
        deaths = sum(deaths; dims = 2)

        # initialize estimate and variance
        logest = 0.0
        logvar = 0.0

        # loop on times
        for (idx, event_time) in enumerate(event_times)

            # exceeds time interval
            if t < event_time
                break
            end

            # update logestimate
            logest += log(1 - deaths[idx] / atrisk[idx])

            # update logvariance
            logvar += deaths[idx] / ( atrisk[idx] * (atrisk[idx] - deaths[idx]) )

        end

        # store estimate and variance
        logestimate[l+1] = logest
        logvariance[l+1] = logvar

        # compute log-transformed confidence intervals
        if logvariance[l+1] != 0  # zero logvariance

            # log-minus-log transformation
            loglower[l+1] = logest * exp( quantile(Normal(), lower) * sqrt(logvar) / logest )
            logupper[l+1] = logest * exp( quantile(Normal(), upper) * sqrt(logvar) / logest )

        end

    end

    # drop singular dimensions
    if L == 0 
        return (estimate = exp(logestimate[begin]), lower = exp(loglower[begin]), upper = exp(logupper[begin]))
    end

    return (estimate = exp.(logestimate), lower = exp.(loglower), upper = exp.(logupper))

end # kaplan_meier

"""
    kaplan_meier(crd::CompetingRisksDataset, times::Vector{Float64}; lower::Float64 = 0.05, upper::Float64 = 0.95)

"""
function kaplan_meier(crd::CompetingRisksDataset, times::Vector{Float64}; lower::Float64 = 0.05, upper::Float64 = 0.95)

    # number of categorical levels
    L = maximum(crd.predictors)

    # initialize estimate
    logestimate = zeros(Float64, length(times), L+1)
    logvariance = zeros(Float64, length(times), L+1)

    # initialize confidence intervals
    loglower = zeros(Float64, length(times), L+1)
    logupper = zeros(Float64, length(times), L+1)

    # loop on categorical predictors
    for l in 0:L

        # retrieve vectors
        (event_times, deaths, atrisk) = frequentist_setup(crd; l = l)
        num_event_times = length(event_times)

        # cumulate deaths
        deaths = sum(deaths; dims = 2)

        # current index
        idx = 1

        # loop on times
        for (t, time) in enumerate(times)

            while idx <= num_event_times && time > event_times[idx]     # same time interval

                # update logestimate
                logestimate[t,l+1] += log(1 - deaths[idx] / atrisk[idx])

                # update logvariance
                logvariance[t,l+1] += deaths[idx] / ( atrisk[idx] * (atrisk[idx] - deaths[idx]) )

                # update current index
                idx += 1

            end

        end

        # compute logestimate
        logestimate[:,l+1] = cumsum(logestimate[:,l+1])

        # compute log-transformed variance
        logvariance[:,l+1] = cumsum(logvariance[:,l+1])

        # compute log-transformed confidence intervals
        for t in eachindex(times)

            if logvariance[t,l+1] != 0   # zero logvariance

                # log-minus-log transformation
                loglower[t,l+1] = logestimate[t,l+1] * exp( quantile(Normal(), lower) * sqrt(logvariance[t,l+1]) / logestimate[t,l+1] )
                logupper[t,l+1] = logestimate[t,l+1] * exp( quantile(Normal(), upper) * sqrt(logvariance[t,l+1]) / logestimate[t,l+1] )

            end

        end

    end

    # drop singular dimensions
    if L == 0 
        logestimate = dropdims(logestimate, dims = 2)
        loglower = dropdims(loglower, dims = 2)
        logupper = dropdims(logupper, dims = 2)
    end

    return (estimate = exp.(logestimate), lower = exp.(loglower), upper = exp.(logupper))

end # kaplan_meier

"""
    nelson_aalen(crd::CompetingRisksDataset, t::Float64)

"""
function nelson_aalen(crd::CompetingRisksDataset, t::Float64)

    # number of categorical levels
    L = maximum(crd.predictors)

    # initialize estimate
    estimate = zeros(Float64, L+1, crd.D)

    # loop on categorical predictors
    for l in 0:L

        # retrieve vectors
        (event_times, deaths, atrisk) = frequentist_setup(crd; l = l)

        # loop on times
        for (idx, event_time) in enumerate(event_times)

            # exceeds time interval
            if t < event_time
                break
            end

            # update logestimate
            estimate[l+1,:] += deaths[idx,:] ./ atrisk[idx]

        end

    end

    # drop singular dimensions
    if L == 0 estimate = dropdims(estimate, dims = 1) end

    return estimate

end # nelson_aalen

"""
    nelson_aalen(crd::CompetingRisksDataset, times::Vector{Float64})

"""
function nelson_aalen(crd::CompetingRisksDataset, times::Vector{Float64})

    # number of categorical levels
    L = maximum(crd.predictors)

    # initialize estimate
    estimate = zeros(Float64, length(times), L+1, crd.D)

    # loop on categorical predictors
    for l in 0:L

        # retrieve vectors
        (event_times, deaths, atrisk) = frequentist_setup(crd; l = l)
        num_event_times = length(event_times)

        # current index
        idx = 1

        # loop on times
        for (t, time) in enumerate(times)

            while idx <= num_event_times && time > event_times[idx]     # same time interval

                # update logestimate
                estimate[t,l+1,:] += deaths[idx,:] ./ atrisk[idx]

                # update current index
                idx += 1

            end

        end

    end

    # compute estimate
    estimate = cumsum(estimate; dims = 1)

    # drop singular dimensions
    if L == 0 estimate = dropdims(estimate, dims = 2) end

    return estimate

end # nelson_aalen

"""
    aalen_johansen(crd::CompetingRisksDataset, t::Float64)

"""
function aalen_johansen(crd::CompetingRisksDataset, t::Float64)

    # number of categorical levels
    L = maximum(crd.predictors)

    # initialize estimate
    estimate = zeros(Float64, L+1, crd.D)

    # loop on categorical predictors
    for l in 0:L

        # retrieve vectors
        (event_times, deaths, atrisk) = frequentist_setup(crd; l = l)

        # initialize logsurvival
        logsurvival = 0.0

        # loop on times
        for (idx, event_time) in enumerate(event_times)

            # exceeds time interval
            if t < event_time
                break
            end

            # update logestimate
            estimate[l+1,:] += deaths[idx,:] ./ atrisk[idx] .* exp(logsurvival)

            # update logsurvival
            logsurvival += log(1 - sum(deaths[idx,:]) / atrisk[idx])

        end

    end

    # drop singular dimensions
    if L == 0 estimate = dropdims(estimate, dims = 1) end

    return estimate

end # aalen_johansen

"""
    aalen_johansen(crd::CompetingRisksDataset, times::Vector{Float64})

"""
function aalen_johansen(crd::CompetingRisksDataset, times::Vector{Float64})

    # number of categorical levels
    L = maximum(crd.predictors)

    # initialize estimate
    estimate = zeros(Float64, length(times), L+1, crd.D)

    # loop on categorical predictors
    for l in 0:L

        # retrieve vectors
        (event_times, deaths, atrisk) = frequentist_setup(crd, l = l)
        num_event_times = length(event_times)

        # initialize logsurvival
        logsurvival = 0.0

        # current index
        idx = 1

        # loop on times
        for (t, time) in enumerate(times)

            while idx <= num_event_times && time > event_times[idx]     # same time interval

                # update logestimate
                estimate[t,l+1,:] += deaths[idx,:] ./ atrisk[idx] .* exp(logsurvival)

                # update logsurvival
                logsurvival += log(1 - sum(deaths[idx,:]) / atrisk[idx])

                # update current index
                idx += 1

            end

        end

    end

    # compute estimate
    estimate = cumsum(estimate; dims = 1)

    # drop singular dimensions
    if L == 0 estimate = dropdims(estimate, dims = 2) end

    return estimate

end # aalen_johansen

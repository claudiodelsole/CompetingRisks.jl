# imports
import MCMCDiagnosticTools: ess

"""
    ess_survival(estimator::Estimator, time::Float64; l::Int64 = 0)

"""
function ess_survival(estimator::Estimator, time::Float64; l::Int64 = 0)

    # reshape samples
    post_samples = reshape(estimator.survival_samples, length(estimator.times), estimator.L + 1, :)

    # retrieve trace
    t = sum(time .>= estimator.times)
    trace = post_samples[t,l+1,:]

    # compute ess
    return ess(trace)

end # ess_survival

"""
    timepoints(times::Vector{Float64}, values::Vector{Float64}, evaltimes::Vector{Float64}; survival::Bool = false)

"""
function timepoints(times::Vector{Float64}, values::Vector{Float64}, evaltimes::Vector{Float64}; survival::Bool = false)

    # initialize estimate
    estimate = zeros(Float64, length(evaltimes))
    
    # loop on times
    for (t, time) in enumerate(evaltimes)

        # retrieve index
        idx = sum(time .>= times)

        # compute value
        if idx == 0 estimate[t] = Float64(survival) else estimate[t] = values[idx] end

    end
    
    return estimate

end # timepoints

# imports
import Distributions: UnivariateDistribution, pdf, ccdf
import MCMCDiagnosticTools: ess

"""
    hazard(d::Type, t::Float64) where Type <: UnivariateDistribution

"""
hazard(d::Type, t::Float64) where Type <: UnivariateDistribution = pdf(d, t) / ccdf(d, t)

"""
    survival(d::Type, t::Float64) where Type <: UnivariateDistribution

"""
survival(d::Type, t::Float64) where Type <: UnivariateDistribution = ccdf(d, t)

"""
    integrate_trapz(values::Vector{Float64}, times::Vector{Float64}; cum::Bool = false)

"""
function integrate_trapz(values::Vector{Float64}, times::Vector{Float64}; cum::Bool = false)

    # trapezoid integrals
    trapz = 0.5 * (values[begin:end-1] + values[begin+1:end]) .* diff(times)

    # compute integrals
    if cum == true
        return pushfirst!(cumsum(trapz), 0.0)
    end

    return sum(trapz)

end # integrate_trapz

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

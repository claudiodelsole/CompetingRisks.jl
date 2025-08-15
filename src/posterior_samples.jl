"""
    append(estimator::Estimator, crms::Vector{CRM}, kernelpars::AbstractKernel, CoxProd::Union{Vector{Float64},Nothing})

"""
function append(estimator::Estimator, crms::Vector{CRM}, kernelpars::AbstractKernel, CoxProd::Union{Vector{Float64},Nothing})

    # compute survival estimate
    survival = survival_measures(crms, estimator.times, kernelpars, CoxProd)

    # compute hazard estimate
    hazard = hazard_measures(crms, estimator.times, kernelpars, CoxProd)

    # append estimates
    append!(estimator.survival_samples, survival)
    append!(estimator.hazard_samples, hazard)

end # append

"""
    survival_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, _::Nothing)

"""
function survival_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, _::Nothing)

    # initialize estimate vector
    estimate = zeros(length(times))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times

        # initialize estimate
        est = 0.0

        for crm in crms     # loop on causes
            for (atom, jump) in zip(crm.locations, crm.jumps)       # loop on locations and jumps
                est -= KernelInt(atom, time, nothing, kernelpars) * jump
            end
        end

        # store estimate
        estimate[t] = est

    end

    return exp.(estimate)

end # survival_measures

"""
    survival_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, CoxProd::Vector{Float64})

"""
function survival_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, CoxProd::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (l, cp) in enumerate(CoxProd)   # loop on predictors

            # initialize estimate
            est = 0.0

            for crm in crms     # loop on causes
                for (atom, jump) in zip(crm.locations, crm.jumps)       # loop on locations and jumps
                    est -= KernelInt(atom, time, cp, kernelpars) * jump
                end
            end

            # store estimate
            estimate[t,l] = est

        end
    end

    return exp.(estimate)

end # survival_measures

"""
    hazard_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, _::Nothing)

"""
function hazard_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, _::Nothing)

    # initialize estimate vector
    estimate = zeros(length(times), length(crms))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (d, crm) in enumerate(crms)     # loop on causes

            # initialize estimate
            est = 0.0

            # loop on locations and jumps
            for (atom, jump) in zip(crm.locations, crm.jumps)
                est += kernel(atom, time, nothing, kernelpars) * jump
            end

            # store estimate
            estimate[t,d] = est

        end
    end

    return estimate

end # hazard_measures

"""
    hazard_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, CoxProd::Vector{Float64})

"""
function hazard_measures(crms::Vector{CRM}, times::Vector{Float64}, kernelpars::AbstractKernel, CoxProd::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), length(crms))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (l, cp) in enumerate(CoxProd)   # loop on predictors
            for (d, crm) in enumerate(crms)     # loop on causes

                # initialize estimate
                est = 0.0

                # loop on locations and jumps
                for (atom, jump) in zip(crm.locations, crm.jumps)
                    est += kernel(atom, time, cp, kernelpars) * jump
                end

                # store estimate
                estimate[t,l,d] = est

            end
        end
    end

    return estimate

end # hazard_measures

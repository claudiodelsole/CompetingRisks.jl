# export structs
export ConditionalEstimator, HazardConditional, SurvivalConditional

abstract type ConditionalEstimator end

"""
    struct HazardConditional <: ConditionalEstimator

"""
struct HazardConditional <: ConditionalEstimator

    # dimensions
    D::Int64        # number of diseases
    L::Int64        # number of categorical predictors

    # times vector
    times::Vector{Float64}

    # posterior samples
    post_samples::Vector{Float64}

    # explicit constructor
    function HazardConditional(times::Vector{Float64}, D::Int64; L::Int64 = 0)

        # initialize vector
        post_samples = Array{Float64}(undef, 0)

        # create HazardConditional
        return new(D, L, times, post_samples)

    end # HazardConditional

end # struct

"""
    struct SurvivalConditional <: ConditionalEstimator

"""
struct SurvivalConditional <: ConditionalEstimator

    # dimensions
    L::Int64        # number of categorical predictors

    # times vector
    times::Vector{Float64}

    # posterior samples
    post_samples::Vector{Float64}

    # explicit constructor
    function SurvivalConditional(times::Vector{Float64}; L::Int64 = 0)

        # initialize vector
        post_samples = Array{Float64}(undef, 0)

        # create SurvivalConditional
        return new(L, times, post_samples)

    end # SurvivalConditional

end # struct

"""
    append(estimator::HazardConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64)

"""
function append(estimator::HazardConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64)

    # compute estimate
    estimate = hazard_measures(CRMs, estimator.times, alpha, eta)

    # append estimate
    append!(estimator.post_samples, estimate)

end # append

"""
    append(estimator::SurvivalConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64)

"""
function append(estimator::SurvivalConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64)

    # compute estimate
    estimate = survival_measures(CRMs, estimator.times, alpha, eta)

    # append estimate
    append!(estimator.post_samples, estimate)

end # append

"""
    hazard_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64)

"""
function hazard_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64)

    # initialize estimate vector
    estimate = zeros(length(times), hCRM.D)

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (d, jumps) in enumerate(eachcol(hCRM.jumps))     # loop on diseases

            # initialize estimate
            est = 0.0

            # loop on locations and jumps
            for (atom, jump) in zip(hCRM.locations, jumps)
                est += alpha * kernel(atom, time, eta) * jump
            end

            # store estimate
            estimate[t,d] = est

        end
    end

    return estimate

end # hazard_measures

"""
    hazard_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64)

"""
function hazard_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64)

    # initialize estimate vector
    estimate = zeros(length(times), CRMs.D)

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times

        # initialize estimate
        est = zeros(Float64, CRMs.D)

        # loop on locations and jumps
        for (d, atom, jump) in zip(CRMs.jump_disease, CRMs.locations, CRMs.jumps)   
            if d != 0    # jump assigned to disease
                est[d] += alpha * kernel(atom, time, eta) * jump
            end
        end

        # store estimate
        estimate[t,:] = est

    end

    return estimate

end # hazard_measures

"""
    survival_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64)

"""
function survival_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64)

    # initialize estimate vector
    estimate = zeros(length(times))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times

        # initialize estimate
        est = 0.0

        for jumps in eachcol(hCRM.jumps)     # loop on diseases
            for (atom, jump) in zip(hCRM.locations, jumps)      # loop on locations and jumps
                est -= alpha * KernelInt(atom, time, eta) * jump
            end
        end

        # store estimate
        estimate[t] = est

    end

    return exp.(estimate)

end # survival_measures

"""
    survival_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64)

"""
function survival_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64)

    # initialize estimate vector
    estimate = zeros(length(times))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times

        # initialize estimate
        est = 0.0

        # loop on locations and jumps
        for (atom, jump) in zip(CRMs.locations, CRMs.jumps)
            est -= alpha * KernelInt(atom, time, eta) * jump
        end

        # store estimate
        estimate[t] = est

    end

    return exp.(estimate)

end # survival_measures

"""
    append(estimator::HazardConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

"""
function append(estimator::HazardConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

    # compute estimate
    estimate = hazard_measures(CRMs, estimator.times, alpha, eta, CoxProd)

    # append estimate
    append!(estimator.post_samples, estimate)

end # append

"""
    append(estimator::SurvivalConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, CoxProd::Vector{Float64})

"""
function append(estimator::SurvivalConditional, CRMs::Union{HierarchicalCRM,CRMArray}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

    # compute estimate
    estimate = survival_measures(CRMs, estimator.times, alpha, eta, CoxProd)

    # append estimate
    append!(estimator.post_samples, estimate)

end # append

"""
    hazard_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

"""
function hazard_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), hCRM.D)

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (l, cp) in enumerate(CoxProd)   # loop on predictors
            for (d, jumps) in enumerate(eachcol(hCRM.jumps))    # loop on diseases

                # initialize estimate
                est = 0.0

                # loop on locations and jumps
                for (atom, jump) in zip(hCRM.locations, jumps)  
                    est += alpha * kernel(atom, time, cp, eta) * jump
                end

                # store estimate
                estimate[t,l,d] = est

            end
        end
    end

    return estimate

end # hazard_measures

"""
    hazard_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

"""
function hazard_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), CRMs.D)

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (l, cp) in enumerate(CoxProd)   # loop on predictors

            # initialize estimate
            est = zeros(Float64, CRMs.D)

            # loop on locations and jumps
            for (d, atom, jump) in zip(CRMs.jump_disease, CRMs.locations, CRMs.jumps)   
                if d != 0    # jump assigned to disease
                    est[d] += alpha * kernel(atom, time, cp, eta) * jump
                end

            end

            # store estimate
            estimate[t,l,:] = est

        end
    end

    return estimate

end # hazard_measures

"""
    survival_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

"""
function survival_measures(hCRM::HierarchicalCRM, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (l, cp) in enumerate(CoxProd)   # loop on predictors

            # initialize estimate
            est = 0.0

            for jumps in eachcol(hCRM.jumps)    # loop on diseases
                for (atom, jump) in zip(hCRM.locations, jumps)    # loop on locations and jumps
                    est -= alpha * KernelInt(atom, time, cp, eta) * jump
                end
            end

            # store estimate
            estimate[t,l] = est

        end
    end

    return exp.(estimate)

end # survival_measures

"""
    survival_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

"""
function survival_measures(CRMs::CRMArray, times::Vector{Float64}, alpha::Float64, eta::Float64, CoxProd::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # compute estimates
    for (t, time) in enumerate(times)   # loop on times
        for (l, cp) in enumerate(CoxProd)   # loop on predictors

            # initialize estimate
            est = 0.0

            # loop on locations and jumps
            for (atom, jump) in zip(CRMs.locations, CRMs.jumps)
                est -= alpha * KernelInt(atom, time, cp, eta) * jump
            end

            # store estimate
            estimate[t,l] = est

        end
    end

    return exp.(estimate)

end # survival_measures

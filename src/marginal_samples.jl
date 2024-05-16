# export structs
export MarginalEstimator, HazardMarginal, SurvivalMarginal, IncidenceMarginal

abstract type MarginalEstimator end

"""
    struct HazardMarginal <: MarginalEstimator

"""
struct HazardMarginal <: MarginalEstimator

    # dimensions
    D::Int64        # number of diseases
    L::Int64        # number of categorical predictors

    # times vector
    times::Vector{Float64}

    # posterior samples
    post_samples::Vector{Float64}

    # explicit constructor
    function HazardMarginal(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64})

        # initialize vectors
        post_samples = Array{Float64}(undef, 0)

        # create HazardMarginal
        return new(rf.D, 0, times, post_samples)

    end # HazardMarginal

    # explicit constructor
    function HazardMarginal(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, times::Vector{Float64})

        # initialize vectors
        post_samples = Array{Float64}(undef, 0)

        # create HazardMarginal
        return new(rf.D, cm.L, times, post_samples)

    end # HazardMarginal

end # struct

"""
    struct SurvivalMarginal <: MarginalEstimator

"""
struct SurvivalMarginal <: MarginalEstimator

    # dimensions
    L::Int64        # number of categorical predictors

    # times vector
    times::Vector{Float64}

    # posterior samples
    post_samples::Vector{Float64}

    # explicit constructor
    function SurvivalMarginal(_::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64})

        # initialize vectors
        post_samples = Array{Float64}(undef, 0)

        # create SurvivalMarginal
        return new(0, times, post_samples)

    end # SurvivalMarginal

    # explicit constructor
    function SurvivalMarginal(_::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, times::Vector{Float64})

        # initialize vectors
        post_samples = Array{Float64}(undef, 0)

        # create SurvivalMarginal
        return new(cm.L, times, post_samples)

    end # SurvivalMarginal

end # struct

"""
    struct IncidenceMarginal <: MarginalEstimator

"""
struct IncidenceMarginal <: MarginalEstimator

    # dimensions
    D::Int64        # number of diseases
    L::Int64        # number of categorical predictors

    # times vector
    times::Vector{Float64}

    # posterior samples
    post_samples::Vector{Float64}

    # explicit constructor
    function IncidenceMarginal(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64})

        # initialize vectors
        post_samples = Array{Float64}(undef, 0)

        # create IncidenceMarginal
        return new(rf.D, 0, times, post_samples)

    end # IncidenceMarginal

    # explicit constructor
    function IncidenceMarginal(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, times::Vector{Float64})

        # initialize vectors
        post_samples = Array{Float64}(undef, 0)

        # create IncidenceMarginal
        return new(rf.D, cm.L, times, post_samples)

    end # IncidenceMarginal

end # struct

"""
    append(estimator::HazardMarginal, rf::RestaurantFranchise)

"""
function append(estimator::HazardMarginal, rf::RestaurantFranchise)

    # compute hazard estimates
    tables = hazard_tables(rf, estimator.times)
    dishes = hazard_dishes(rf, estimator.times)
    base = hazard_base(rf, estimator.times)

    # combine estimates
    samples = tables .+ dishes .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    append(estimator::HazardMarginal, rf::RestaurantArray)

"""
function append(estimator::HazardMarginal, rf::RestaurantArray)

    # compute hazard estimates
    tables = hazard_tables(rf, estimator.times)
    base = hazard_base(rf, estimator.times)

    # combine estimates
    samples = tables .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    append(estimator::SurvivalMarginal, rf::RestaurantFranchise)

"""
function append(estimator::SurvivalMarginal, rf::RestaurantFranchise)

    # compute estimates
    tables = survival_tables(rf, estimator.times)
    dishes = survival_dishes(rf, estimator.times)
    base = survival_base(rf, estimator.times)

    # combine estimates
    samples = tables .* dishes .* base

    # append estimates
    append!(estimator.post_samples, samples)

end # append

"""
    append(estimator::SurvivalMarginal, rf::RestaurantArray)

"""
function append(estimator::SurvivalMarginal, rf::RestaurantArray)

    # compute estimates
    tables = survival_tables(rf, estimator.times)
    base = survival_base(rf, estimator.times)

    # combine estimates
    samples = tables .* base

    # append estimates
    append!(estimator.post_samples, samples)

end # append

"""
    append(estimator::IncidenceMarginal, rf::RestaurantFranchise)

"""
function append(estimator::IncidenceMarginal, rf::RestaurantFranchise)

    # compute incidence estimates
    tables = incidence_tables(rf, estimator.times)
    dishes = incidence_dishes(rf, estimator.times)
    base = incidence_base(rf, estimator.times)

    # combine estimates
    samples = tables .+ dishes .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    append(estimator::IncidenceMarginal, rf::RestaurantArray)

"""
function append(estimator::IncidenceMarginal, rf::RestaurantArray)

    # compute incidence estimates
    tables = incidence_tables(rf, estimator.times)
    base = incidence_base(rf, estimator.times)

    # combine estimates
    samples = tables .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    hazard_tables(rf::RestaurantFranchise, times::Vector{Float64})

"""
function hazard_tables(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), rf.D)

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # retrieve restaurant
        rest = rf.table_rest[table]

        # compute mass
        mass = tau_ratio(qtable, 0.0, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[dish]) 

        # store values
        for (t, time) in enumerate(times)
            estimate[t, rest] += rf.alpha * kernel(dish_value, time, rf.eta) * mass
        end
        
    end

    return estimate

end # hazard_tables

"""
    hazard_tables(rf::RestaurantArray, times::Vector{Float64})

"""
function hazard_tables(rf::RestaurantArray, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), rf.D)

    # tables level
    for (table, ntable) in enumerate(rf.n)  # loop on tables

        if ntable == 0  # no customers at table
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # retrieve restaurant
        rest = rf.table_rest[table]

        # precompute mass
        mass = tau_ratio(ntable, 0.0, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[table]) 

        # store values
        for (t, time) in enumerate(times)
            estimate[t, rest] += rf.alpha * kernel(dish_value, time, rf.eta) * mass
        end
        
    end

    return estimate

end # hazard_tables

"""
    hazard_dishes(rf::RestaurantFranchise, times::Vector{Float64})

"""
function hazard_dishes(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0   # no tables at dish
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute mass
        mass = tau(0.0, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[dish]) * tau_ratio(rdish, 0.0, rf.beta0, rf.sigma0; 
                posterior = rf.D * psi(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma)) 

        # store values
        for (t, time) in enumerate(times)
            estimate[t] += rf.alpha * kernel(dish_value, time, rf.eta) * mass
        end

    end

    return estimate

end # hazard_dishes

"""
    hazard_base(rf::RestaurantFranchise, times::Vector{Float64})

"""
function hazard_base(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.eta)

        # compute integrand
        return rf.alpha * kernel(x, t, rf.eta) * tau(0.0, rf.beta, rf.sigma; posterior = KInt) * tau(0.0, rf.beta0, rf.sigma0; 
                posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = rf.theta * integrate(x::Float64 -> f(x, time), rf.legendre; lower = 0.0, upper = time)
    end

    return estimate

end # hazard_base

"""
    hazard_base(rf::RestaurantArray, times::Vector{Float64})

"""
function hazard_base(rf::RestaurantArray, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # compute integrand
        return rf.alpha * kernel(x, t, rf.eta) * tau(0.0, rf.beta, rf.sigma; posterior = rf.alpha * KernelInt(x, rf.T, rf.eta)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = rf.theta * integrate(x::Float64 -> f(x, time), rf.legendre; lower = 0.0, upper = time)
    end

    return estimate

end # hazard_base

"""
    survival_tables(rf::RestaurantFranchise, times::Vector{Float64})

"""
function survival_tables(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]
        
        # store values
        for (t, time) in enumerate(times)
            estimate[t] += logtau_diff(qtable, rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; posterior = KInt)
        end
        
    end

    return exp.(estimate)

end # survival_tables

"""
    survival_tables(rf::RestaurantArray, times::Vector{Float64})

"""
function survival_tables(rf::RestaurantArray, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # tables level
    for (table, ntable) in enumerate(rf.n)  # loop on tables

        if ntable == 0  # no customers at table
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[table]
        
        # store values
        for (t, time) in enumerate(times)
            estimate[t] += logtau_diff(ntable, rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; posterior = KInt)
        end
        
    end

    return exp.(estimate)

end # survival_tables

"""
    survival_dishes(rf::RestaurantFranchise, times::Vector{Float64})

"""
function survival_dishes(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0   # no tables at dish
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            estimate[t] += logtau_diff(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; posterior = KInt),
                    rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        end

    end

    return exp.(estimate)

end # survival_dishes

"""
    survival_base(rf::RestaurantFranchise, times::Vector{Float64})

"""
function survival_base(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate function
    estimate = ones(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.eta)

        # compute integrand
        return psi(rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma; posterior = KInt), rf.beta0, rf.sigma0; 
            posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = exp( - rf.theta * integrate(x::Float64 -> f(x, time), rf.legendre; lower = 0.0, upper = time) )
    end

    return estimate

end # survival_base

"""
    survival_base(rf::RestaurantArray, times::Vector{Float64})

"""
function survival_base(rf::RestaurantArray, times::Vector{Float64})

    # initialize estimate function
    estimate = ones(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # compute integrand
        return psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma; posterior = rf.alpha * KernelInt(x, rf.T, rf.eta)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = exp( - rf.theta * rf.D * integrate(x::Float64 -> f(x, time), rf.legendre; lower = 0.0, upper = time) )
    end

    return estimate

end # survival_base

"""
    incidence_tables(rf::RestaurantFranchise, times::Vector{Float64})

"""
function incidence_tables(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), rf.D)

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # retrieve restaurant
        rest = rf.table_rest[table]
        
        # store values
        for (t, time) in enumerate(times)
            estimate[t, rest] += rf.alpha * kernel(dish_value, time, rf.eta) * tau_ratio(qtable, 
                    rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; posterior = KInt)
        end
        
    end

    return estimate

end # incidence_tables

"""
    incidence_tables(rf::RestaurantArray, times::Vector{Float64})

"""
function incidence_tables(rf::RestaurantArray, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), rf.D)

    # tables level
    for (table, ntable) in enumerate(rf.n)  # loop on tables

        if ntable == 0  # no customers at table
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[table]

        # retrieve restaurant
        rest = rf.table_rest[table]
        
        # store values
        for (t, time) in enumerate(times)
            estimate[t, rest] += rf.alpha * kernel(dish_value, time, rf.eta) * tau_ratio(ntable, 
                    rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; posterior = KInt)
        end
        
    end

    return estimate

end # incidence_tables

"""
    incidence_dishes(rf::RestaurantFranchise, times::Vector{Float64})

"""
function incidence_dishes(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0   # no tables at dish
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            estimate[t] += rf.alpha * kernel(dish_value, time, rf.eta) * tau(rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; 
                    posterior = KInt) * tau_ratio(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, rf.eta), rf.beta, rf.sigma; posterior = KInt), 
                    rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        end

    end

    return estimate

end # incidence_dishes

"""
    incidence_base(rf::RestaurantFranchise, times::Vector{Float64})

"""
function incidence_base(rf::RestaurantFranchise, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.eta)

        # compute integrand
        return rf.alpha * kernel(x, t, rf.eta) * tau(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma; 
                posterior = KInt) * tau(rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma; posterior = KInt), 
                rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) * pdf(rf.base_measure, x)
        return out

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = rf.theta * integrate(x::Float64 -> f(x, time), rf.legendre; lower = 0.0, upper = time)
    end

    return estimate

end # incidence_base

"""
    incidence_base(rf::RestaurantArray, times::Vector{Float64})

"""
function incidence_base(rf::RestaurantArray, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # compute integrand
        return rf.alpha * kernel(x, t, rf.eta) * tau(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma; 
                posterior = rf.alpha * KernelInt(x, rf.T, rf.eta)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = rf.theta * integrate(x::Float64 -> f(x, time), rf.legendre; lower = 0.0, upper = time)
    end

    return estimate

end # incidence_base

"""
    append(estimator::HazardMarginal, rf::RestaurantFranchise, CoxProd::Vector{Float64})

"""
function append(estimator::HazardMarginal, rf::RestaurantFranchise, CoxProd::Vector{Float64})

    # compute hazard estimates
    tables = hazard_tables(rf, CoxProd, estimator.times)
    dishes = hazard_dishes(rf, CoxProd, estimator.times)
    base = hazard_base(rf, CoxProd, estimator.times)

    # combine estimates
    samples = tables .+ dishes .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    append(estimator::HazardMarginal, rf::RestaurantArray, CoxProd::Vector{Float64})

"""
function append(estimator::HazardMarginal, rf::RestaurantArray, CoxProd::Vector{Float64})

    # compute hazard estimates
    tables = hazard_tables(rf, CoxProd, estimator.times)
    base = hazard_base(rf, CoxProd, estimator.times)

    # combine estimates
    samples = tables .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    append(estimator::SurvivalMarginal, rf::RestaurantFranchise, CoxProd::Vector{Float64})

"""
function append(estimator::SurvivalMarginal, rf::RestaurantFranchise, CoxProd::Vector{Float64})

    # compute estimates
    tables = survival_tables(rf, CoxProd, estimator.times)
    dishes = survival_dishes(rf, CoxProd, estimator.times)
    base = survival_base(rf, CoxProd, estimator.times)

    # combine estimates
    samples = tables .* dishes .* base

    # append estimates
    append!(estimator.post_samples, samples)

end # append

"""
    append(estimator::SurvivalMarginal, rf::RestaurantArray, CoxProd::Vector{Float64})

"""
function append(estimator::SurvivalMarginal, rf::RestaurantArray, CoxProd::Vector{Float64})

    # compute estimates
    tables = survival_tables(rf, CoxProd, estimator.times)
    base = survival_base(rf, CoxProd, estimator.times)

    # combine estimates
    samples = tables .* base

    # append estimates
    append!(estimator.post_samples, samples)

end # append

"""
    append(estimator::IncidenceMarginal, rf::RestaurantFranchise, CoxProd::Vector{Float64})

"""
function append(estimator::IncidenceMarginal, rf::RestaurantFranchise, CoxProd::Vector{Float64})

    # compute incidence estimates
    tables = incidence_tables(rf, CoxProd, estimator.times)
    dishes = incidence_dishes(rf, CoxProd, estimator.times)
    base = incidence_base(rf, CoxProd, estimator.times)

    # combine estimates
    samples = tables .+ dishes .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    append(estimator::IncidenceMarginal, rf::RestaurantArray, CoxProd::Vector{Float64})

"""
function append(estimator::IncidenceMarginal, rf::RestaurantArray, CoxProd::Vector{Float64})

    # compute incidence estimates
    tables = incidence_tables(rf, CoxProd, estimator.times)
    base = incidence_base(rf, CoxProd, estimator.times)

    # combine estimates
    samples = tables .+ base

    # append estimates
    append!(estimator.post_samples, vec(samples))

end # append

"""
    hazard_tables(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function hazard_tables(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), rf.D)

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # retrieve restaurant
        rest = rf.table_rest[table]

        # compute mass
        mass = tau_ratio(qtable, 0.0, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[dish]) 

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t, l, rest] += rf.alpha * kernel(dish_value, time, cp, rf.eta) * mass
            end
        end
        
    end

    return estimate

end # hazard_tables

"""
    hazard_tables(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function hazard_tables(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), rf.D)

    # tables level
    for (table, ntable) in enumerate(rf.n)  # loop on tables

        if ntable == 0  # no customers at table
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # retrieve restaurant
        rest = rf.table_rest[table]

        # compute mass
        mass = tau_ratio(ntable, 0.0, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[table]) 

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t, l, rest] += rf.alpha * kernel(dish_value, time, cp, rf.eta) * mass
            end
        end
        
    end

    return estimate

end # hazard_tables

"""
    hazard_dishes(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function hazard_dishes(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0   # no tables at dish
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = tau(0.0, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[dish]) * tau_ratio(rdish, 0.0, rf.beta0, rf.sigma0; 
                posterior = rf.D * psi(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma)) 

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] += rf.alpha * kernel(dish_value, time, cp, rf.eta) * mass
            end
        end

    end

    return estimate

end # hazard_dishes

"""
    hazard_base(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function hazard_base(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)

        # compute integrand
        return rf.alpha * kernel(x, t, cp, rf.eta) * tau(0.0, rf.beta, rf.sigma; posterior = KInt) * tau(0.0, rf.beta0, rf.sigma0; 
                posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = rf.theta * integrate(x::Float64 -> f(x, time, cp), rf.legendre; lower = 0.0, upper = time)
        end
    end

    return estimate

end # hazard_base

"""
    hazard_base(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function hazard_base(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # compute integrand
        return rf.alpha * kernel(x, t, cp, rf.eta) * tau(0.0, rf.beta, rf.sigma; posterior = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = rf.theta * integrate(x::Float64 -> f(x, time, cp), rf.legendre; lower = 0.0, upper = time)
        end
    end

    return estimate

end # hazard_base

"""
    survival_tables(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_tables(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]
        
        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] += logtau_diff(qtable, rf.alpha * KernelInt(dish_value, time, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt)
            end
        end
        
    end

    return exp.(estimate)

end # survival_tables

"""
    survival_tables(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_tables(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # tables level
    for (table, ntable) in enumerate(rf.n)  # loop on tables

        if ntable == 0  # no customers at table
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[table]
        
        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] += logtau_diff(ntable, rf.alpha * KernelInt(dish_value, time, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt)
            end
        end
        
    end

    return exp.(estimate)

end # survival_tables

"""
    survival_dishes(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_dishes(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0   # no tables at dish
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] += logtau_diff(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt), 
                        rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
            end
        end

    end

    return exp.(estimate)

end # survival_dishes

"""
    survival_base(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_base(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate function
    estimate = ones(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)

        # compute integrand
        return psi(rf.D * psi(rf.alpha * KernelInt(x, t, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt), rf.beta0, rf.sigma0; 
            posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = exp( - rf.theta * integrate(x::Float64 -> f(x, time, cp), rf.legendre; lower = 0.0, upper = time) )
        end
    end

    return estimate

end # survival_base

"""
    survival_base(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_base(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate function
    estimate = ones(length(times), 1)

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # compute integrand
        return psi(rf.alpha * KernelInt(x, t, cp, rf.eta), rf.beta, rf.sigma; posterior = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = exp( - rf.theta * rf.D * integrate(x::Float64 -> f(x, time, cp), rf.legendre; lower = 0.0, upper = time) )
        end
    end

    return estimate

end # survival_base

"""
    incidence_tables(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_tables(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), rf.D)

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # retrieve restaurant
        rest = rf.table_rest[table]
        
        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t, l, rest] += rf.alpha * kernel(dish_value, time, cp, rf.eta) * tau_ratio(qtable, 
                        rf.alpha * KernelInt(dish_value, time, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt)
            end
        end
        
    end

    return estimate

end # incidence_tables

"""
    incidence_tables(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_tables(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), rf.D)

    # tables level
    for (table, ntable) in enumerate(rf.n)  # loop on tables

        if ntable == 0  # no customers at table
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[table]

        # retrieve restaurant
        rest = rf.table_rest[table]
        
        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t, l, rest] += rf.alpha * kernel(dish_value, time, cp, rf.eta) * tau_ratio(ntable, rf.alpha * KernelInt(dish_value, time, cp, rf.eta), 
                        rf.beta, rf.sigma; posterior = KInt)
            end
        end
        
    end

    return estimate

end # incidence_tables

"""
    incidence_dishes(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_dishes(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0   # no tables at dish
            continue
        end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] = rf.alpha * kernel(dish_value, time, cp, rf.eta) * tau(rf.alpha * KernelInt(dish_value, time, cp, rf.eta), rf.beta, rf.sigma; 
                        posterior = KInt) * tau_ratio(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt), 
                        rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
            end
        end

    end

    return estimate

end # incidence_dishes

"""
    incidence_base(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_base(rf::RestaurantFranchise, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)

        # compute integrand
        return rf.alpha * kernel(x, t, cp, rf.eta) * tau(rf.alpha * KernelInt(x, t, cp, rf.eta), rf.beta, rf.sigma; 
                posterior = KInt) * tau(rf.D * psi(rf.alpha * KernelInt(x, t, cp, rf.eta), rf.beta, rf.sigma; posterior = KInt), rf.beta0, rf.sigma0; 
                posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = rf.theta * integrate(x::Float64 -> f(x, time, cp), rf.legendre; lower = 0.0, upper = time)
        end
    end

    return estimate

end # incidence_base

"""
    incidence_base(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_base(rf::RestaurantArray, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # compute integrand
        return rf.alpha * kernel(x, t, cp, rf.eta) * tau(rf.alpha * KernelInt(x, t, cp, rf.eta), rf.beta, rf.sigma; 
                posterior = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)) * pdf(rf.base_measure, x)

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = rf.theta * integrate(x::Float64 -> f(x, time, cp), rf.legendre; lower = 0.0, upper = time)
        end
    end

    return estimate

end # incidence_base

"""
    append(estimator::Estimator, rf::Restaurants, CoxProd::Union{Vector{Float64},Nothing})

"""
function append(estimator::Estimator, rf::Restaurants, CoxProd::Union{Vector{Float64},Nothing})

    # compute survival estimates
    tables = survival_tables(rf, CoxProd, estimator.times)
    dishes = survival_dishes(rf, CoxProd, estimator.times) 
    base = survival_base(rf, CoxProd, estimator.times)

    # append estimates
    append!(estimator.survival_samples, exp.(tables .+ dishes .+ base))

    # compute incidence estimates
    tables = incidence_tables(rf, CoxProd, estimator.times)
    dishes = incidence_dishes(rf, CoxProd, estimator.times)
    base = incidence_base(rf, CoxProd, estimator.times)

    # append estimates
    append!(estimator.hazard_samples, vec(tables .+ dishes .+ base))

end # append

"""
    survival_tables(rf::Restaurants, _::Nothing, times::Vector{Float64})

"""
function survival_tables(rf::Restaurants, _::Nothing, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        # no customers at table
        if qtable == 0 continue end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]
        
        # store values
        for (t, time) in enumerate(times)
            estimate[t] += logtau_diff(qtable, rf.alpha * KernelInt(dish_value, time, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt)
        end
        
    end

    return estimate

end # survival_tables

"""
    survival_tables(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_tables(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        # no customers at table
        if qtable == 0 continue end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]
        
        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] += logtau_diff(qtable, rf.alpha * KernelInt(dish_value, time, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt)
            end
        end
        
    end

    return estimate

end # survival_tables

"""
    survival_dishes(rf::Restaurants, _::Nothing, times::Vector{Float64})

"""
function survival_dishes(rf::Restaurants, _::Nothing, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    if !rf.hierarchical     # restaurant array
        return estimate
    end

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        # no tables at dish
        if rdish == 0 continue end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            estimate[t] += logtau_diff(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt), rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        end

    end

    return estimate

end # survival_dishes

"""
    survival_dishes(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_dishes(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    if !rf.hierarchical     # restaurant array
        return estimate
    end

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        # no tables at dish
        if rdish == 0 continue end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] += logtau_diff(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt), rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
            end
        end

    end

    return estimate

end # survival_dishes

"""
    survival_base(rf::Restaurants, _::Nothing, times::Vector{Float64})

"""
function survival_base(rf::Restaurants, _::Nothing, times::Vector{Float64})

    # initialize estimate function
    estimate = ones(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, nothing, rf.kappa)

        # compute integrand
        feval = rf.D * psi(rf.alpha * KernelInt(x, t, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt)

        # compute integrand
        if rf.hierarchical  # restaurant franchise
            feval = psi(feval, rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) 
        end

        # compute integrand
        return feval       

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = - rf.theta * integrate(x::Float64 -> f(x, time), 0.0, time)
    end

    return estimate

end # survival_base

"""
    survival_base(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function survival_base(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate function
    estimate = ones(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa)

        # compute integrand
        feval = rf.D * psi(rf.alpha * KernelInt(x, t, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt)

        # compute integrand
        if rf.hierarchical  # restaurant franchise
            feval = psi(feval, rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) 
        end

        # compute integrand
        return feval

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = - rf.theta * integrate(x::Float64 -> f(x, time, cp), 0.0, time)
        end
    end

    return estimate

end # survival_base

"""
    incidence_tables(rf::Restaurants, _::Nothing, times::Vector{Float64})

"""
function incidence_tables(rf::Restaurants, _::Nothing, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), rf.D)

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        # no customers at table
        if qtable == 0 continue end

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # retrieve restaurant
        rest = rf.table_rest[table]
        
        # store values
        for (t, time) in enumerate(times)
            estimate[t, rest] += rf.alpha * kernel(dish_value, time, nothing, rf.kappa) * 
                    tau_ratio(qtable, rf.alpha * KernelInt(dish_value, time, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt)
        end
        
    end

    return estimate

end # incidence_tables

"""
    incidence_tables(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_tables(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd), rf.D)

    # tables level
    for (table, qtable) in enumerate(rf.q)  # loop on tables

        # no customers at table
        if qtable == 0 continue end

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
                estimate[t, l, rest] += rf.alpha * kernel(dish_value, time, cp, rf.kappa) * 
                        tau_ratio(qtable, rf.alpha * KernelInt(dish_value, time, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt)
            end
        end
        
    end

    return estimate

end # incidence_tables

"""
    incidence_dishes(rf::Restaurants, _::Nothing, times::Vector{Float64})

"""
function incidence_dishes(rf::Restaurants, _::Nothing, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    if !rf.hierarchical     # restaurant array
        return estimate
    end

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        # no tables at dish
        if rdish == 0 continue end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            estimate[t] += rf.alpha * kernel(dish_value, time, nothing, rf.kappa) * 
                    tau(rf.alpha * KernelInt(dish_value, time,nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt) * 
                    tau_ratio(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt), rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        end

    end

    return estimate

end # incidence_dishes

"""
    incidence_dishes(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_dishes(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    if !rf.hierarchical     # restaurant array
        return estimate
    end

    # dish level
    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        # no tables at dish
        if rdish == 0 continue end

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # store values
        for (t, time) in enumerate(times)
            for (l, cp) in enumerate(CoxProd)
                estimate[t,l] = rf.alpha * kernel(dish_value, time, cp, rf.kappa) * 
                        tau(rf.alpha * KernelInt(dish_value, time, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt) * 
                        tau_ratio(rdish, rf.D * psi(rf.alpha * KernelInt(dish_value, time, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt), rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
            end
        end

    end

    return estimate

end # incidence_dishes

"""
    incidence_base(rf::Restaurants, _::Nothing, times::Vector{Float64})

"""
function incidence_base(rf::Restaurants, _::Nothing, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, nothing, rf.kappa)

        # compute integrand
        feval = rf.alpha * kernel(x, t, nothing, rf.kappa) * tau(rf.alpha * KernelInt(x, t, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt)

        # compute integrand
        if rf.hierarchical  # restaurant franchise
            feval *= tau(rf.D * psi(rf.alpha * KernelInt(x, t, nothing, rf.kappa), rf.beta, rf.sigma, posterior = KInt), rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma)) 
        end

        return feval

    end

    # store values
    for (t, time) in enumerate(times)
        estimate[t] = rf.theta * integrate(x::Float64 -> f(x, time), 0.0, time)
    end

    return estimate

end # incidence_base

"""
    incidence_base(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

"""
function incidence_base(rf::Restaurants, CoxProd::Vector{Float64}, times::Vector{Float64})

    # initialize estimate vector
    estimate = zeros(length(times), length(CoxProd))

    # base level integrand
    function f(x::Float64, t::Float64, cp::Float64)

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa)

        # compute integrand
        feval = rf.alpha * kernel(x, t, cp, rf.kappa) * tau(rf.alpha * KernelInt(x, t, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt)

        # compute integrand
        if rf.hierarchical  # restaurant franchise
            feval *= tau(rf.D * psi(rf.alpha * KernelInt(x, t, cp, rf.kappa), rf.beta, rf.sigma, posterior = KInt), rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        end

        # compute integrand
        return feval

    end

    # store values
    for (t, time) in enumerate(times)
        for (l, cp) in enumerate(CoxProd)
            estimate[t,l] = rf.theta * integrate(x::Float64 -> f(x, time, cp), 0.0, time)
        end
    end

    return estimate

end # incidence_base

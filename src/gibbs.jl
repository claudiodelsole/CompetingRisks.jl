# import from Distributions
import Distributions: pdf
import ProgressMeter: @showprogress

# export functions
export Gibbs

"""
    Gibbs(rf::Union{RestaurantFranchise,RestaurantArray}, marginal_estimators::Vector{TypeM}, conditional_estimators::Vector{TypeC}, 
            num_samples::@NamedTuple{n::Int64, m::Int64}; 
            thin::Int64 = 10, burn_in::Int64 = 0, started::Bool = false) where {TypeM <: MarginalEstimator, TypeC <: ConditionalEstimator}

"""
function Gibbs(rf::Union{RestaurantFranchise,RestaurantArray}, marginal_estimators::Vector{TypeM}, conditional_estimators::Vector{TypeC}, 
        num_samples::@NamedTuple{n::Int64, m::Int64}; 
        thin::Int64 = 10, burn_in::Int64 = 0, started::Bool = false) where {TypeM <: MarginalEstimator, TypeC <: ConditionalEstimator}
    
    # create Diagnostics
    dgn = Diagnostics()

    # initialize latent variables
    if started == false
        Gibbs_initialize(rf)
    end

    # burn-in phase
    @showprogress for _ in 1:burn_in
        Gibbs_step(rf, dgn)
    end

    # inference phase
    @showprogress for _ in 1:num_samples.n

        # run chain
        for _ in 1:thin
            Gibbs_step(rf, dgn)
        end

        # inference
        for est in marginal_estimators
            append(est, rf)
        end

        # posterior sampling
        for _ in 1:num_samples.m

            # posterior sampling
            CRMs = sample_measures(rf)

            # inference
            for est in conditional_estimators
                append(est, CRMs, rf.alpha, rf.eta)
            end

        end

    end

    # return Diagnostics
    return dgn

end # Gibbs

"""
    Gibbs_initialize(rf::Union{RestaurantFranchise,RestaurantArray})

"""
function Gibbs_initialize(rf::Union{RestaurantFranchise,RestaurantArray})

    # precompute quantities
    precompute_mass_base(rf)

    # sample allocations
    for (cust, disease) in enumerate(rf.Delta)
        if disease != 0
            sample_allocation_init(cust, rf)
        end
    end

    # resampling
    resample_dishes(rf)
    resample_theta(rf)
    resample_alpha(rf)
    resample_eta(rf)

end # Gibbs_initialize

"""
    Gibbs_step(rf::Union{RestaurantFranchise,RestaurantArray}, dgn::Diagnostics)

"""
function Gibbs_step(rf::Union{RestaurantFranchise,RestaurantArray}, dgn::Diagnostics)

    # sample allocations
    for (cust, disease) in enumerate(rf.Delta)
        if disease != 0
            sample_allocation(cust, rf)
        end
    end

    # resampling
    accept_dishes = resample_dishes(rf)
    resample_theta(rf)
    (accept_alpha, flag_alpha) = resample_alpha(rf)
    (accept_eta, flag_eta) = resample_eta(rf)

    # precompute quantities
    if flag_alpha || flag_eta precompute_mass_base(rf) end

    # diagnostics
    append(dgn, rf, accept_dishes, accept_alpha, accept_eta)

end # Gibbs_step

"""
    precompute_mass_base(rf::Union{RestaurantFranchise,RestaurantArray})

"""
function precompute_mass_base(rf::Union{RestaurantFranchise,RestaurantArray})

    if isnothing(rf.CoxProd)     # exchangeable model

        for (cust, disease) in enumerate(rf.Delta)
            if disease != 0
                rf.mass_base[cust] = mass_base(rf.T[cust], rf)
            end
        end

    else    # regression model

        for (cust, disease) in enumerate(rf.Delta)
            if disease != 0
                rf.mass_base[cust] = mass_base(rf.T[cust], rf.CoxProd[cust], rf)
            end
        end

    end

end # precompute_mass_base

"""
    mass_base(time::Float64, rf::Union{RestaurantFranchise,RestaurantArray})

"""
function mass_base(time::Float64, rf::Union{RestaurantFranchise,RestaurantArray})

    # integrand function
    f(x::Float64) = likelihood_base(x, time, rf) * pdf(rf.base_measure, x)

    # compute integral
    return integrate(f, rf.legendre; lower = 0.0, upper = time)

end # mass_base

"""
    likelihood_base(x::Float64, time::Float64, rf::RestaurantFranchise)

"""
function likelihood_base(x::Float64, time::Float64, rf::RestaurantFranchise)

    # precompute KernelInt
    KInt = rf.alpha * KernelInt(x, rf.T, rf.eta)
        
    # compute likelihood ratio
    return rf.alpha * kernel(x, time, rf.eta) * tau(KInt, rf.beta, rf.sigma) * tau(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

end # likelihood_base

"""
    likelihood_base(x::Float64, time::Float64, rf::RestaurantArray)

"""
function likelihood_base(x::Float64, time::Float64, rf::RestaurantArray)
        
    # compute likelihood ratio
    return rf.alpha * kernel(x, time, rf.eta) * tau(rf.alpha * KernelInt(x, rf.T, rf.eta), rf.beta, rf.sigma)

end # likelihood_base

"""
    sample_allocation_init(cust::Int64, rf::RestaurantFranchise)

"""
function sample_allocation_init(cust::Int64, rf::RestaurantFranchise)

    # step 2: compute masses
    (mass_tables, mass_dishes) = compute_mass(cust, rf)
    
    # step 3: sample from probabilities
    new_allocation(cust, mass_tables, mass_dishes, rf)

end # sample_allocation_init

"""
    sample_allocation_init(cust::Int64, rf::RestaurantArray)

"""
function sample_allocation_init(cust::Int64, rf::RestaurantArray)

    # step 2: compute masses
    mass_tables = compute_mass(cust, rf)
    
    # step 3: sample from probabilities
    new_allocation(cust, mass_tables, rf)

end # sample_allocation_init

"""
    sample_allocation(cust::Int64, rf::RestaurantFranchise)

"""
function sample_allocation(cust::Int64, rf::RestaurantFranchise)

    # step 1: remove customer
    remove_customer(cust, rf)

    # step 2: compute masses
    (mass_tables, mass_dishes) = compute_mass(cust, rf)
    
    # step 3: sample from probabilities
    new_allocation(cust, mass_tables, mass_dishes, rf)

end # sample_allocation

"""
    sample_allocation(cust::Int64, rf::RestaurantArray)

"""
function sample_allocation(cust::Int64, rf::RestaurantArray)

    # step 1: remove customer
    remove_customer(cust, rf)

    # step 2: compute masses
    mass_tables = compute_mass(cust, rf)
    
    # step 3: sample from probabilities
    new_allocation(cust, mass_tables, rf)

end # sample_allocation

"""
    remove_customer(cust::Int64, rf::RestaurantFranchise)

"""
function remove_customer(cust::Int64, rf::RestaurantFranchise)

    # retrieve customer data
    dish, table = rf.X[cust], rf.Z[cust]

    # update counts matrices
    rf.n[dish] -= 1         # customers per dish
    rf.q[table] -= 1        # customers per table

    if rf.q[table] == 0     # no customers at table
        rf.r[dish] -= 1             # tables per dish
        rf.table_dish[table] = 0    # no dish index per table
        rf.table_rest[table] = 0    # no rest index per table
    end

    # update latent variables
    rf.X[cust] = 0      # customer dish
    rf.Z[cust] = 0      # customer table

end # remove_customer

"""
    remove_customer(cust::Int64, rf::RestaurantArray)

"""
function remove_customer(cust::Int64, rf::RestaurantArray)

    # retrieve customer data
    table = rf.X[cust]

    # update counts matrices
    rf.n[table] -= 1    # customers per table

    if rf.n[table] == 0     # no customers at table
        rf.table_rest[table] = 0    # no rest index per table
    end

    # update latent variables
    rf.X[cust] = 0      # customer dish

end # remove_customer

"""
    compute_mass(cust::Int64, rf::RestaurantFranchise)

"""
function compute_mass(cust::Int64, rf::RestaurantFranchise)

    if isnothing(rf.CoxProd)     # exchangeable model

        # retrieve customer data
        time, rest = rf.T[cust], rf.Delta[cust]

        # case 1: sit at old table
        mass_tables = zeros(length(rf.q))       # initialize vector
        
        for (table, qtable) in enumerate(rf.q)  # loop on tables
            if rf.table_rest[table] == rest         # table in restaurant
                mass_tables[table] = mass_table(time, table, qtable, rf)
            end
        end

        # case 2: sit at new table, eating old dish
        mass_dishes = zeros(length(rf.r))       # initialize vector

        for (dish, rdish) in enumerate(rf.r)    # loop on dishes
            mass_dishes[dish] = mass_dish(time, dish, rdish, rf)
        end

    else    # regression model

        # retrieve customer data
        time, cp, rest = rf.T[cust], rf.CoxProd[cust], rf.Delta[cust]

        # case 1: sit at old table
        mass_tables = zeros(length(rf.q))       # initialize vector
        
        for (table, qtable) in enumerate(rf.q)  # loop on tables
            if rf.table_rest[table] == rest         # table in restaurant
                mass_tables[table] = mass_table(time, cp, table, qtable, rf)
            end
        end

        # case 2: sit at new table, eating old dish
        mass_dishes = zeros(length(rf.r))       # initialize vector

        for (dish, rdish) in enumerate(rf.r)    # loop on dishes
            mass_dishes[dish] = mass_dish(time, cp, dish, rdish, rf)
        end

    end

    # return masses
    return (mass_tables, mass_dishes)

end # compute_mass

"""
    compute_mass(cust::Int64, rf::RestaurantArray)

"""
function compute_mass(cust::Int64, rf::RestaurantArray)

    if isnothing(rf.CoxProd)     # exchangeable model

        # retrieve customer data
        time, rest = rf.T[cust], rf.Delta[cust]

        # initialize vector
        mass_tables = zeros(length(rf.n))

        # loop on tables
        for (table, ntable) in enumerate(rf.n)
            if rf.table_rest[table] == rest     # table in restaurant
                mass_tables[table] = mass_table(time, table, ntable, rf)
            end
        end

    else    # regression model

        # retrieve customer data
        time, cp, rest = rf.T[cust], rf.CoxProd[cust], rf.Delta[cust]

        # initialize vector
        mass_tables = zeros(length(rf.n))

        # loop on tables
        for (table, ntable) in enumerate(rf.n)
            if rf.table_rest[table] == rest     # table in restaurant
                mass_tables[table] = mass_table(time, cp, table, ntable, rf)
            end
        end

    end

    # return masses
    return mass_tables

end # compute_mass

"""
    mass_table(time::Float64, table::Int64, qtable::Int64, rf::RestaurantFranchise)

"""
function mass_table(time::Float64, table::Int64, qtable::Int64, rf::RestaurantFranchise)

    # initialize mass
    mass = 0.0

    if qtable != 0      # customers at table

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = rf.alpha * kernel(dish_value, time, rf.eta) * tau_ratio(qtable, rf.alpha * rf.KInt[dish], rf.beta, rf.sigma)

    end
    
    return mass

end # mass_table

"""
    mass_table(time::Float64, table::Int64, ntable::Int64, rf::RestaurantArray)

"""
function mass_table(time::Float64, table::Int64, ntable::Int64, rf::RestaurantArray)

    # initialize mass
    mass = 0.0

    if ntable != 0      # customers at table

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # compute mass
        mass = rf.alpha * kernel(dish_value, time, rf.eta) * tau_ratio(ntable, rf.alpha * rf.KInt[table], rf.beta, rf.sigma)

    end

    return mass
    
end # mass_table

"""
    mass_dish(time::Float64, dish::Int64, rdish::Int64, rf::RestaurantFranchise)

"""
function mass_dish(time::Float64, dish::Int64, rdish::Int64, rf::RestaurantFranchise)

    # initialize mass
    mass = 0.0

    if rdish != 0       # tables at dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = rf.alpha * kernel(dish_value, time, rf.eta) * tau(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma)
        mass *= tau_ratio(rdish, rf.D * psi(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    end

    return mass
    
end # mass_dish

"""
    new_allocation(cust::Int64, mass_tables::Vector{Float64}, mass_dishes::Vector{Float64}, rf::RestaurantFranchise)

"""
function new_allocation(cust::Int64, mass_tables::Vector{Float64}, mass_dishes::Vector{Float64}, rf::RestaurantFranchise)

    # retrieve customer data
    if isnothing(rf.CoxProd)    # exchangeable model
        time, rest = rf.T[cust], rf.Delta[cust]
    else    # regression model
        time, cp, rest = rf.T[cust], rf.CoxProd[cust], rf.Delta[cust]
    end

    # initialize latent variables
    dish = 0
    table = 0

    # retrieve base mass
    mass_base = rf.theta * rf.mass_base[cust]

    # masses vector
    masses = [sum(mass_tables), sum(mass_dishes), mass_base]

    # sample case
    # d = Categorical(masses / sum(masses))       # categorical distribution
    # case = rand(d)                              # sample case
    case = sample_categorical(masses)       # sample case

    if case == 1    # sit at old table

        # sample table
        # d = Categorical(mass_tables / sum(mass_tables))     # categorical distribution
        # table = rand(d)                                     # sample table
        table = sample_categorical(mass_tables)     # sample table

        # retrieve dish
        dish = rf.table_dish[table]

        # update counts matrices
        rf.n[dish] += 1         # customers per dish
        rf.q[table] += 1        # customers per table

    elseif case == 2    # sit at new table, eating old dish

        # sample dish
        # d = Categorical(mass_dishes / sum(mass_dishes))     # categorical distribution
        # dish = rand(d)                                      # sample dish
        dish = sample_categorical(mass_dishes)      # sample dish

        # update counts matrices
        rf.n[dish] += 1     # customers per dish
        rf.r[dish] += 1     # tables per  dish

        # find new table index
        table = find_table(rf)

        # update counts and lookup matrices
        rf.q[table] += 1                # customers per table
        rf.table_dish[table] = dish     # dish index per table
        rf.table_rest[table] = rest     # rest index per table

    elseif case == 3    # sit at new table, eating new dish

        # initialize dish value
        dish_value = 0.0

        # sample dish value
        flag = false
        while flag == false     # rejection sampling

            # sample dish proposal from base measure
            dish_value = rand(rf.base_measure)

            # acceptance probability
            if isnothing(rf.CoxProd)    # exchangeable model
                accept_prob = likelihood_base(dish_value, time, rf) / likelihood_base(time, time, rf)
            else    # regression model
                accept_prob = likelihood_base(dish_value, time, cp, rf) / likelihood_base(time, time, cp, rf)
            end

            if rand() < accept_prob     # accept proposal
                flag = true
            end

        end

        # find new dish index
        dish = find_dish(rf)

        # find new table index
        table = find_table(rf)

        # update latent variables
        rf.Xstar[dish] = dish_value

        # update precomputed variables
        if isnothing(rf.CoxProd)    # exchangeable model
            rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.eta)
        else    # regression model
            rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.eta)
        end

        # update counts vectors
        rf.n[dish] += 1         # customers per dish
        rf.r[dish] += 1         # tables per dish
        rf.q[table] += 1        # customers per table

        # update lookup vectors
        rf.table_dish[table] = dish     # dish index per table
        rf.table_rest[table] = rest     # rest index per table

    end # if

    # update latent variables
    rf.X[cust] = dish       # customer dish
    rf.Z[cust] = table      # customer table

end # new_allocation

"""
    new_allocation(cust::Int64, mass_tables::Vector{Float64}, rf::RestaurantArray)

"""
function new_allocation(cust::Int64, mass_tables::Vector{Float64}, rf::RestaurantArray)

    # retrieve customer data
    if isnothing(rf.CoxProd)    # exchangeable model
        time, rest = rf.T[cust], rf.Delta[cust]
    else    # regression model
        time, cp, rest = rf.T[cust], rf.CoxProd[cust], rf.Delta[cust]
    end

    # initialize latent table
    table = 0

    # retrieve base mass
    mass_base = rf.theta * rf.mass_base[cust]

    # masses vector
    masses = [sum(mass_tables), mass_base]

    # sample case
    # d = Categorical(masses / sum(masses))       # categorical distribution
    # case = rand(d)                              # sample case
    case = sample_categorical(masses)       # sample case

    if case == 1    # sit at old table

        # sample table
        # d = Categorical(mass_tables / sum(mass_tables))     # categorical distribution
        # table = rand(d)                                     # sample table
        table = sample_categorical(mass_tables)     # sample table

        # update counts matrices
        rf.n[table] += 1        # customers per table

    elseif case == 2    # sit at new table

        # initialize dish value
        dish_value = 0.0

        # sample dish value
        flag = false
        while flag == false     # rejection sampling

            # sample dish proposal from base measure
            dish_value = rand(rf.base_measure)

            # acceptance probability
            if isnothing(rf.CoxProd)    # exchangeable model
                accept_prob = likelihood_base(dish_value, time, rf) / likelihood_base(time, time, rf)
            else    # regression model
                accept_prob = likelihood_base(dish_value, time, cp, rf) / likelihood_base(time, time, cp, rf)
            end

            if rand() < accept_prob     # accept proposal
                flag = true
            end

        end

        # find new table index
        table = find_table(rf)

        # update latent variables
        rf.Xstar[table] = dish_value

        # update precomputed variables
        if isnothing(rf.CoxProd)    # exchangeable model
            rf.KInt[table] = KernelInt(dish_value, rf.T, rf.eta)
        else    # regression model
            rf.KInt[table] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.eta)
        end

        # update counts vectors
        rf.n[table] += 1        # customers per table

        # update lookup vectors
        rf.table_rest[table] = rest      # rest index per table

    end # if

    # update latent variables
    rf.X[cust] = table          # customer table

end # new_allocation

"""
    find_table(rest::Int64, rf::RestaurantFranchise)

"""
function find_table(rf::RestaurantFranchise)

    # find first empty table at restaurant
    table = findfirst(rf.q .== 0)

    if (table === nothing)      # no empty table

        # tables
        rtot = length(rf.q)

        # double vectors lengths
        append!(rf.q, zeros(Int64, rtot))
        append!(rf.table_dish, zeros(Int64, rtot))
        append!(rf.table_rest, zeros(Int64, rtot))

        # new table index
        table = rtot + 1

    end

    return table

end # find_table

"""
    find_table(rf::RestaurantArray)

"""
function find_table(rf::RestaurantArray)

    # find first empty table index
    table = findfirst(rf.n .== 0)

    if (table === nothing)       # no empty table

        # tables
        k = length(rf.n)

        # double vectors lengths
        append!(rf.Xstar, zeros(Float64, k))
        append!(rf.KInt, zeros(Float64, k))
        append!(rf.n, zeros(Int64, k))
        append!(rf.table_rest, zeros(Int64, k))

        # new table index
        table = k + 1
        
    end

    return table

end # find_table

"""
    find_dish(rf::RestaurantFranchise)

"""
function find_dish(rf::RestaurantFranchise)

    # find first empty dish index
    dish = findfirst(rf.r .== 0)

    if (dish === nothing)       # no empty dish

        # dishes
        k = length(rf.r)

        # double vectors lengths
        append!(rf.Xstar, zeros(Float64, k))
        append!(rf.KInt, zeros(Float64, k))
        append!(rf.n, zeros(Int64, k))
        append!(rf.r, zeros(Int64, k))

        # new dish index
        dish = k + 1
        
    end

    return dish

end # find_dish

"""
    Gibbs(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, marginal_estimators::Vector{TypeM}, conditional_estimators::Vector{TypeC}, 
            num_samples::@NamedTuple{n::Int64, m::Int64}; 
            thin::Int64 = 10, burn_in::Int64 = 0, started::Bool = false) where {TypeM <: MarginalEstimator, TypeC <: ConditionalEstimator}

"""
function Gibbs(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, marginal_estimators::Vector{TypeM}, conditional_estimators::Vector{TypeC}, 
        num_samples::@NamedTuple{n::Int64, m::Int64}; 
        thin::Int64 = 10, burn_in::Int64 = 0, started::Bool = false) where {TypeM <: MarginalEstimator, TypeC <: ConditionalEstimator}
    
    # create Diagnostics
    dgn = Diagnostics()

    # initialize latent variables
    if started == false
        Gibbs_initialize(rf, cm)
    end

    # burn-in phase
    @showprogress for _ in 1:burn_in
        Gibbs_step(rf, cm, dgn)
    end

    # inference phase
    @showprogress for _ in 1:num_samples.n

        # run chain
        for _ in 1:thin
            Gibbs_step(rf, cm, dgn)
        end

        # inference
        for est in marginal_estimators
            append(est, rf, pushfirst!(exp.(cm.xi), 1.0))
        end

        # posterior sampling
        for _ in 1:num_samples.m

            # posterior sampling
            CRMs = sample_measures(rf)

            # inference
            for est in conditional_estimators
                append(est, CRMs, rf.alpha, rf.eta, pushfirst!(exp.(cm.xi), 1.0))
            end

        end

    end

    # return Diagnostics
    return dgn

end # Gibbs

"""
    Gibbs_initialize(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel)

"""
function Gibbs_initialize(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel)

    # precompute quantities
    precompute_mass_base(rf)

    # sample allocations
    for (cust, disease) in enumerate(rf.Delta)
        if disease != 0
            sample_allocation_init(cust, rf)
        end
    end

    # resampling
    resample_dishes(rf)
    resample_theta(rf)
    resample_alpha(rf)
    resample_eta(rf)
    resample_coefficients(rf, cm)

end # Gibbs_initialize

"""
    Gibbs_step(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, dgn::Diagnostics)

"""
function Gibbs_step(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, dgn::Diagnostics)

    # sample allocations
    for (cust, disease) in enumerate(rf.Delta)
        if disease != 0
            sample_allocation(cust, rf)
        end
    end

    # resampling
    accept_dishes = resample_dishes(rf)
    resample_theta(rf)
    (accept_alpha, flag_alpha) = resample_alpha(rf)
    (accept_eta, flag_eta) = resample_eta(rf)
    (accept_coeffs, flag_coeffs) = resample_coefficients(rf, cm)

    # precompute quantities
    if (flag_alpha || flag_eta ||flag_coeffs) precompute_mass_base(rf) end

    # diagnostics
    append(dgn, rf, cm, accept_dishes, accept_alpha, accept_eta, accept_coeffs)

end # Gibbs_step

"""
    mass_base(time::Float64, cp::Float64, rf::Union{RestaurantFranchise,RestaurantArray})

"""
function mass_base(time::Float64, cp::Float64, rf::Union{RestaurantFranchise,RestaurantArray})

    # integrand function
    f(x::Float64) = likelihood_base(x, time, cp, rf) * pdf(rf.base_measure, x)

    # compute integral
    return integrate(f, rf.legendre; lower = 0.0, upper = time)

end # mass_base

"""
    likelihood_base(x::Float64, time::Float64, cp::Float64, rf::RestaurantFranchise)

"""
function likelihood_base(x::Float64, time::Float64, cp::Float64, rf::RestaurantFranchise)

    # precompute KernelInt
    KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)
        
    # compute likelihood ratio
    return rf.alpha * kernel(x, time, cp, rf.eta) * tau(KInt, rf.beta, rf.sigma) * tau(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

end # likelihood_base

"""
    likelihood_base(x::Float64, time::Float64, cp::Float64, rf::RestaurantArray)

"""
function likelihood_base(x::Float64, time::Float64, cp::Float64, rf::RestaurantArray)
        
    # compute likelihood ratio
    return rf.alpha * kernel(x, time, cp, rf.eta) * tau(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta), rf.beta, rf.sigma)

end # likelihood_base

"""
    mass_table(time::Float64, cp::Float64, table::Int64, qtable::Int64, rf::RestaurantFranchise)

"""
function mass_table(time::Float64, cp::Float64, table::Int64, qtable::Int64, rf::RestaurantFranchise)

    # initialize mass
    mass = 0.0

    if qtable != 0      # customers at table

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = rf.alpha * kernel(dish_value, time, cp, rf.eta) * tau_ratio(qtable, rf.alpha * rf.KInt[dish], rf.beta, rf.sigma)

    end
    
    return mass

end # mass_table

"""
    mass_table(time::Float64, cp::Float64, table::Int64, ntable::Int64, rf::RestaurantArray)

"""
function mass_table(time::Float64, cp::Float64, table::Int64, ntable::Int64, rf::RestaurantArray)

    # initialize mass
    mass = 0.0

    if ntable != 0      # customers at table

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # compute mass
        mass = rf.alpha * kernel(dish_value, time, cp, rf.eta) * tau_ratio(ntable, rf.alpha * rf.KInt[table], rf.beta, rf.sigma)

    end

    return mass
    
end # mass_table

"""
    mass_dish(time::Float64, cp::Float64, dish::Int64, rdish::Int64, rf::RestaurantFranchise)

"""
function mass_dish(time::Float64, cp::Float64, dish::Int64, rdish::Int64, rf::RestaurantFranchise)

    # initialize mass
    mass = 0.0

    if rdish != 0       # tables at dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = rf.alpha * kernel(dish_value, time, cp, rf.eta) * tau(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma)
        mass *= tau_ratio(rdish, rf.D * psi(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    end

    return mass
    
end # mass_dish

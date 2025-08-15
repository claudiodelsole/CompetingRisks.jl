# export functions
export posterior_sampling

"""
    posterior_sampling(data::DataFrame, cmprsk::CompetingRisksModel;
        nsamples::Int64 = 1000, thin::Int64 = 10, burn_in::Int64 = 1000, nsamples_crms::Int64 = 1,
        times::Vector{Float64} = collect(range(0.0, maximum(rf.T), 200)))

Gibbs sampling algorithm for posterior inference in mixture hazard models for competing risks.

# Main arguments:
- `data::DataFrame`: contains survival times in `data.T` and event types in `data.Delta` 
- `cmprsk::CompetingRisksModel`: kernel specification, random measures parameters, hierachical or independent model, regression or exchangeable data

# Algorithm arguments:
- `nsamples::Int64 = 1000`: number of posterior samples
- `thin::Int64 = 10`: thinning parameter
- `burn_in::Int64 = 1000`: initial discarded samples
- `nsamples_crms::Int64 = 1`: number of independent samples from posterior random measures for each sample from the latent partition

Parameter `times::Vector{Float64}` contains evaluation timepoints for the estimators. Default is 200 points between 0.0 and the maximum observed survival time.

# `Outputs:
- `marginal_estimator::Estimator`: collection of posterior samples obtained with marginal algorithm (not true posteriors)
- `posterior_estimator::Estimator`: collection of posterior samples
- `params::Parameters`: collection of posterior samples of model parameters
"""
function posterior_sampling(data::DataFrame, cmprsk::CompetingRisksModel;
        nsamples::Int64 = 1000, thin::Int64 = 10, burn_in::Int64 = 1000, nsamples_crms::Int64 = 1,
        times::Vector{Float64} = collect(range(0.0, maximum(rf.T), 200)))

    # create Restaurants
    rf = Restaurants(data, cmprsk.KernelType(), cmprsk.beta, cmprsk.sigma, cmprsk.beta0, cmprsk.sigma0, cmprsk.hierarchical)

    # create CoxModel
    cm = cmprsk.regression ? CoxModel(data) : nothing

    # initialize estimators
    marginal_estimator, posterior_estimator = Estimator(rf, cm, times), Estimator(rf, cm, times)
    
    # initialize Parameters
    params = Parameters(typeof(rf.kernelpars))

    # initialize latent variables
    initialize(rf, cm)

    # burn-in phase
    @showprogress for _ in range(1, burn_in)
        gibbs_step(rf, cm, params)
    end

    # inference phase
    @showprogress for _ in range(1, nsamples)

        # run chain
        for _ in range(1, thin)
            gibbs_step(rf, cm, params)
        end

        # marginal inference
        append(marginal_estimator, rf, isnothing(cm) ? nothing : pushfirst!(exp.(cm.eta), 1.0))

        # posterior inference
        for _ in range(1, nsamples_crms)

            # posterior sampling
            crms = sample_measures(rf, maximum(times))

            # inference
            append(posterior_estimator, crms, rf.kernelpars, isnothing(cm) ? nothing : pushfirst!(exp.(cm.eta), 1.0))

        end

    end

    # return Parameters
    return marginal_estimator, posterior_estimator, params

end # posterior_sampling

"""
    initialize(rf::Restaurants, cm::Union{CoxModel,Nothing})

"""
function initialize(rf::Restaurants, cm::Union{CoxModel,Nothing})

    # precompute quantities
    precompute_mass_base(rf)

    # sample allocations
    for (obs, cause) in enumerate(rf.Delta)
        if cause != 0
            sample_allocation(obs, rf, init = true)
        end
    end

    # resampling dishes
    resample_dishes(rf)
    
    # resampling hyperparameters
    resample_theta(rf)

    # resampling kernel parameters
    resample_kernelpars(rf)

    # resampling coefficients
    if !isnothing(cm) resample_coefficients(rf, cm) end

end # initialize

"""
    gibbs_step(rf::Restaurants, cm::Union{CoxModel,Nothing}, params::Parameters)

"""
function gibbs_step(rf::Restaurants, cm::Union{CoxModel,Nothing}, params::Parameters)

    # sample allocations
    for (obs, cause) in enumerate(rf.Delta)
        if cause != 0
            sample_allocation(obs, rf)
        end
    end

    # resampling dishes
    accept_dishes = resample_dishes(rf)

    # resampling hyperparameters
    resample_theta(rf)

    # resampling hyperparameters
    (accept_kernelpars, flag_kernelpars) = resample_kernelpars(rf)

    # resampling coefficients
    (accept_coeffs, flag_coeffs) = zeros(Float64, 1), false
    if !isnothing(cm)
        (accept_coeffs, flag_coeffs) = resample_coefficients(rf, cm)
    end

    # precompute quantities
    if (flag_kernelpars || flag_coeffs) precompute_mass_base(rf) end

    # Parameters
    append(params, rf, cm, accept_dishes, accept_kernelpars, accept_coeffs)

end # gibbs_step

"""
    precompute_mass_base(rf::Restaurants)

"""
function precompute_mass_base(rf::Restaurants)

    for (obs, cause) in enumerate(rf.Delta)
        if cause != 0
            rf.mass_base[obs] = mass_base(rf.T[obs], isnothing(rf.CoxProd) ? nothing : rf.CoxProd[obs], rf)
        end
    end

end # precompute_mass_base

"""
    mass_base(time::Float64, cp::Union{Float64,Nothing}, rf::Restaurants)

"""
function mass_base(time::Float64, cp::Union{Float64,Nothing}, rf::Restaurants)

    # compute integral
    return integrate(x -> likelihood_base(x, time, cp, rf), 0.0, time)

end # mass_base

"""
    likelihood_base(x::Float64, time::Float64, cp::Union{Float64,Nothing}, rf::Restaurants)

"""
function likelihood_base(x::Float64, time::Float64, cp::Union{Float64,Nothing}, rf::Restaurants)

    # compute KernelInt
    KInt = KernelInt(x, rf.T, rf.CoxProd, rf.kernelpars)

    # compute likelihood ratio
    loglik = kernel(x, time, cp, rf.kernelpars) * tau(KInt, rf.beta, rf.sigma)

    # compute likelihood ratio
    if rf.hierarchical      # restaurant franchise
        loglik *= tau(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)
    end
        
    # compute likelihood ratio
    return loglik

end # likelihood_base

"""
    sample_allocation(obs::Int64, rf::Restaurants; init::Bool = false)

"""
function sample_allocation(obs::Int64, rf::Restaurants; init::Bool = false)

    # remove observation
    if init == false
        remove_observation(obs, rf)
    end

    # compute masses
    (mass_tables, mass_dishes) = compute_mass(obs, rf)
    
    # sample allocation
    new_allocation(obs, mass_tables, mass_dishes, rf)

end # sample_allocation

"""
    remove_observation(obs::Int64, rf::Restaurants)

"""
function remove_observation(obs::Int64, rf::Restaurants)

    # retrieve customer data
    dish, table = rf.X[obs], rf.Z[obs]

    # update counts matrices
    rf.n[dish] -= 1         # customers per dish
    rf.q[table] -= 1        # customers per table
    if !rf.hierarchical rf.r[table] -= 1 end # tables per dish

    # no customers at table
    if rf.q[table] == 0
        if rf.hierarchical rf.r[dish] -= 1 end  # tables per dish
        rf.table_dish[table] = 0    # no dish index per table
        rf.table_rest[table] = 0    # no rest index per table
    end

    # update latent variables
    rf.X[obs] = 0      # customer dish
    rf.Z[obs] = 0      # customer table

end # remove_observation

"""
    compute_mass(obs::Int64, rf::Restaurants)

"""
function compute_mass(obs::Int64, rf::Restaurants)

    # retrieve customer data
    time, rest = rf.T[obs], rf.Delta[obs]
    cp = (isnothing(rf.CoxProd) ? nothing : rf.CoxProd[obs])

    # initialize vectors
    mass_tables = zeros(length(rf.q)) 
    mass_dishes = zeros(length(rf.r))

    ### sit at old table
    
    # loop on tables
    for (table, qtable) in enumerate(rf.q)
        if rf.table_rest[table] == rest         # table in restaurant
            mass_tables[table] = mass_table(time, cp, table, qtable, rf)
        end
    end

    if !rf.hierarchical     # restaurant array
        return (mass_tables, mass_dishes)
    end

    ### sit at new table, eating old dish

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)
        mass_dishes[dish] = mass_dish(time, cp, dish, rdish, rf)
    end

    # return masses
    return (mass_tables, mass_dishes)

end # compute_mass

"""
    mass_table(time::Float64, cp::Union{Float64,Nothing}, table::Int64, qtable::Int64, rf::Restaurants)

"""
function mass_table(time::Float64, cp::Union{Float64,Nothing}, table::Int64, qtable::Int64, rf::Restaurants)

    # initialize mass
    mass = 0.0

    if qtable != 0      # customers at table

        # retrieve dish and dish value
        dish = rf.table_dish[table]
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = kernel(dish_value, time, cp, rf.kernelpars) * tau_ratio(qtable, rf.KInt[dish], rf.beta, rf.sigma)

    end
    
    return mass

end # mass_table

"""
    mass_dish(time::Float64, cp::Union{Float64,Nothing}, dish::Int64, rdish::Int64, rf::Restaurants)

"""
function mass_dish(time::Float64, cp::Union{Float64,Nothing}, dish::Int64, rdish::Int64, rf::Restaurants)

    # initialize mass
    mass = 0.0

    if rdish != 0       # tables at dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # compute mass
        mass = kernel(dish_value, time, cp, rf.kernelpars) * tau(rf.KInt[dish], rf.beta, rf.sigma) * tau_ratio(rdish, rf.D * psi(rf.KInt[dish], rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    end

    return mass
    
end # mass_dish

"""
    new_allocation(obs::Int64, mass_tables::Vector{Float64}, mass_dishes::Vector{Float64}, rf::Restaurants)

"""
function new_allocation(obs::Int64, mass_tables::Vector{Float64}, mass_dishes::Vector{Float64}, rf::Restaurants)

    # retrieve customer data
    time, rest = rf.T[obs], rf.Delta[obs]
    cp = (isnothing(rf.CoxProd) ? nothing : rf.CoxProd[obs])

    # initialize latent variables
    dish, table = 0, 0

    # retrieve base mass
    mass_base = rf.theta * rf.mass_base[obs]

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
        if !rf.hierarchical rf.r[table] += 1 end    # tables per dish

    elseif case == 2    # sit at new table, eating old dish

        # sample dish
        # d = Categorical(mass_dishes / sum(mass_dishes))     # categorical distribution
        # dish = rand(d)                                      # sample dish
        dish = sample_categorical(mass_dishes)      # sample dish

        # update counts matrices
        rf.n[dish] += 1     # customers per dish
        rf.r[dish] += 1     # tables per dish

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
            dish_value = rand() * time

            # acceptance probability
            accept_prob = likelihood_base(dish_value, time, cp, rf) / likelihood_base(time, time, cp, rf)

            if rand() < accept_prob     # accept proposal
                flag = true
            end

        end

        # find new dish and table index
        dish = find_dish(rf)
        table = (rf.hierarchical ? find_table(rf) : dish)

        # update latent variables
        rf.Xstar[dish] = dish_value

        # update precomputed variables
        rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.kernelpars)

        # update counts vectors
        rf.n[dish] += 1         # customers per dish
        rf.r[dish] += 1         # tables per dish
        rf.q[table] += 1        # customers per table

        # update lookup vectors
        rf.table_dish[table] = dish     # dish index per table
        rf.table_rest[table] = rest     # rest index per table

    end # if

    # update latent variables
    rf.X[obs] = dish        # customer dish
    rf.Z[obs] = table       # customer table

end # new_allocation

"""
    find_table(rest::Int64, rf::Restaurants)

"""
function find_table(rf::Restaurants)

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
    find_dish(rf::Restaurants)

"""
function find_dish(rf::Restaurants)

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

        # double vectors lengths
        if !rf.hierarchical     # restaurant array
            append!(rf.q, zeros(Int64, k))
            append!(rf.table_dish, zeros(Int64, k))
            append!(rf.table_rest, zeros(Int64, k))
        end

        # new dish index
        dish = k + 1
        
    end

    return dish

end # find_dish

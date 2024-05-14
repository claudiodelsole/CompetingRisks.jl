# import from Distributions
import Distributions: Exponential, Categorical

# export struct
export HierarchicalCRM

"""
    struct HierarchicalCRM

"""
struct HierarchicalCRM

    # dimensions
    D::Int64        # number of diseases
    H::Int64        # number of jumps

    # locations
    locations::Vector{Float64} 

    # jumps heights
    jumps_base::Vector{Float64}     # jumps in base measure
    jumps::Matrix{Float64}          # jumps in dependent measures

end # struct

"""
    struct CRMArray

"""
struct CRMArray

    # dimensions
    D::Int64        # number of diseases
    H::Int64        # number of jumps

    # locations
    locations::Vector{Float64} 

    # jumps heights
    jumps::Vector{Float64} 

    # lookup vector
    jump_disease::Vector{Int64}    # diseases indices (k), disease index per jump

end # struct

"""
    sample_measures(rf::RestaurantFranchise)

"""
function sample_measures(rf::RestaurantFranchise)

    # sample base measure
    locations, jumps_base = sample_base_measure(rf)

    # sample dependent measures
    jumps = sample_dependent_measures(rf, locations, jumps_base)

    # create HierarchicalCRM
    return HierarchicalCRM(rf.D, length(locations), locations, jumps_base, jumps)

end # sample_measures

"""
    sample_measures(rf::RestaurantArray)

"""
function sample_measures(rf::RestaurantArray)

    # sample measures
    locations, jumps, jump_disease = sample_independent_measures(rf)

    # create HierarchicalCRM
    return CRMArray(rf.D, length(locations), locations, jumps, jump_disease)

end # sample_measures

"""
    sample_base_measure(rf::RestaurantFranchise)

"""
function sample_base_measure(rf::RestaurantFranchise)

    # initialize vectors
    locations = copy(rf.Xstar)
    jumps_base = zeros(length(rf.r))

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        if rdish == 0       # no tables at dish
            continue
        end

        # sample jump height
        jumps_base[dish] = sample_jump(rdish, rf.beta0, rf.sigma0; posterior = rf.D * psi(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma))

    end

    # truncation parameter
    eps = 1.0e-3

    # initialize total mass
    sumjumps = sum(jumps_base)

    # initialize standard Poisson process
    spp = 0.0

    while spp <= rf.theta * jumps_measure(log(eps * sumjumps), rf.beta0, rf.sigma0)

        # update Poisson process
        spp += rand(Exponential())

        # sample location
        atom = rand(rf.base_measure)

        # precompute KernelInt
        if isnothing(rf.CoxProd)    # no regression model
            KInt = rf.alpha * KernelInt(atom, rf.T, rf.eta)
        else    # regression model
            KInt = rf.alpha * KernelInt(atom, rf.T, rf.CoxProd, rf.eta)
        end

        # algorithm starting point
        start = -8.0

        # define functions
        f = logh::Float64 -> jumps_measure(logh, rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        fp = logh::Float64 -> jumps_measure_grad(logh, rf.beta0, rf.sigma0; posterior = rf.D * psi(KInt, rf.beta, rf.sigma))

        # solve equation
        logjump = newton(spp / rf.theta, f, fp, start)
        jump = exp(logjump)

        # append atom and jump
        append!(locations, atom)
        append!(jumps_base, jump)

        # update total mass
        sumjumps += jump

    end

    return (locations, jumps_base)

end # sample_base_measure

"""
    sample_dependent_measures(rf::RestaurantFranchise, locations::Vector{Float64}, jumps_base::Vector{Float64})

"""
function sample_dependent_measures(rf::RestaurantFranchise, locations::Vector{Float64}, jumps_base::Vector{Float64})

    # initialize matrix
    jumps = zeros(length(locations), rf.D)

    # loop on tables
    for (table, qtable) in enumerate(rf.q)

        if qtable == 0  # no customers at table
            continue
        end

        # retrieve dish and restaurant
        dish = rf.table_dish[table]
        rest = rf.table_rest[table]

        # sample jump height
        jumps[dish, rest] += sample_jump(qtable, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[dish])

    end

    # base random measure
    mass_base = sum(jumps_base)
    baseCRM = Categorical(jumps_base / mass_base)

    # loop on dependent measures
    for d in 1:rf.D

        # truncation parameter
        eps = 1.0e-3

        # initialize total mass
        sumjumps = sum(jumps[:,d])

        # initialize standard Poisson process
        spp = 0.0

        while spp <= mass_base * jumps_measure(log(eps * sumjumps), rf.beta, rf.sigma)

            # update Poisson process
            spp += rand(Exponential())

            # sample location
            atom = rand(baseCRM)

            # precompute KernelInt
            if isnothing(rf.CoxProd)    # no regression model
                KInt = rf.alpha * KernelInt(locations[atom], rf.T, rf.eta)
            else    # regression model
                KInt = rf.alpha * KernelInt(locations[atom], rf.T, rf.CoxProd, rf.eta)
            end

            # algorithm starting point
            start = -8.0

            # define functions
            f = logh::Float64 -> jumps_measure(logh, rf.beta, rf.sigma; posterior = KInt)
            fp = logh::Float64 -> jumps_measure_grad(logh, rf.beta, rf.sigma; posterior = KInt)

            # solve equation
            logjump = newton(spp / mass_base, f, fp, start)
            jump = exp(logjump)

            # update jump at atom
            jumps[atom, d] += jump

            # update total mass
            sumjumps += jump

        end

    end

    return jumps

end # sample_dependent_measures

"""
    sample_independent_measures(rf::RestaurantArray)

"""
function sample_independent_measures(rf::RestaurantArray)

    # initialize vectors
    locations = copy(rf.Xstar)
    jumps = zeros(length(rf.n))
    jump_disease = copy(rf.table_rest)

    # loop on dishes
    for (table, ntable) in enumerate(rf.n)

        if ntable == 0  # no customers at table
            continue
        end

        # sample jump height
        jumps[table] = sample_jump(ntable, rf.beta, rf.sigma; posterior = rf.alpha * rf.KInt[table])

    end

    # loop on independent measures
    for d in 1:rf.D

        # truncation parameter
        eps = 1.0e-3

        # initialize total mass
        sumjumps = sum(jumps)

        # initialize standard Poisson process
        spp = 0.0

        while spp <= rf.theta * jumps_measure(log(eps * sumjumps), rf.beta, rf.sigma)

            # update Poisson process
            spp += rand(Exponential())

            # sample location
            atom = rand(rf.base_measure)

            # precompute KernelInt
            if isnothing(rf.CoxProd)    # no regression model
                KInt = rf.alpha * KernelInt(atom, rf.T, rf.eta)
            else    # regression model
                KInt = rf.alpha * KernelInt(atom, rf.T, rf.CoxProd, rf.eta)
            end

            # algorithm starting point
            start = -8.0

            # define functions
            f = logh::Float64 -> jumps_measure(logh, rf.beta, rf.sigma; posterior = KInt)
            fp = logh::Float64 -> jumps_measure_grad(logh, rf.beta, rf.sigma; posterior = KInt)

            # solve equation
            logjump = newton(spp / rf.theta, f, fp, start)
            jump = exp(logjump)

            # append atom and jump
            append!(locations, atom)
            append!(jumps, jump)
            append!(jump_disease, d)

            # update total mass
            sumjumps += jump

        end

    end

    return (locations, jumps, jump_disease)

end # sample_independent_measures

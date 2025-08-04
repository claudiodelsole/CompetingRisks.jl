"""
    struct CRM

"""
struct CRM

    # locations
    locations::Vector{Float64} 

    # jumps heights
    jumps::Vector{Float64}

end # CRM

"""
    sample_measures(rf::Restaurants, Tmax::Float64)

"""
function sample_measures(rf::Restaurants, Tmax::Float64)

    if rf.hierarchical      # restaurant franchise

        # sample base measure
        base_measure = sample_base_measure(rf, Tmax)

        # sample dependent measures
        return sample_dependent_measures(rf, base_measure)

    else    # restaurant array

        return sample_independent_measures(rf, Tmax)

    end

end # sample_measures

"""
    sample_base_measure(rf::Restaurants, Tmax::Float64; eps::Float64 = 1.0e-4, maxIter::Int64 = 1000)

"""
function sample_base_measure(rf::Restaurants, Tmax::Float64; eps::Float64 = 1.0e-4, maxIter::Int64 = 1000)

    # initialize vectors
    locations = copy(rf.Xstar)
    jumps = zeros(length(rf.r))

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        # no tables at dish
        if rdish == 0 continue end

        # sample jump height
        jumps[dish] = sample_jump(rdish, rf.beta0, rf.sigma0, posterior = rf.D * psi(rf.alpha * rf.KInt[dish], rf.beta, rf.sigma))

    end

    # initialize total mass
    sumjumps, numjumps = 0.0, 0

    # initialize standard Poisson process
    logjump, spp = -1.0, 0.0

    while spp <= tail_integral(log(eps * sumjumps), rf.beta0, rf.sigma0) && numjumps < maxIter

        # update Poisson process
        spp += rand(Exponential()) / (rf.theta * Tmax)

        # sample location
        atom = rand() * Tmax

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(atom, rf.T, rf.CoxProd, rf.kappa)

        # define functions
        f(logh::Float64) = tail_integral(logh, rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))
        fp(logh::Float64) = tail_integral_grad(logh, rf.beta0, rf.sigma0, posterior = rf.D * psi(KInt, rf.beta, rf.sigma))

        # algorithm starting point
        if logjump >= 0.0 logjump = -1.0 end
        while f(logjump) <= spp
            logjump *= 2.0
        end

        # solve equation
        logjump = newton(spp, f, fp, logjump)
        jump = exp(logjump)

        # append atom and jump
        append!(locations, atom)
        append!(jumps, jump)

        # update total mass
        sumjumps += jump
        numjumps += 1

    end

    return CRM(locations, jumps)

end # sample_base_measure

"""
    sample_dependent_measures(rf::Restaurants, base_measure::CRM; eps::Float64 = 1.0e-4, maxIter::Int64 = 1000)

"""
function sample_dependent_measures(rf::Restaurants, base_measure::CRM; eps::Float64 = 1.0e-4, maxIter::Int64 = 1000)

    # initialize matrix
    jumps = zeros(length(base_measure.locations), rf.D)

    # loop on tables
    for (table, qtable) in enumerate(rf.q)

        # no customers at table
        if qtable == 0 continue end

        # retrieve dish and restaurant
        dish = rf.table_dish[table]
        rest = rf.table_rest[table]

        # sample jump height
        jumps[dish, rest] += sample_jump(qtable, rf.beta, rf.sigma, posterior = rf.alpha * rf.KInt[dish])

    end

    # base random measure
    mass_base = sum(base_measure.jumps)
    baseCRM = Categorical(base_measure.jumps / mass_base)

    # loop on dependent measures
    for rest in axes(jumps, 2)

        # initialize total mass
        sumjumps, numjumps = 0.0, 0

        # initialize standard Poisson process
        logjump, spp = -1.0, 0.0

        while spp <= tail_integral(log(eps * sumjumps), rf.beta, rf.sigma) && numjumps < maxIter

            # update Poisson process
            spp += rand(Exponential()) / mass_base

            # sample location
            atom = rand(baseCRM)

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(base_measure.locations[atom], rf.T, rf.CoxProd, rf.kappa)

            # define functions
            f(logh::Float64) = tail_integral(logh, rf.beta, rf.sigma, posterior = KInt)
            fp(logh::Float64) = tail_integral_grad(logh, rf.beta, rf.sigma, posterior = KInt)

            # algorithm starting point
            if logjump >= 0.0 logjump = -1.0 end
            while f(logjump) <= spp
                logjump *= 2.0
            end

            # solve equation
            logjump = newton(spp, f, fp, logjump)
            jump = exp(logjump)

            # update jump at atom
            jumps[atom, rest] += jump

            # update total mass
            sumjumps += jump
            numjumps += 1

        end

    end

    return [CRM(base_measure.locations, jumpsvec) for jumpsvec in eachcol(jumps)]

end # sample_dependent_measures

"""
    sample_independent_measures(rf::Restaurants, Tmax::Float64; eps::Float64 = 1.0e-4, maxIter::Int64 = 1000)

"""
function sample_independent_measures(rf::Restaurants, Tmax::Float64; eps::Float64 = 1.0e-4, maxIter::Int64 = 1000)

    # initialize vector
    crms = Vector{CRM}(undef, rf.D)

    # loop on independent measures
    for d in eachindex(crms)

        # initialize vectors
        locations = copy(rf.Xstar)
        jumps = zeros(length(rf.n))

        # loop on dishes
        for (table, ntable) in enumerate(rf.n)

            # no customers at table
            if ntable == 0 || rf.table_rest[table] != d continue end

            # sample jump height
            jumps[table] = sample_jump(ntable, rf.beta, rf.sigma, posterior = rf.alpha * rf.KInt[table])

        end

        # initialize total mass
        sumjumps, numjumps = 0.0, 0

        # initialize standard Poisson process
        logjump, spp = -1.0, 0.0

        while spp <= tail_integral(log(eps * sumjumps), rf.beta, rf.sigma) && numjumps < maxIter

            # update Poisson process
            spp += rand(Exponential()) / (rf.theta * Tmax)

            # sample location
            atom = rand() * Tmax

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(atom, rf.T, rf.CoxProd, rf.kappa)

            # define functions
            f(logh::Float64) = tail_integral(logh, rf.beta, rf.sigma, posterior = KInt)
            fp(logh::Float64) = tail_integral_grad(logh, rf.beta, rf.sigma, posterior = KInt)

            # algorithm starting point
            if logjump >= 0.0 logjump = -1.0 end
            while f(logjump) <= spp
                logjump *= 2.0
            end

            # solve equation
            logjump = newton(spp, f, fp, logjump)
            jump = exp(logjump)

            # append atom and jump
            append!(locations, atom)
            append!(jumps, jump)

            # update total mass
            sumjumps += jump
            numjumps += 1

        end

        # create CRM
        crms[d] = CRM(locations, jumps)

    end

    return crms

end # sample_independent_measures

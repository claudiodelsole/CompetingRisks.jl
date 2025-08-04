"""
    resample_dishes(rf::Restaurants)

"""
function resample_dishes(rf::Restaurants)

    # acceptance counter
    accept = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        # no tables at dish
        if rdish == 0 continue end

        # dish value
        dish_value = rf.Xstar[dish]

        # random walk step
        eps = exp( stdev_dishes[] * randn() )

        # acceptance probability
        logprob = loglikelihood_dishes(dish, rdish, dish_value * eps, rf) + log(eps)
        logprob -= loglikelihood_dishes(dish, rdish, dish_value, rf)
        accept += min(exp(logprob), 1.0)
        
        if log(rand()) < logprob    # accept proposal

            # update dish value
            rf.Xstar[dish] = dish_value * eps

            # update KernelInt for dish value
            rf.KInt[dish] = KernelInt(dish_value * eps, rf.T, rf.CoxProd, rf.kappa)

        end

    end

    # return acceptance probabilities
    return accept

end # resample_dishes

"""
    loglikelihood_dishes(dish::Int64, rdish::Int64, dish_value::Float64, rf::Restaurants)

"""
function loglikelihood_dishes(dish::Int64, rdish::Int64, dish_value::Float64, rf::Restaurants)

    # precompute KernelInt
    KInt = rf.alpha * KernelInt(dish_value, rf.T, rf.CoxProd, rf.kappa)

    # retrieve indices
    customers = (rf.X .== dish)         # customers eating dish

    # compute loglikelihood
    loglik = sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], isnothing(rf.CoxProd) ? nothing : rf.CoxProd[customers], rf.kappa)))

    if rf.hierarchical     # restaurant franchise
        
        # retrieve indices
        tables = (rf.table_dish .== dish)   # tables eating dish

        # compute loglikelihood
        loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma)) + logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    else    # restaurant array

        # compute loglikelihood 
        loglik += logtau(rdish, KInt, rf.beta, rf.sigma)

    end

    return loglik

end # loglikelihood_dishes

"""
    resample_theta(rf::Restaurants)

"""
function resample_theta(rf::Restaurants)
    
    # compute rate
    if rf.hierarchical     # restaurant franchise
        rate = integrate(x -> psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant array
        rate = integrate(x -> rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    # resample theta
    rf.theta = rand(Gamma(1.0 + count(rf.r .> 0))) / (0.1 + rate)

end # resample theta

"""
    resample_alpha(rf::Restaurants)

"""
function resample_alpha(rf::Restaurants)

    # prior for alpha
    prior = Gamma(1.0, 10.0)

    # random walk step
    eps = exp( stdev_alpha[] * randn() )

    # acceptance probability
    logprob = loglikelihood_alpha(rf.alpha * eps, rf) + log(pdf(prior, rf.alpha * eps)) + log(eps)
    logprob -= loglikelihood_alpha(rf.alpha, rf) + log(pdf(prior, rf.alpha))
    accept = min(exp(logprob), 1.0)

    if (flag = log(rand()) < logprob)   # accept proposal

        # update alpha
        rf.alpha *= eps

    end

    # return acceptance probability
    return (accept, flag)

end # resample_alpha

"""
    loglikelihood_alpha(alpha::Float64, rf::Restaurants)

"""
function loglikelihood_alpha(alpha::Float64, rf::Restaurants)

    # initialize loglikelihood
    loglik = count(rf.Delta .!= 0) * log(alpha)

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        # no tables at dish
        if rdish == 0 continue end

        # precompute KernelInt
        KInt = alpha * rf.KInt[dish]

        if rf.hierarchical     # restaurant franchise

            # retrieve indices
            tables = (rf.table_dish .== dish)   # tables eating dish

            # compute loglikelihood
            loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma)) + logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

        else    # restaurant array

            # compute loglikelihood
            loglik += logtau(rdish, KInt, rf.beta, rf.sigma)

        end

    end

    # compute loglikelihood
    if rf.hierarchical     # restaurant franchise
        loglik -= rf.theta * integrate(x -> psi(rf.D * psi(alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant array
        loglik -= rf.theta * integrate(x -> rf.D * psi(alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    return loglik

end # loglikelihood_alpha

"""
    resample_kappa(rf::Restaurants)

"""
function resample_kappa(rf::Restaurants)

    # prior for eta
    prior = Gamma(1.0, 10.0)

    # random walk step
    eps = exp( stdev_kappa[] * randn() )

    # acceptance probability
    logprob = loglikelihood_kappa(rf.kappa * eps, rf) + log(pdf(prior, rf.kappa * eps)) + log(eps)
    logprob -= loglikelihood_kappa(rf.kappa, rf) + log(pdf(prior, rf.kappa))
    accept = min(exp(logprob), 1.0)

    if (flag = log(rand()) < logprob)   # accept proposal

        # update eta
        rf.kappa *= eps

        # update KernelInt for dish values
        for (dish, dish_value) in enumerate(rf.Xstar)
            rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.kappa)
        end

    end

    # return acceptance probability
    return (accept, flag)

end # resample_kappa

"""
    loglikelihood_kappa(kappa::Float64, rf::Restaurants)

"""
function loglikelihood_kappa(kappa::Float64, rf::Restaurants)

    # initialize loglikelihood
    loglik = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        # no tables at dish
        if rdish == 0 continue end

        # retrieve indices
        customers = (rf.X .== dish)         # customers eating dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(dish_value, rf.T, rf.CoxProd, kappa)

        # compute loglikelihood
        loglik += sum(log.(kernel.(dish_value, rf.T[customers], isnothing(rf.CoxProd) ? nothing : rf.CoxProd[customers], kappa)))

        if rf.hierarchical     # restaurant franchise

            # retrieve indices
            tables = (rf.table_dish .== dish)   # tables eating dish

            # compute loglikelihood
            loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma)) + logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

        else    # restaurant array

            # compute loglikelihood
            loglik += logtau(rdish, KInt, rf.beta, rf.sigma)

        end

    end

    # compute loglikelihood
    if rf.hierarchical     # restaurant franchise
        loglik -= rf.theta * integrate(x -> psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, kappa), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant array
        loglik -= rf.theta * integrate(x -> rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, kappa), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    return loglik

end # loglikelihood_kappa

"""
    resample_coefficients(rf::Restaurants, cm::CoxModel)

"""
function resample_coefficients(rf::Restaurants, cm::CoxModel)

    # acceptance counter
    accept = zeros(Float64, cm.L)

    # resampling flag
    flag = false

    # prior for coefficients
    prior = Normal(0.0, 10.0)

    # initialize proposed exponential CoxProducts
    CoxProd = ones(Float64, rf.N)

    # loop on coefficients
    for (l, coeff) in enumerate(cm.eta)

        # retrieve indices
        customers = (cm.predictors .== l)

        # random walk step
        eps = stdev_coefficients[] * randn()

        # proposed exponential CoxProducts
        CoxProd[:] = rf.CoxProd[:]
        CoxProd[customers] .= exp(coeff + eps)

        # acceptance probability
        logprob = loglikelihood_coeffs(CoxProd, rf) + log(pdf(prior, coeff + eps))
        logprob -= loglikelihood_coeffs(rf.CoxProd, rf) + log(pdf(prior, coeff))
        accept[l] = min(exp(logprob), 1.0)
        
        if log(rand()) < logprob    # accept proposal

            # update coefficient value
            cm.eta[l] += eps

            # update exponential CoxProduct for observations
            rf.CoxProd[customers] .= exp(coeff + eps)

            # set flag
            flag = true

        end

    end

    # update KernelInt for dish values
    for (dish, dish_value) in enumerate(rf.Xstar)
        rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.kappa)
    end

    # return acceptance probabilities
    return (accept, flag)

end

"""
    loglikelihood_coeffs(CoxProd::Vector{Float64}, rf::Restaurants)

"""
function loglikelihood_coeffs(CoxProd::Vector{Float64}, rf::Restaurants)

    # initialize loglikelihood
    loglik = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        # no tables at dish
        if rdish == 0 continue end

        # retrieve indices
        customers = (rf.X .== dish)         # customers eating dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(dish_value, rf.T, CoxProd, rf.kappa)

        # compute loglikelihood
        loglik += sum(log.(kernel.(dish_value, rf.T[customers], CoxProd[customers], rf.kappa)))

        if rf.hierarchical     # restaurant franchise

            # retrieve indices
            tables = (rf.table_dish .== dish)   # tables eating dish

            # compute loglikelihood
            loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma)) + logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

        else    # restaurant array

            # compute loglikelihood
            loglik += logtau(rdish, KInt, rf.beta, rf.sigma)

        end

    end

    # compute loglikelihood
    if rf.hierarchical     # restaurant franchise
        loglik -= rf.theta * integrate(x -> psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, CoxProd, rf.kappa), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant array
        loglik -= rf.theta * integrate(x -> rf.D * psi(rf.alpha * KernelInt(x, rf.T, CoxProd, rf.kappa), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    return loglik

end # loglikelihood_coeffs

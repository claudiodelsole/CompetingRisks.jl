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
            rf.KInt[dish] = KernelInt(dish_value * eps, rf.T, rf.CoxProd, rf.kernelpars)

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
    KInt = KernelInt(dish_value, rf.T, rf.CoxProd, rf.kernelpars)

    # retrieve indices
    customers = (rf.X .== dish)         # customers eating dish

    # compute loglikelihood
    loglik = sum(log.(kernel.(dish_value, rf.T[customers], isnothing(rf.CoxProd) ? nothing : rf.CoxProd[customers], [rf.kernelpars])))

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
        rate = integrate(x -> psi(rf.D * psi(KernelInt(x, rf.T, rf.CoxProd, rf.kernelpars), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant array
        rate = integrate(x -> rf.D * psi(KernelInt(x, rf.T, rf.CoxProd, rf.kernelpars), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    # resample theta
    rf.theta = rand(Gamma(1.0 + count(rf.r .> 0))) / (0.1 + rate)

end # resample theta

"""
    resample_kernelpars(rf::Restaurants)

"""
function resample_kernelpars(rf::Restaurants)

    # prior for eta
    prior = Gamma(1.0, 10.0)

    # random walk step
    eps = exp.( stdev_kernelpars[] * randn(nfields(rf.kernelpars)) )
    kernelpars = typeof(rf.kernelpars)((vectorize(rf.kernelpars) .* eps)...)

    # acceptance probability
    logprob = loglikelihood_kernel(kernelpars, rf.CoxProd, rf) + sum(log.(pdf.(prior, vectorize(kernelpars)))) + sum(log.(eps))
    logprob -= loglikelihood_kernel(rf.kernelpars, rf.CoxProd, rf) + sum(log.(pdf.(prior, vectorize(rf.kernelpars))))
    accept = min(exp(logprob), 1.0)

    if (flag = log(rand()) < logprob)   # accept proposal

        # update kernel
        rf.kernelpars = kernelpars

        # update KernelInt for dish values
        for (dish, dish_value) in enumerate(rf.Xstar)
            rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.kernelpars)
        end

    end

    # return acceptance probability
    return (accept, flag)

end # resample_kernelparams

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
        logprob = loglikelihood_kernel(rf.kernelpars, CoxProd, rf) + log(pdf(prior, coeff + eps))
        logprob -= loglikelihood_kernel(rf.kernelpars, rf.CoxProd, rf) + log(pdf(prior, coeff))
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
        rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.kernelpars)
    end

    # return acceptance probabilities
    return (accept, flag)

end

"""
    loglikelihood_kernel(kernelpars::AbstractKernel, CoxProd::Union{Vector{Float64},Nothing}, rf::Restaurants)

"""
function loglikelihood_kernel(kernelpars::AbstractKernel, CoxProd::Union{Vector{Float64},Nothing}, rf::Restaurants)

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
        KInt = KernelInt(dish_value, rf.T, CoxProd, kernelpars)

        # compute loglikelihood
        loglik += sum(log.(kernel.(dish_value, rf.T[customers], isnothing(CoxProd) ? nothing : CoxProd[customers], [kernelpars])))

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
        loglik -= rf.theta * integrate(x -> psi(rf.D * psi(KernelInt(x, rf.T, CoxProd, kernelpars), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant array
        loglik -= rf.theta * integrate(x -> rf.D * psi(KernelInt(x, rf.T, CoxProd, kernelpars), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    return loglik

end # loglikelihood_kernel

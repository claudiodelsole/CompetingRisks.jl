"""
    resample_dishes(rf::RestaurantFranchise)

"""
function resample_dishes(rf::RestaurantFranchise)

    # initialize acceptance counter
    accept_dishes = 0.0

    # relative standard deviation
    mhdev = 0.5

    for (dish, rdish) in enumerate(rf.r)    # loop on dishes

        if rdish == 0       # no tables at dish
            continue
        end

        # dish value
        dish_value = rf.Xstar[dish]

        # random walk step
        eps = exp( mhdev * randn() )

        # acceptance probability
        accept_logprob = loglikelihood_dishes(dish, rdish, dish_value * eps, rf) + log(pdf(rf.base_measure, dish_value * eps)) + log(eps)
        accept_logprob -= loglikelihood_dishes(dish, rdish, dish_value, rf) + log(pdf(rf.base_measure, dish_value))
        accept_dishes += min(exp(accept_logprob), 1.0)
        
        if log(rand()) < accept_logprob     # accept proposal

            # update dish value
            rf.Xstar[dish] = dish_value * eps

            # update KernelInt for dish value
            rf.KInt[dish] = KernelInt(dish_value * eps, rf.T, rf.eta)

        end

    end

    # return acceptance probabilities
    return accept_dishes

end # resample_dishes

"""
    resample_dishes(rf::RestaurantArray)

"""
function resample_dishes(rf::RestaurantArray)

    # initialize acceptance counter
    accept_dishes = 0.0

    # relative standard deviation
    mhdev = 0.5

    for (table, ntable) in enumerate(rf.n)    # loop on tables

        if ntable == 0      # no customers at table
            continue
        end

        # dish value
        dish_value = rf.Xstar[table]

        # random walk step
        eps = exp( mhdev * randn() )

        # acceptance probability
        accept_logprob = loglikelihood_dishes(table, ntable, dish_value * eps, rf) + log(pdf(rf.base_measure, dish_value * eps)) + log(eps)
        accept_logprob -= loglikelihood_dishes(table, ntable, dish_value, rf) + log(pdf(rf.base_measure, dish_value))
        accept_dishes += min(exp(accept_logprob), 1.0)
        
        if log(rand()) < accept_logprob     # accept proposal

            # update dish value
            rf.Xstar[table] = dish_value * eps

            # update KernelInt for dish value
            rf.KInt[table] = KernelInt(dish_value * eps, rf.T, rf.eta)

        end

    end

    # return acceptance probabilities
    return accept_dishes

end # resample_dishes

"""
    loglikelihood_dishes(dish::Int64, rdish::Int64, dish_value::Float64, rf::RestaurantFranchise)

"""
function loglikelihood_dishes(dish::Int64, rdish::Int64, dish_value::Float64, rf::RestaurantFranchise)

    # retrieve indices
    customers = (rf.X .== dish)         # customers eating dish
    tables = (rf.table_dish .== dish)   # tables eating dish

    if isnothing(rf.CoxProd)    # exchangeable model

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(dish_value, rf.T, rf.eta)

        # compute loglikelihood
        loglik = sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.eta)))
        loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
        loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    else    # regression model

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(dish_value, rf.T, rf.CoxProd, rf.eta)

        # compute loglikelihood
        loglik = sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.CoxProd[customers], rf.eta)))
        loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
        loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    end

    return loglik

end # loglikelihood_dishes

"""
    loglikelihood_dishes(dish::Int64, ndish::Int64, dish_value::Float64, rf::RestaurantArray)

"""
function loglikelihood_dishes(table::Int64, ntable::Int64, dish_value::Float64, rf::RestaurantArray)

    # retrieve indices
    customers = (rf.X .== table)         # customers eating dish

    if isnothing(rf.CoxProd)    # exchangeable model

        # compute loglikelihood
        loglik = sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.eta))) 
        loglik += logtau(ntable, rf.alpha * KernelInt(dish_value, rf.T, rf.eta), rf.beta, rf.sigma)
    
    else    # regression model

        # compute loglikelihood
        loglik = sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.CoxProd[customers], rf.eta))) 
        loglik += logtau(ntable, rf.alpha * KernelInt(dish_value, rf.T, rf.eta), rf.beta, rf.sigma)

    end

    return loglik

end # loglikelihood_dishes

"""
    resample_theta(rf::RestaurantFranchise)

"""
function resample_theta(rf::RestaurantFranchise)

    # number of dishes (with tables)
    k = sum(rf.r .> 0)
    
    if isnothing(rf.CoxProd)    # exchangeable model

        # integrand function
        f(x::Float64) = psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        # compute integral
        I = integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        g(x::Float64) = psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        # compute integral
        I = integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    # resample theta
    rf.theta = rand(Gamma(1.0 + k, 1.0 / (0.1 + I)))

end # resample theta

"""
    resample_theta(rf::RestaurantArray)

"""
function resample_theta(rf::RestaurantArray)

    # number of tables (with customers)
    k = sum(rf.n .> 0)

    if isnothing(rf.CoxProd)    # exchangeable model

        # integrand function
        f(x::Float64) = psi(rf.alpha * KernelInt(x, rf.T, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        # compute integral
        I = integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        g(x::Float64) = psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        # compute integral
        I = integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    # resample theta
    rf.theta = rand(Gamma(1.0 + k, 1.0 / (0.1 + rf.D * I)))

end # resample theta

"""
    resample_alpha(rf::Union{RestaurantFranchise,RestaurantArray})

"""
function resample_alpha(rf::Union{RestaurantFranchise,RestaurantArray})

    # resampling flag
    flag = false

    # relative standard deviation
    mhdev = 2.0

    # prior for alpha
    prior = Gamma(1.0, 10.0)

    # random walk step
    eps = exp( mhdev * randn() )

    # acceptance probability
    accept_logprob = loglikelihood_alpha(rf.alpha * eps, rf) + log(pdf(prior, rf.alpha * eps)) + log(eps)
    accept_logprob -= loglikelihood_alpha(rf.alpha, rf) + log(pdf(prior, rf.alpha))
    accept_alpha = min(exp(accept_logprob), 1.0)

    if log(rand()) < accept_logprob     # accept proposal

        # update alpha
        rf.alpha *= eps

        # set flag
        flag = true

    end

    # return acceptance probability
    return (accept_alpha, flag)

end # resample_alpha

"""
    loglikelihood_alpha(alpha::Float64, rf::RestaurantFranchise)

"""
function loglikelihood_alpha(alpha::Float64, rf::RestaurantFranchise)

    # initialize loglikelihood
    loglik = sum(rf.Delta .!= 0) * log(alpha)

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        if rdish == 0       # no tables at dish
            continue
        end

        # retrieve indices
        tables = (rf.table_dish .== dish)   # tables eating dish

        # precompute KernelInt
        KInt = alpha * rf.KInt[dish]

        # compute loglikelihood
        loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
        loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    end

    if isnothing(rf.CoxProd)    # exchangeable model

        # integrand function
        function f(x::Float64) 

            # precompute KernelInt
            KInt = alpha * KernelInt(x, rf.T, rf.eta)
            
            # compute integrand
            return psi(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        function g(x::Float64) 

            # precompute KernelInt
            KInt = alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)
            
            # compute integrand
            return psi(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    return loglik

end # loglikelihood_alpha

"""
    loglikelihood_alpha(alpha::Float64, rf::RestaurantArray)

"""
function loglikelihood_alpha(alpha::Float64, rf::RestaurantArray)

    # initialize loglikelihood
    loglik = sum(rf.Delta .!= 0) * log(alpha)

    # loop on dishes
    for (table, ntable) in enumerate(rf.n)

        if ntable == 0      # no customers at table
            continue
        end

        # precompute KernelInt
        KInt = alpha * rf.KInt[table]

        # compute loglikelihood
        loglik += logtau(ntable, KInt, rf.beta, rf.sigma)

    end

    if isnothing(rf.CoxProd)    # exchangeable model

        # integrand function
        function f(x::Float64) 

            # precompute KernelInt
            KInt = alpha * KernelInt(x, rf.T, rf.eta)
            
            # compute integrand
            return psi(KInt, rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * rf.D * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        function g(x::Float64) 

            # precompute KernelInt
            KInt = alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta)
            
            # compute integrand
            return psi(KInt, rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * rf.D * integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    return loglik

end # loglikelihood_alpha

"""
    resample_eta(rf::Union{RestaurantFranchise,RestaurantArray})

"""
function resample_eta(rf::Union{RestaurantFranchise,RestaurantArray})

    # resampling flag
    flag = false

    # relative standard deviation
    mhdev = 1.0

    # prior for eta
    prior = Gamma(1.0, 10.0)

    # random walk step
    eps = exp( mhdev * randn() )

    # acceptance probability
    accept_logprob = loglikelihood_eta(rf.eta * eps, rf) + log(pdf(prior, rf.eta * eps)) + log(eps)
    accept_logprob -= loglikelihood_eta(rf.eta, rf) + log(pdf(prior, rf.eta))
    accept_eta = min(exp(accept_logprob), 1.0)

    if log(rand()) < accept_logprob     # accept proposal

        # update eta
        rf.eta *= eps

        if isnothing(rf.CoxProd)    # exchangeable model

            # update KernelInt for dish values
            for (dish, dish_value) in enumerate(rf.Xstar)
                rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.eta)
            end

        else    # regression model

            # update KernelInt for dish values
            for (dish, dish_value) in enumerate(rf.Xstar)
                rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.eta)
            end

        end

        # set flag
        flag = true

    end

    # return acceptance probability
    return (accept_eta, flag)

end # resample_eta

"""
    loglikelihood_eta(eta::Float64, rf::RestaurantFranchise)

"""
function loglikelihood_eta(eta::Float64, rf::RestaurantFranchise)

    # initialize loglikelihood
    loglik = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        if rdish == 0       # no tables at dish
            continue
        end

        # retrieve indices
        customers = (rf.X .== dish)         # customers eating dish
        tables = (rf.table_dish .== dish)   # tables eating dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        if isnothing(rf.CoxProd)    # exchangeable model

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(dish_value, rf.T, eta)

            # compute loglikelihood
            loglik += sum(log.(kernel.(dish_value, rf.T[customers], eta)))
            loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
            loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

        else    # regression model

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(dish_value, rf.T, rf.CoxProd, eta)

            # compute loglikelihood
            loglik += sum(log.(kernel.(dish_value, rf.T[customers], rf.CoxProd[customers], eta)))
            loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
            loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

        end

    end

    if isnothing(rf.CoxProd)    # exchangeable model

        # integrand function
        function f(x::Float64) 

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(x, rf.T, eta)
            
            # compute integrand
            return psi(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        function g(x::Float64) 

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, eta)
            
            # compute integrand
            return psi(rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    return loglik

end # loglikelihood_eta

"""
    loglikelihood_eta(eta::Float64, rf::RestaurantArray)

"""
function loglikelihood_eta(eta::Float64, rf::RestaurantArray)

    # initialize loglikelihood
    loglik = 0.0

    # loop on dishes
    for (table, ntable) in enumerate(rf.n)

        if ntable == 0      # no customers at table
            continue
        end

        # retrieve indices
        customers = (rf.X .== table)         # customers seated at table

        # retrieve dish value
        dish_value = rf.Xstar[table]

        if isnothing(rf.CoxProd)    # exchangeable model

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(dish_value, rf.T, eta)

            # compute loglikelihood
            loglik += sum(log.(kernel.(dish_value, rf.T[customers], eta)))
            loglik += logtau(ntable, KInt, rf.beta, rf.sigma)

        else    # regression model

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(dish_value, rf.T, rf.CoxProd, eta)

            # compute loglikelihood
            loglik += sum(log.(kernel.(dish_value, rf.T[customers], rf.CoxProd[customers], eta)))
            loglik += logtau(ntable, KInt, rf.beta, rf.sigma)

        end

    end

    if isnothing(rf.CoxProd)    # exchangeable model

        # integrand function
        function f(x::Float64) 

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(x, rf.T, eta)
            
            # compute integrand
            return psi(KInt, rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * rf.D * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        function g(x::Float64) 

            # precompute KernelInt
            KInt = rf.alpha * KernelInt(x, rf.T, rf.CoxProd, eta)
            
            # compute integrand
            return psi(KInt, rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * rf.D * integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    return loglik

end # loglikelihood_eta

"""
    resample_coefficients(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel)

"""
function resample_coefficients(rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel)

    # initialize acceptance
    accept_coeffs = zeros(Float64, cm.L)

    # resampling flag
    flag = false

    # prior for coefficients
    prior = Normal(0.0, 10.0)

    # standard deviation
    mhdev = 1.0

    # initialize proposed exponential CoxProducts
    CoxProd = ones(Float64, rf.N)

    for (l, coeff) in enumerate(cm.xi)      # loop on coefficients

        # random walk step
        eps = mhdev * randn()

        # compute proposed exponential CoxProducts
        for (cust, predictor) in enumerate(cm.predictors)
            if predictor == l
                CoxProd[cust] = exp(coeff + eps)
            else
                CoxProd[cust] = rf.CoxProd[cust]
            end
        end

        # acceptance probability
        accept_logprob = loglikelihood_coeffs(CoxProd, rf) + log(pdf(prior, coeff + eps))
        accept_logprob -= loglikelihood_coeffs(rf.CoxProd, rf) + log(pdf(prior, coeff))
        accept_coeffs[l] = min(exp(accept_logprob), 1.0)
        
        if log(rand()) < accept_logprob     # accept proposal

            # update coefficient value
            cm.xi[l] += eps

            # update exponential CoxProduct for observations
            for (cust, predictor) in enumerate(cm.predictors)
                if predictor == l
                    rf.CoxProd[cust] = CoxProd[cust]
                end
            end

            # set flag
            flag = true

        end

    end

    # update KernelInt for dish values
    for (dish, dish_value) in enumerate(rf.Xstar)
        rf.KInt[dish] = KernelInt(dish_value, rf.T, rf.CoxProd, rf.eta)
    end

    # return acceptance probabilities
    return (accept_coeffs, flag)

end

"""
    loglikelihood_coeffs(CoxProd::Vector{Float64}, rf::RestaurantFranchise)

"""
function loglikelihood_coeffs(CoxProd::Vector{Float64}, rf::RestaurantFranchise)

    # initialize loglikelihood
    loglik = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        if rdish == 0       # no tables at dish
            continue
        end

        # retrieve indices
        customers = (rf.X .== dish)         # customers eating dish
        tables = (rf.table_dish .== dish)   # tables eating dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * KernelInt(dish_value, rf.T, CoxProd, rf.eta)

        # compute loglikelihood
        loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], CoxProd[customers], rf.eta)))
        loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
        loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0)

    end

    # integrand function
    function f(x::Float64)
            
        # compute integrand
        return psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, CoxProd, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

    end

    # compute loglikelihood
    loglik -= rf.theta * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    return loglik

end # loglikelihood_coeffs

"""
    loglikelihood_coeffs(CoxProd::Vector{Float64}, rf::RestaurantArray)

"""
function loglikelihood_coeffs(CoxProd::Vector{Float64}, rf::RestaurantArray)

    # initialize loglikelihood
    loglik = 0.0

    # loop on tables
    for (table, ntable) in enumerate(rf.n)

        if ntable == 0      # no customers at dish
            continue
        end

        # retrieve indices
        customers = (rf.X .== table)         # customers seated at table

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # compute loglikelihood
        loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], CoxProd[customers], rf.eta)))
        loglik += logtau(ntable, rf.alpha * KernelInt(dish_value, rf.T, CoxProd, rf.eta), rf.beta, rf.sigma)

    end

    # integrand function
    function f(x::Float64)
            
        # compute integrand
        return psi(rf.alpha * KernelInt(x, rf.T, CoxProd, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

    end

    # compute loglikelihood
    loglik -= rf.theta * rf.D * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    return loglik

end # loglikelihood_coeffs

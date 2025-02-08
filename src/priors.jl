# export functions
export hazard_prior, survival_prior, incidence_prior

"""
    hazard_prior(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64}; cum::Bool = false)

"""
function hazard_prior(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64}; cum::Bool = false)

    # initialize prior vector
    prior_estimate = zeros(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        if isa(rf, RestaurantFranchise)

            # compute integrand
            return rf.alpha * kernel(x, t, rf.eta) * tau(0.0, rf.beta, rf.sigma) * tau(0.0, rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        elseif isa(rf, RestaurantArray)

            # compute integrand
            return rf.alpha * kernel(x, t, rf.eta) * tau(0.0, rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        return 0.0

    end

    # store values
    for (t, time) in enumerate(times)
        prior_estimate[t] = rf.theta * integrate(x::Float64 -> f(x, time), legendre; lower = 0.0, upper = time)
    end

    if cum == true  # integrate over times
        prior_estimate = integrate_trapz(prior_estimate, times; cum = true)
    end

    return prior_estimate

end # hazard_prior

"""
    survival_prior(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64})

"""
function survival_prior(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64})

    # initialize prior vector
    prior_estimate = ones(length(times))

    # base level integrand
    function f(x::Float64, t::Float64)

        if isa(rf, RestaurantFranchise)

            # compute integrand
            return psi(rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        elseif isa(rf, RestaurantArray)

            # compute integrand
            return rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        return 0.0

    end

    # store values
    for (t, time) in enumerate(times)
        prior_estimate[t] = exp( - rf.theta * integrate(x::Float64 -> f(x, time), legendre; lower = 0.0, upper = time) )
    end

    return prior_estimate

end # survival_prior

"""
    incidence_prior(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64}; cum::Bool = false)

"""
function incidence_prior(rf::Union{RestaurantFranchise,RestaurantArray}, times::Vector{Float64}; cum::Bool = false)

    # initialize prior vector
    prior_estimate = zeros(length(times))

    # hazard base level integrand
    function f_hazard(x::Float64, t::Float64)

        if isa(rf, RestaurantFranchise)

            # compute integrand
            out = rf.alpha * kernel(x, t, rf.eta) * tau(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma) 
            out *= tau(rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)
            return out

        elseif isa(rf, RestaurantArray)

            # compute integrand
            return rf.alpha * kernel(x, t, rf.eta) * tau(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        return 0.0

    end

    # survival base level integrand
    function f_survival(x::Float64, t::Float64)

        if isa(rf, RestaurantFranchise)

            # compute integrand
            return psi(rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        elseif isa(rf, RestaurantArray)

            # compute integrand
            return rf.D * psi(rf.alpha * KernelInt(x, t, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        return 0.0

    end

    # store values
    for (t, time) in enumerate(times)
        prior_estimate[t] = rf.theta * integrate(x::Float64 -> f_hazard(x, time), legendre; lower = 0.0, upper = time)
        prior_estimate[t] *= exp( - rf.theta * integrate(x::Float64 -> f_survival(x, time), legendre; lower = 0.0, upper = time) )
    end

    if cum == true  # integrate over times
        prior_estimate = integrate_trapz(prior_estimate, times; cum = true)
    end

    return prior_estimate

end # incidence_prior

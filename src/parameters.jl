# export struct
export Parameters

"""
    struct Parameters

"""
struct Parameters

    # latent structure
    dishes_number::Vector{Int64}
    accept_dishes::Vector{Float64}

    # hyperparameters
    theta::Vector{Float64}    
    logalpha::Vector{Float64}
    logkappa::Vector{Float64}

    # acceptance probability
    accept_alpha::Vector{Float64}
    accept_kappa::Vector{Float64}

    # regression coefficients
    eta::Vector{Float64}
    accept_coeffs::Vector{Float64}

    # loglikelihood
    loglik::Vector{Float64}

    # explicit constructor
    function Parameters()

        # initialize latent structure vector
        dishes_number = Array{Int64}(undef, 0)
        accept_dishes = Array{Float64}(undef, 0)

        # hyperparameters
        theta = Array{Float64}(undef, 0)
        logalpha = Array{Float64}(undef, 0)
        logkappa = Array{Float64}(undef, 0)

        # acceptance probability
        accept_alpha = Array{Float64}(undef, 0)
        accept_kappa = Array{Float64}(undef, 0)

        # regression coefficients
        eta = Array{Float64}(undef, 0)
        accept_coeffs = Array{Float64}(undef, 0)

        # loglikelihood
        loglik = Array{Float64}(undef, 0)

        # create Parameters
        return new(dishes_number, accept_dishes, theta, logalpha, logkappa, accept_alpha, accept_kappa, eta, accept_coeffs, loglik)

    end # Parameters

end # struct

"""
    append(params::Parameters, rf::Restaurants, cm::Union{CoxModel,Nothing}, accept_dishes::Float64, accept_alpha::Float64, accept_kappa::Float64, accept_coeffs::Vector{Float64})

"""
function append(params::Parameters, rf::Restaurants, cm::Union{CoxModel,Nothing}, accept_dishes::Float64, accept_alpha::Float64, accept_kappa::Float64, accept_coeffs::Vector{Float64})

    # dishes number
    dishes_number = count(rf.n .> 0)

    # append dishes number and acceptance probability
    push!(params.dishes_number, dishes_number)
    push!(params.accept_dishes, accept_dishes / dishes_number)

    # append hyperparameters
    push!(params.theta, rf.theta)
    push!(params.logalpha, log(rf.alpha))
    push!(params.logkappa, log(rf.kappa))

    # append acceptance probability
    push!(params.accept_alpha, accept_alpha)
    push!(params.accept_kappa, accept_kappa)

    # append coefficients and acceptance probability
    append!(params.eta, isnothing(cm) ? 0.0 : cm.eta)
    append!(params.accept_coeffs, accept_coeffs)

    # append loglikelihood
    push!(params.loglik, loglikelihood(rf))

end # append

"""
    loglikelihood(rf::Restaurants)

"""
function loglikelihood(rf::Restaurants)

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
        KInt = rf.alpha * rf.KInt[dish]

        # compute loglikelihood
        loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], isnothing(rf.CoxProd) ? nothing : rf.CoxProd[customers], rf.kappa)))

        if rf.hierarchical      # restaurant franchise

            # retrieve indices
            tables = (rf.table_dish .== dish)   # tables eating dish

            # compute loglikelihood
            loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma)) + logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) + log(rf.theta)

        else    # restaurant Array

            # compute loglikelihood
            loglik += logtau(rdish, KInt, rf.beta, rf.sigma) + log(rf.theta) 

        end

    end

    # compute loglikelihood
    if rf.hierarchical      # restaurant franchise
        loglik -= rf.theta * integrate(x -> psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant Array
        loglik -= rf.theta * integrate(x -> rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.kappa), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    return loglik

end # loglikelihood

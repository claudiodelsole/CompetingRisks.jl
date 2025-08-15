# export struct
export Parameters

"""
    struct Parameters

Posterior samples of model parameters and acceptance rates. 
Output of [`posterior_sampling`](@ref) and input for the summary functions [`summary_dishes`](@ref), [`summary_theta`](@ref), [`summary_kernelpars`](@ref), [`summary_coefficients`](@ref), and [`summary_logevidence`](@ref).
"""
struct Parameters

    # latent structure
    dishes_number::Vector{Int64}
    accept_dishes::Vector{Float64}

    # hyperparameters
    theta::Vector{Float64}    

    # kernel parameters
    KernelType::DataType
    kernelpars::Vector{Float64}
    accept_kernelpars::Vector{Float64}

    # regression coefficients
    eta::Vector{Float64}
    accept_coeffs::Vector{Float64}

    # logevidence
    logevidence::Vector{Float64}

    # explicit constructor
    function Parameters(KernelType::DataType)

        # latent structure
        dishes_number = Array{Int64}(undef, 0)
        accept_dishes = Array{Float64}(undef, 0)

        # hyperparameters
        theta = Array{Float64}(undef, 0)

        # kernel parameters
        kernelpars = Array{Float64}(undef, 0)
        accept_kernelpars = Array{Float64}(undef, 0)

        # regression coefficients
        eta = Array{Float64}(undef, 0)
        accept_coeffs = Array{Float64}(undef, 0)

        # logevidence
        logevidence = Array{Float64}(undef, 0)

        # create Parameters
        return new(dishes_number, accept_dishes, theta, KernelType, kernelpars, accept_kernelpars, eta, accept_coeffs, logevidence)

    end # Parameters

end # struct

"""
    append(params::Parameters, rf::Restaurants, cm::Union{CoxModel,Nothing}, accept_dishes::Float64, accept_kernelpars::Float64, accept_coeffs::Vector{Float64})

"""
function append(params::Parameters, rf::Restaurants, cm::Union{CoxModel,Nothing}, accept_dishes::Float64, accept_kernelpars::Float64, accept_coeffs::Vector{Float64})

    # dishes number
    dishes_number = count(rf.n .> 0)

    # append dishes number and acceptance probability
    push!(params.dishes_number, dishes_number)
    push!(params.accept_dishes, accept_dishes / dishes_number)

    # append hyperparameters
    push!(params.theta, rf.theta)

    # append kernel parameters
    append!(params.kernelpars, vectorize(rf.kernelpars))
    push!(params.accept_kernelpars, accept_kernelpars)

    # append coefficients and acceptance probability
    append!(params.eta, isnothing(cm) ? 0.0 : cm.eta)
    append!(params.accept_coeffs, accept_coeffs)

    # append logevidence
    push!(params.logevidence, logevidence(rf))

end # append

"""
    logevidence(rf::Restaurants)

"""
function logevidence(rf::Restaurants)

    # initialize logevidence
    logevidence = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        # no tables at dish
        if rdish == 0 continue end

        # retrieve indices
        customers = (rf.X .== dish)         # customers eating dish
        
        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.KInt[dish]

        # compute logevidence
        logevidence += sum(log.(kernel.(dish_value, rf.T[customers], isnothing(rf.CoxProd) ? nothing : rf.CoxProd[customers], [rf.kernelpars])))

        if rf.hierarchical      # restaurant franchise

            # retrieve indices
            tables = (rf.table_dish .== dish)   # tables eating dish

            # compute logevidence
            logevidence += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma)) + logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) + log(rf.theta)

        else    # restaurant Array

            # compute logevidence
            logevidence += logtau(rdish, KInt, rf.beta, rf.sigma) + log(rf.theta) 

        end

    end

    # compute logevidence
    if rf.hierarchical      # restaurant franchise
        logevidence -= rf.theta * integrate(x -> psi(rf.D * psi(KernelInt(x, rf.T, rf.CoxProd, rf.kernelpars), rf.beta, rf.sigma), rf.beta0, rf.sigma0), 0.0, maximum(rf.T))
    else    # restaurant Array
        logevidence -= rf.theta * integrate(x -> rf.D * psi(KernelInt(x, rf.T, rf.CoxProd, rf.kernelpars), rf.beta, rf.sigma), 0.0, maximum(rf.T))
    end

    return logevidence

end # logevidence

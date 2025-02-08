module CompetingRisks

# import from packages
include("imports.jl")

# export structs
export CompetingRisksDataset, CoxModel, RestaurantFranchise, RestaurantArray

"""
    struct CompetingRisksDataset

"""
struct CompetingRisksDataset

    # dimensions
    N::Int64                # number of patients
    D::Int64                # number of causes

    # observed variables
    T::Vector{Float64}          # observations, time-to-event (dim N)
    Delta::Vector{Int64}        # event causes (dim N)
    predictors::Vector{Int64}   # categorical predictors (dim N)

    # explicit constructor
    function CompetingRisksDataset(T::Vector{Float64}, Delta::Vector{Int64})

        # retrieve dimension
        N = length(Delta)       # number of patients
        D = maximum(Delta)      # number of causes

        # check consistency
        @assert length(T) == N

        # indexing in descending order
        idxs = sortperm(T, rev = true)

        # define empty predictors
        predictors = zeros(Int64, N)

        # create CompetingRisksDataset
        return new(N, D, T[idxs], Delta[idxs], predictors[idxs])

    end # CompetingRisksDataset

    # explicit constructor
    function CompetingRisksDataset(T::Vector{Float64}, Delta::Vector{Int64}, predictors::Vector{Int64})

        # retrieve dimension
        N = length(Delta)       # number of patients
        D = maximum(Delta)      # number of causes

        # check consistency
        @assert length(T) == N
        @assert length(predictors) == N

        # indexing in descending order
        idxs = sortperm(T, rev = true)

        # create CompetingRisksDataset
        return new(N, D, T[idxs], Delta[idxs], predictors[idxs])

    end # CompetingRisksDataset

end # struct

# functions and utilities
include("model_functions.jl")
include("utils.jl")
include("constants.jl")

"""
    struct CoxModel

"""
struct CoxModel

    # dimensions
    N::Int64        # number of patients
    L::Int64        # number of categorical levels

    # categorical predictors
    predictors::Vector{Int64}

    # regression coefficients
    xi::Vector{Float64}             # regression coefficients (dim L)

    # explicit constructor
    function CoxModel(crd::CompetingRisksDataset)

        # retrieve predictors
        predictors = crd.predictors

        # compute dimensions
        L = maximum(predictors)     # number of categorical levels

        # initialize regression coefficients
        xi = zeros(Float64, L)          # regression coefficients (dim L)

        # create CoxModel
        return new(crd.N, L, predictors, xi)

    end # CoxModel

end # struct

"""
    mutable struct RestaurantFranchise

"""
mutable struct RestaurantFranchise

    # dimensions
    N::Int64                # number of customers
    D::Int64                # number of restaurants

    # observed variables
    T::Vector{Float64}      # observations, time-to-event (dim N)
    Delta::Vector{Int64}    # restaurants (dim N)

    # Cox regression model
    CoxProd::Union{Vector{Float64},Nothing}     # exponential CoxProduct for observations (dim N)

    # latent variables
    X::Vector{Int64}        # dishes indices (dim N)
    Z::Vector{Int64}        # tables indices (dim N), restaurant is implicit
    Xstar::Vector{Float64}  # dishes distinct values, i.e. menu (dim k)

    # counts vectors
    n::Vector{Int64}        # customers counts (dim k), customers per dish
    r::Vector{Int64}        # tables counts (dim k), tables per dish
    q::Vector{Int64}        # customers counts (dim D x rmax), customers per table

    # lookup vectors
    table_dish::Vector{Int64}   # dish indices (dim D x rmax), dish index per table
    table_rest::Vector{Int64}   # restaurant indices (dim D x rmax), restaurant index per table

    # precomputed quantities
    KInt::Vector{Float64}       # KernelInt for dishes distinct values (dim k)
    mass_base::Vector{Float64}  # mass_base for observations (dim N)

    # hierarchical CRM parameters
    beta::Float64           # restaurant-level measures
    sigma::Float64          # restaurant-level measures
    beta0::Float64          # base measure
    sigma0::Float64         # base measure

    # model hyperparameters
    theta::Float64          # base measure mass
    alpha::Float64          # kernel height
    eta::Float64            # kernel shape

    # base measure
    base_measure::Uniform

    # integration utility
    # legendre::LegendreIntegral      # quadrature nodes and weights

    # explicit constructor
    function RestaurantFranchise(crd::CompetingRisksDataset; theta::Float64 = 1.0, alpha::Float64 = 1.0, eta::Float64 = 1.0,
            beta::Float64 = 1.0, sigma::Float64 = 0.0, beta0::Float64 = 1.0, sigma0::Float64 = 0.0)

        # retrieve dimensions
        N = crd.N       # number of customers  
        D = crd.D       # number of restaurants

        # define Cox regression model
        CoxProd = ones(Float64, N)      # exponential CoxProduct for observations (dim N)
        if maximum(crd.predictors) == 0 
            CoxProd = nothing 
        end

        # set dimensions
        k = crd.D       # number of dishes
        rmax = 3        # number of tables per restaurant
        
        # initialize latent variables
        X = zeros(Int64, N)         # dishes indices (dim N)
        Z = zeros(Int64, N)         # tables indices (dim N)
        Xstar = zeros(Float64, k)   # dishes distinct values, i.e. menu (dim k)

        # initialize counts vectors
        n = zeros(Int64, k)         # customers counts (dim k), customers per dish
        r = zeros(Int64, k)         # tables counts (dim k), tables per dish
        q = zeros(Int64, D * rmax)  # customers counts (dim D x rmax), customers per table

        # initialize lookup vectors
        table_dish = zeros(Int64, D * rmax)     # dish indices (dim D x rmax), dish index per table
        table_rest = zeros(Int64, D * rmax)     # restaurant indices (dim D x rmax), restaurant index per table

        # precomputed quantities
        KInt = zeros(Float64, k)        # KernelInt for dishes distinct values (dim k)
        mass_base = zeros(Float64, N)   # mass_base for observations (dim N)

        # base measure
        base_measure = Uniform(0.0, maximum(crd.T))

        # integration utility
        # legendre = LegendreIntegral()       # quadrature nodes and weights

        # create RestaurantFranchise
        return new(N, D, crd.T, crd.Delta, CoxProd, X, Z, Xstar, n, r, q, table_dish, table_rest, KInt, mass_base, 
            beta, sigma, beta0, sigma0, theta, alpha, eta, base_measure)

    end # RestaurantFranchise

end # struct

"""
    mutable struct RestaurantArray

"""
mutable struct RestaurantArray

    # dimensions
    N::Int64                # number of customers
    D::Int64                # number of restaurants

    # observed variables
    T::Vector{Float64}      # observations, time-to-event (dim N)
    Delta::Vector{Int64}    # restaurants (dim N)

    # Cox regression model
    CoxProd::Union{Vector{Float64},Nothing}     # exponential CoxProduct for observations (dim N)

    # latent variables
    X::Vector{Int64}        # tables indices (dim N), restaurant is implicit
    Xstar::Vector{Float64}  # dishes distinct values (dim k)

    # counts vector
    n::Vector{Int64}        # customers counts (dim k), customers per table

    # lookup vector
    table_rest::Vector{Int64}   # restaurant indices (k), restaurant index per table

    # precomputed quantities
    KInt::Vector{Float64}       # KernelInt for dishes distinct values (dim k)
    mass_base::Vector{Float64}  # mass_base for observations (dim N)

    # hierarchical CRM parameters
    beta::Float64           # restaurant-level measures
    sigma::Float64          # restaurant-level measures

    # model hyperparameters
    theta::Float64          # restaurant-level measure mass (starting value)
    alpha::Float64          # kernel height
    eta::Float64            # kernel shape

    # base measure
    base_measure::Uniform

    # integration utility
    # legendre::LegendreIntegral      # quadrature nodes and weights

    # explicit constructor
    function RestaurantArray(crd::CompetingRisksDataset; theta::Float64 = 1.0, alpha::Float64 = 1.0, eta::Float64 = 1.0, 
            beta::Float64 = 1.0, sigma::Float64 = 0.0)

        # retrieve dimensions
        N = crd.N       # number of customers  
        D = crd.D       # number of restaurants

        # define Cox regression model
        CoxProd = ones(Float64, N)          # exponential CoxProduct for observations (dim N)
        if maximum(crd.predictors) == 0 
            CoxProd = nothing 
        end

        # set dimensions
        k = crd.D       # number of tables
        
        # initialize latent variables
        X = zeros(Int64, N)         # tables indices (dim N), restaurant is implicit
        Xstar = zeros(Float64, k)   # dishes distinct values (dim k)

        # initialize counts vector
        n = zeros(Int64, k)         # customers counts (dim k), customers per table

        # initialize lookup vector
        table_rest = zeros(Int64, k)    # restaurant indices (dim k), restaurant index per table

        # precomputed quantities
        KInt = zeros(Float64, k)        # KernelInt for dishes distinct values (dim k)
        mass_base = zeros(Float64, N)   # mass_base for observations (dim N)

        # base measure
        base_measure = Uniform(0.0, maximum(crd.T))

        # integration utility
        # legendre = LegendreIntegral()       # quadrature nodes and weights

        # create RestaurantArray
        return new(N, D, crd.T, crd.Delta, CoxProd, X, Xstar, n, table_rest, KInt, mass_base, 
            beta, sigma, theta, alpha, eta, base_measure)

    end # RestaurantArray

end # struct

# create datasets
include("create_datasets.jl")

# random measures
include("random_measures.jl")

# prior estimates
include("priors.jl")

# marginal and conditional estimates
include("marginal_samples.jl")
include("conditional_samples.jl")
include("estimates.jl")

# diagnostics
include("diagnostics.jl")

# gibbs sampling
include("gibbs.jl")
include("resampling.jl")

# plots
include("plots.jl")

# errors
include("errors.jl")

# frequentist estimators
include("frequentist.jl")

end # module

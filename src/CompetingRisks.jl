module CompetingRisks

# import from packages
include("imports.jl")

# export structs
export Restaurants, CoxModel, Estimator

"""
    mutable struct Restaurants

"""
mutable struct Restaurants

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
    q::Vector{Int64}        # customers counts (dim sumr), customers per table

    # lookup vectors
    table_dish::Vector{Int64}   # dish indices (dim sumr), dish index per table
    table_rest::Vector{Int64}   # restaurant indices (dim sumr), restaurant index per table

    # precomputed quantities
    KInt::Vector{Float64}       # KernelInt for dishes distinct values (dim k)
    mass_base::Vector{Float64}  # mass_base for observations (dim N)

    # CRM parameters
    beta::Float64           # restaurant-level measures
    sigma::Float64          # restaurant-level measures
    beta0::Float64          # base measure
    sigma0::Float64         # base measure

    # model hyperparameters
    theta::Float64          # concentration
    alpha::Float64          # kernel height
    kappa::Float64          # kernel shape

    # hierarchical flag
    hierarchical::Bool      # hierarchical model flag

end # Restaurants

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
    eta::Vector{Float64}            # regression coefficients (dim L)

end # CoxModel

"""
    struct Estimator

"""
struct Estimator

    # dimensions
    D::Int64        # number of causes
    L::Int64        # number of categorical predictors

    # times vector
    times::Vector{Float64}

    # posterior samples
    survival_samples::Vector{Float64}
    hazard_samples::Vector{Float64}

end # Estimator

# constructors
include("constructors.jl")

# functions and utilities
include("model_functions.jl")
include("utils.jl")
include("constants.jl")

# random measures
include("random_measures.jl")

# marginal and conditional estimates
include("marginal_samples.jl")
include("conditional_samples.jl")
include("estimates.jl")

# parameters
include("parameters.jl")

# gibbs sampling
include("gibbs.jl")
include("resampling.jl")

# plots
include("plots_parameters.jl")
include("traceplots.jl")
include("plots.jl")

end # module

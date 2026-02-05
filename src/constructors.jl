"""
    CompetingRisksModel(KernelType::Type{<:AbstractKernel}; beta::Float64 = 1.0, sigma::Float64 = 0.0, beta0::Float64 = 1.0, sigma0::Float64 = 0.0, hierarchical::Bool = true, regression::Bool = false)

Kernel specification, random measures parameters, hierachical or independent model, regression or exchangeable data.

The `KernelType` is a subtype of `AbstractKernel`, to choose among `DykstraLaudKernel`, `OrnsteinUhlenbeckKernel` and `RectangularKernel`.
Parameters `beta` and `beta0` are rate parameters, `sigma` and `sigma0` are discount paramters. Base measure parameters `beta0` and `sigma0` are used only if `hierarchical = true`.
The currently available Cox regression model involves categorical predictors.
"""
function CompetingRisksModel(KernelType::Type{<:AbstractKernel}; beta::Float64 = 1.0, sigma::Float64 = 0.0, beta0::Float64 = 1.0, sigma0::Float64 = 0.0, hierarchical::Bool = true, regression::Bool = false)

    # create CompetingRisksModel
    return CompetingRisksModel(KernelType, beta, sigma, beta0, sigma0, hierarchical, regression)

end # CompetingRisksModel

"""
    Restaurants(data::DataFrame, kernelpars::KernelType, beta::Float64, sigma::Float64, beta0::Float64, sigma0::Float64, hierarchical::Bool) where KernelType <: AbstractKernel

"""
function Restaurants(data::DataFrame, kernelpars::KernelType, beta::Float64, sigma::Float64, beta0::Float64, sigma0::Float64, hierarchical::Bool) where KernelType <: AbstractKernel

    # retrieve dimension
    N = length(data.Delta)      # number of customers
    D = maximum(data.Delta)     # number of restaurants

    # define Cox regression model
    CoxProd = ones(Float64, N)      # exponential CoxProduct for observations (dim N)
    if maximum(data.predictor) == 0 
        CoxProd = nothing 
    end

    # set dimensions
    k = D           # number of dishes
    sumr = 3 * k    # number of tables per restaurant
    
    # initialize latent variables
    X = zeros(Int64, N)         # dishes indices (dim N)
    Z = zeros(Int64, N)         # tables indices (dim N)
    Xstar = zeros(Float64, k)   # dishes distinct values (dim k)

    # initialize counts vectors
    n = zeros(Int64, k)         # customers counts (dim k), customers per dish
    r = zeros(Int64, k)         # tables counts (dim k), tables per dish
    q = zeros(Int64, sumr)      # customers counts (dim sumr), customers per table

    # initialize lookup vectors
    table_dish = zeros(Int64, sumr)     # dish indices (dim sumr), dish index per table
    table_rest = zeros(Int64, sumr)     # restaurant indices (dim sumr), restaurant index per table

    # precomputed quantities
    KInt = zeros(Float64, k)        # KernelInt for dishes distinct values (dim k)
    mass_base = zeros(Float64, N)   # mass_base for observations (dim N)

    # initialize CRM hyperparameters
    theta = 1.0     # concentration

    # indexing in descending order
    idxs = sortperm(data.T, rev = true)

    # create Restaurants
    return Restaurants{KernelType}(N, D, data.T[idxs], data.Delta[idxs], CoxProd, X, Z, Xstar, n, r, q, table_dish, table_rest, KInt, mass_base, kernelpars, theta, beta, sigma, beta0, sigma0, hierarchical)

end # Restaurants

"""
    CoxModel(data::DataFrame)

"""
function CoxModel(data::DataFrame)

    # retrieve dimensions
    N = length(data.predictor)      # number of patients
    L = maximum(data.predictor)     # number of categorical levels

    # initialize regression coefficients
    eta = zeros(Float64, L)     # regression coefficients (dim L)

    # indexing in descending order
    idxs = sortperm(data.T, rev = true)

    # create CoxModel
    return CoxModel(N, L, data.predictor[idxs], eta)

end # CoxModel

"""
    Estimator(rf::Restaurants, cm::Union{CoxModel,Nothing}, times::Vector{Float64})

"""
function Estimator(rf::Restaurants, cm::Union{CoxModel,Nothing}, times::Vector{Float64})

    # initialize vectors
    survival_samples = Array{Float64}(undef, 0)
    hazard_samples = Array{Float64}(undef, 0)

    # create Estimator
    return Estimator(rf.D, isnothing(cm) ? 0 : cm.L, times, survival_samples, hazard_samples)

end # Estimator

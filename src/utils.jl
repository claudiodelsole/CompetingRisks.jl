# export struct
export LegendreIntegral

# export functions
export hazard, survival, integrate, integrate_trapz

"""
    hazard(d::Type, t::Float64) where Type <: UnivariateDistribution

"""
hazard(d::Type, t::Float64) where Type <: UnivariateDistribution = pdf(d, t) / ccdf(d, t)

"""
    survival(d::Type, t::Float64) where Type <: UnivariateDistribution

"""
survival(d::Type, t::Float64) where Type <: UnivariateDistribution = ccdf(d, t)

"""
    struct LegendreIntegral

"""
struct LegendreIntegral

    # dimension
    num_nodes::Int64

    # quadrature nodes and weights
    nodes::Vector{Float64}
    weights::Vector{Float64}

    # explicit constructor
    function LegendreIntegral( ; num_nodes::Int64 = 20)

        # quadrature nodes and weights
        nodes, weights = gausslegendre(num_nodes)

        # create LegendreIntegral
        return new(num_nodes, nodes, weights)

    end # LegendreIntegral

end # struct

"""
    integrate(f::Function, legendre::LegendreIntegral; lower::Float64 = -1.0, upper::Float64 = 1.0)

"""
function integrate(f::Function, legendre::LegendreIntegral; lower::Float64 = -1.0, upper::Float64 = 1.0)

    # retrieve domain length
    length = upper - lower

    # rescale nodes
    nodes = 0.5 * length * (legendre.nodes .+ 1.0) .+ lower

    # compute integral
    I = 0.0
    for (weight, node) in zip(legendre.weights, nodes)
        I += weight * f(node)
    end
    I *= (0.5 * length)

    # return integral value
    return I

end # integrate

"""
    integrate_trapz(f::Vector{Float64}, times::Vector{Float64}; cum::Bool = false)

"""
function integrate_trapz(f::Vector{Float64}, times::Vector{Float64}; cum::Bool = false)

    # trapezoid integrals
    trapz = (0.5 * f[begin:end-1] + 0.5 * f[begin+1:end]) .* diff(times)

    # compute integrals
    if cum == true
        return pushfirst!(cumsum(trapz), 0.0)
    else
        return sum(trapz)
    end

end # integrate_trapz

"""
    newton(value::Float64, f::Function, fp::Function, start::Float64)

"""
function newton(value::Float64, f::Function, fp::Function, start::Float64)

    # algorithm parameters
    tol = 1.0e-8        # tolerance
    maxIter = 50        # number of iterations

    # initialize
    sol = start
    fsol = f(sol)

    for _ in 1:maxIter

        # algorithm step
        sol -= (fsol - value) / fp(sol)

        # precompute function
        fsol = f(sol)

        # check convergence
        if abs(fsol - value) < tol
            return sol
        end

    end

    # !! algorithm does not converge !!
    return -Inf

end # newton

"""
    sample_categorical(masses::Vector{Float64})

"""
function sample_categorical(masses::Vector{Float64})

    # sample from uniform distribution
    th = rand() * sum(masses)

    # initialize variables
    sum_masses = masses[begin]
    K = 1

    # sample from categorical distribution
    while sum_masses < th
        K += 1
        sum_masses += masses[K]
    end

    # return sampled category
    return K

end # sample_categorical

"""
    struct LegendreIntegral

"""
struct LegendreIntegral

    # quadrature nodes and weights
    nodes::Vector{Float64}
    weights::Vector{Float64}

end # LegendreIntegral

"""
    function LegendreIntegral( ; num_nodes::Int64 = 5)

"""
function LegendreIntegral( ; num_nodes::Int64 = 5)

    # quadrature nodes and weights on (-1,1)
    nodes, weights = gausslegendre(num_nodes)

    # rescale nodes and weights on (0,1)
    nodes = 0.5 * (nodes .+ 1.0)
    weights = 0.5 * weights

    # create LegendreIntegral
    return LegendreIntegral(nodes, weights)

end # LegendreIntegral

"""
    integrate(f::Function, lower::Float64, upper::Float64; num_intvals::Int64 = 5)

"""
function integrate(f::Function, lower::Float64, upper::Float64; num_intvals::Int64 = 5)

    # retrieve endpoints
    endpoints = range(lower, upper, num_intvals + 1)

    # compute integral
    I = 0.0
    for (lower_, upper_) in zip(endpoints[begin:end-1], endpoints[begin+1:end])
        I += integrate_legendre(f, lower_, upper_)
    end

    # return integral value
    return I

end # integrate

"""
    integrate_legendre(f::Function, lower::Float64, upper::Float64)

"""
function integrate_legendre(f::Function, lower::Float64, upper::Float64)

    # retrieve domain length
    length = (upper - lower)

    # compute integral
    I = 0.0
    for (weight, node) in zip(legendre.weights, legendre.nodes)
        I += weight * f(length * node + lower)
    end
    I *= length

    # return integral value
    return I

end # integrate

"""
    integrate_trapz(values::Vector{Float64}, times::Vector{Float64}; cum::Bool = false)

"""
function integrate_trapz(values::Vector{Float64}, times::Vector{Float64}; cum::Bool = false)

    # trapezoid integrals
    trapz = 0.5 * (values[begin:end-1] + values[begin+1:end]) .* diff(times)

    # compute integrals
    if cum == true
        return pushfirst!(cumsum(trapz), 0.0)
    end

    return sum(trapz)

end # integrate_trapz

"""
    newton(value::Float64, f::Function, fp::Function, start::Float64)

"""
function newton(value::Float64, f::Function, fp::Function, start::Float64)

    # algorithm parameters
    tol = 1.0e-8        # tolerance
    maxIter = 100       # number of iterations

    # initialize
    sol, fsol = start, f(start)

    for _ in range(1, maxIter)

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
    K = 1
    sum_masses = masses[begin]

    # sample from categorical distribution
    while sum_masses < th
        K += 1
        sum_masses += masses[K]
    end

    # return sampled category
    return K

end # sample_categorical

# import utils
import FastGaussQuadrature: gausslegendre
import Distributions: UnivariateDistribution, pdf, ccdf
import KernelDensitySJ: bwsj

# export functions
export hazard, survival, integrate_trapz

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

    # println("error")

    return 0.0

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

"""
    smoother(values::Vector{Float64}, times::Vector{Float64}; bandwidth::Union{Float64,Nothing} = nothing, 
            left::Function = t -> values[begin], right::Function = t -> values[end], eval_times::Vector{Float64} = times)

"""
function smoother(values::Vector{Float64}, times::Vector{Float64}; bandwidth::Union{Float64,Nothing} = nothing,
        left::Function = t -> values[begin], right::Function = t -> values[end], eval_times::Vector{Float64} = times)

    # compute bandwidth
    if isnothing(bandwidth)
        bandwidth = bwsj(eval_times)
    end

    # maximum bandwidth
    max_width = 3.0 * bandwidth

    # define extended times
    dt = diff(times)[begin]
    times_ext = dt:dt:max_width

    # initialize output
    smoothed = zeros(length(eval_times))

    # loop on values
    for (t, time) in enumerate(eval_times)

        # initialize auxiliary variables
        weights = 0.0

        # left extension
        for stime in times_ext

            # compute time distance
            tdist = abs(time - minimum(times)) + stime

            if tdist > max_width    # time distance larger than max_width
                break
            end

            # compute weight
            weight = pdf(Normal(), tdist / bandwidth)

            # update weights and smoothed values
            weights += weight
            smoothed[t] += weight * left(minimum(times) - stime)

        end

        for (s, stime) in enumerate(times)

            # compute time distance
            tdist = abs(stime - time)

            if tdist > max_width    # time distance larger than max_width
                continue
            end

            # compute weight
            weight = pdf(Normal(), tdist / bandwidth)

            # update weights and smoothed values
            weights += weight
            smoothed[t] += weight * values[s]

        end

        # right extension
        for stime in times_ext

            # compute time distance
            tdist = abs(time - maximum(times)) + stime

            if tdist > max_width    # time distance larger than max_width
                break
            end

            # compute weight
            weight = pdf(Normal(), tdist / bandwidth)

            # update weights and smoothed values
            weights += weight
            smoothed[t] += weight * right(maximum(times) + stime)

        end

        # normalize smoothed value
        smoothed[t] /= weights

    end

    return smoothed

end # smoother

"""
    smoother(values::Matrix{Float64}, times::Vector{Float64}; bandwidth::Union{Float64,Nothing} = nothing, 
            left::Function = t -> values[begin,:], right::Function = t -> values[end,:], eval_times::Vector{Float64} = times)

"""
function smoother(values::Matrix{Float64}, times::Vector{Float64}; bandwidth::Union{Float64,Nothing} = nothing,
        left::Function = t -> values[begin,:], right::Function = t -> values[end,:], eval_times::Vector{Float64} = times)

    # number of functions
    D = size(values, 2)

    # initialize output
    smoothed = zeros(length(eval_times), D)

    # loop on functions
    for d in 1:D
        smoothed[:,d] = smoother(values[:,d], times; 
            bandwidth = bandwidth, left = t -> left(t)[d], right = t -> right(t)[d], eval_times = eval_times)
    end

    return smoothed

end # smoother

"""
    smoother(values::Array{Float64,3}, times::Vector{Float64}; bandwidth::Union{Float64,Nothing} = nothing, 
            left::Function = t -> values[begin,:,:], right::Function = t -> values[end,:,:], eval_times::Vector{Float64} = times)

"""
function smoother(values::Array{Float64,3}, times::Vector{Float64}; bandwidth::Union{Float64,Nothing} = nothing,
        left::Function = t -> values[begin,:,:], right::Function = t -> values[end,:,:], eval_times::Vector{Float64} = times)

    # number of categorical levels
    L = size(values, 2)

    # number of functions
    D = size(values, 3)

    # initialize output
    smoothed = zeros(length(eval_times), L, D)

    # loop on functions
    for l in 1:L
        for d in 1:D
            smoothed[:,l,d] = smoother(values[:,l,d], times; 
                bandwidth = bandwidth, left = t -> left(t)[l,d], right = t -> right(t)[l,d], eval_times = eval_times)
        end
    end

    return smoothed

end # smoother

"""
    KernelInt(x::Float64, T::Vector{Float64}, _::Nothing, kernelpars::KernelType) where KernelType <: AbstractKernel

"""
function KernelInt(x::Float64, T::Vector{Float64}, _::Nothing, kernelpars::KernelType) where KernelType <: AbstractKernel

    # initialize KernelInt
    KInt = 0.0

    for t in T      # loop on times

        # !! for times in descending order !!
        if t <= x break end

        # compute KernelInt
        KInt += KernelInt(x, t, nothing, kernelpars)

    end
    
    return KInt
    
end # KernelInt

"""
    KernelInt(x::Float64, T::Vector{Float64}, CoxProd::Vector{Float64}, kernelpars::KernelType) where KernelType <: AbstractKernel

"""
function KernelInt(x::Float64, T::Vector{Float64}, CoxProd::Vector{Float64}, kernelpars::KernelType) where KernelType <: AbstractKernel

    # initialize KernelInt
    KInt = 0.0

    for (t, cp) in zip(T, CoxProd)      # loop on times

        # !! for times in descending order !!
        if t <= x break end

        # compute KernelInt
        KInt += KernelInt(x, t, cp, kernelpars)

    end
    
    return KInt
    
end # KernelInt

"""
    psi(u::Float64, beta::Float64, sigma::Float64)

"""
function psi(u::Float64, beta::Float64, sigma::Float64)
    
    if sigma == 0.0     # gamma CRM
        return log( 1.0 + u / beta )
    end

    return ( (beta + u)^sigma - (beta)^sigma ) / sigma 

end # psi

"""
    tau(u::Float64, beta::Float64, sigma::Float64)
    tau(m::Int64, u::Float64, beta::Float64, sigma::Float64)
    tau_ratio(m::Int64, u::Float64, beta::Float64, sigma::Float64)

"""
tau(u::Float64, beta::Float64, sigma::Float64) = (beta + u) ^ (sigma-1.0)
tau(m::Int64, u::Float64, beta::Float64, sigma::Float64) = gamma(m-sigma) / gamma(1.0-sigma) * (beta + u) ^ (sigma-m)
tau_ratio(m::Int64, u::Float64, beta::Float64, sigma::Float64) = (m-sigma) / (beta + u)

"""
    logtau(m::Int64, u::Float64, beta::Float64, sigma::Float64)
    logtau_diff(m::Int64, u::Float64, beta::Float64, sigma::Float64)

"""
logtau(m::Int64, u::Float64, beta::Float64, sigma::Float64) = loggamma(m-sigma) - loggamma(1.0-sigma) - (m-sigma) * log(beta + u)
logtau_diff(m::Int64, u::Float64, beta::Float64, sigma::Float64) = - (m-sigma) * ( log(beta + u) - log(beta) )

"""
    sample_jump(m::Int64, beta::Float64, sigma::Float64)

"""
sample_jump(m::Int64, beta::Float64, sigma::Float64) = rand(Gamma(m - sigma)) / beta

"""
    tail_integral(logheight::Float64, beta::Float64, sigma::Float64)

"""
function tail_integral(logheight::Float64, beta::Float64, sigma::Float64)

    if sigma == 0.0     # gamma CRM
        return tail_integral(logheight, beta)
    end

    if log(beta) + logheight < - 100.0  # asymptotic behaviour
        return exp( - sigma * logheight ) / ( sigma * gamma(1.0 - sigma) )
    end

    return (beta)^sigma * gamma( - sigma, beta * exp(logheight) ) / gamma(1.0 - sigma)

end # tail_integral

"""
    tail_integral(logheight::Float64, beta::Float64)

"""
function tail_integral(logheight::Float64, beta::Float64)

    if log(beta) + logheight < - 100.0  # asymptotic behaviour
        return - eulergamma - log(beta) - logheight
    end

    return gamma( 0.0, beta * exp(logheight) )

end # tail_integral

"""
    tail_integral_grad(logheight::Float64, beta::Float64, sigma::Float64)

"""
function tail_integral_grad(logheight::Float64, beta::Float64, sigma::Float64)

    if sigma == 0.0     # gamma CRM 
        return tail_integral_grad(logheight, beta)
    end

    if log(beta) + logheight < - 100.0  # asymptotic behaviour
        return - exp( - sigma * logheight ) / gamma(1.0 - sigma)
    end

    return - exp( - sigma * logheight - beta * exp(logheight) ) / gamma(1.0 - sigma)

end # tail_integral_grad

"""
    tail_integral_grad(logheight::Float64, beta::Float64)

"""
function tail_integral_grad(logheight::Float64, beta::Float64)

    if log(beta) + logheight < - 100.0  # asymptotic behaviour
        return - 1.0
    end

    return - exp( - beta * exp(logheight) )

end # tail_integral_grad

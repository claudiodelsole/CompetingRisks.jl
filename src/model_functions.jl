"""
    kernel(x::Float64, t::Float64, cp::Float64, kappa::Float64)

"""
kernel(_::Float64, _::Float64, _::Nothing, _::Float64) = NaN
kernel(_::Float64, _::Float64, _::Float64, _::Float64) = NaN

# Dykstra & Laud kernel
kernel_DL(x::Float64, t::Float64, _::Nothing, _::Float64) = Float64(x <= t)
kernel_DL(x::Float64, t::Float64, cp::Float64, _::Float64) = (cp) * Float64(x <= t)

# Ornstein–Uhlenbeck kernel
kernel_OU(x::Float64, t::Float64, _::Nothing, kappa::Float64) = Float64(x <= t) * sqrt(2.0 * kappa) * exp( - kappa * (t-x) )
kernel_OU(x::Float64, t::Float64, cp::Float64, kappa::Float64) = (cp) * Float64(x <= t) * sqrt(2.0 * kappa) * exp( - kappa * (t-x) )

# rectangular kernel
# kernel_rect(x::Float64, t::Float64, _::Nothing, kappa::Float64) = Float64(x <= t <= x + kappa)
# kernel_rect(x::Float64, t::Float64, cp::Float64, kappa::Float64) = (cp) * Float64(x <= t <= x + kappa)

"""
    KernelInt(x::Float64, t::Float64, kappa::Float64)
    KernelInt(x::Float64, t::Float64, cp::Float64, kappa::Float64)

"""
KernelInt(_::Float64, _::Float64, _::Nothing, _::Float64) = NaN
KernelInt(_::Float64, _::Float64, _::Float64, _::Float64) = NaN

# Dykstra & Laud kernel
KernelInt_DL(x::Float64, t::Float64, _::Nothing, _::Float64) = Float64(x <= t) * (t-x)
KernelInt_DL(x::Float64, t::Float64, cp::Float64, _::Float64) = (cp) * Float64(x <= t) * (t-x)

# Ornstein–Uhlenbeck kernel
KernelInt_OU(x::Float64, t::Float64, _::Nothing, kappa::Float64) = Float64(x <= t) * sqrt(2.0 / kappa) * (1.0 - exp( - kappa * (t-x) ))
KernelInt_OU(x::Float64, t::Float64, cp::Float64, kappa::Float64) = (cp) * Float64(x <= t) * sqrt(2.0 / kappa) * (1.0 - exp( - kappa * (t-x) ))

# rectangular kernel
# KernelInt_rect(x::Float64, t::Float64, _::Nothing, kappa::Float64) = Float64(x <= t <= x + kappa) * (t-x)
# KernelInt_rect(x::Float64, t::Float64, cp::Float64, kappa::Float64) = (cp) * Float64(x <= t <= x + kappa) * (t-x)

"""
    KernelInt(x::Float64, T::Vector{Float64}, _::Nothing, kappa::Float64)

"""
function KernelInt(x::Float64, T::Vector{Float64}, _::Nothing, kappa::Float64) 

    # initialize KernelInt
    KInt = 0.0

    for t in T      # loop on times

        # !! for times in descending order !!
        if t <= x break end

        # compute KernelInt
        KInt += KernelInt(x, t, nothing, kappa)

    end
    
    return KInt
    
end # KernelInt

"""
    KernelInt(x::Float64, T::Vector{Float64}, CoxProd::Vector{Float64}, kappa::Float64)

"""
function KernelInt(x::Float64, T::Vector{Float64}, CoxProd::Vector{Float64}, kappa::Float64) 

    # initialize KernelInt
    KInt = 0.0

    for (t, cp) in zip(T, CoxProd)      # loop on times

        # !! for times in descending order !!
        if t <= x break end

        # compute KernelInt
        KInt += KernelInt(x, t, cp, kappa)

    end
    
    return KInt
    
end # KernelInt

"""
    psi(u::Float64, beta::Float64, sigma::Float64; posterior::Float64)

"""
function psi(u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    
    if sigma == 0.0     # gamma CRM
        return log( 1.0 + u / (beta + posterior) )
    end

    return ( (beta + posterior + u)^sigma - (beta + posterior)^sigma ) / sigma 

end # psi

"""
    tau(u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    tau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    tau_ratio(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
tau(u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = (beta + posterior + u) ^ (sigma-1.0)
tau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = gamma(m-sigma) / gamma(1.0-sigma) * (beta + posterior + u) ^ (sigma-m)
tau_ratio(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = (m-sigma) / (beta + posterior + u)

"""
    logtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    logtau_diff(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
logtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = loggamma(m-sigma) - loggamma(1.0-sigma) - (m-sigma) * log(beta + posterior + u)
logtau_diff(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = - (m-sigma) * ( log(beta + posterior + u) - log(beta + posterior) )

"""
    sample_jump(m::Int64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
sample_jump(m::Int64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = rand(Gamma(m - sigma)) / (beta + posterior)

"""
    tail_integral(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
function tail_integral(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

    if sigma == 0.0     # gamma CRM
        return tail_integral(logheight, beta, posterior = posterior)
    end

    if logheight < -500.0   # asymptotic behaviour
        return exp( - sigma * logheight ) / ( sigma * gamma(1.0-sigma) )
    end

    return (beta + posterior)^sigma * gamma( - sigma, (beta + posterior) * exp(logheight) ) / gamma(1.0-sigma)

end # tail_integral

"""
    tail_integral(logheight::Float64, beta::Float64; posterior::Float64 = 0.0)

"""
function tail_integral(logheight::Float64, beta::Float64; posterior::Float64 = 0.0)

    if logheight < -100.0   # asymptotic behaviour
        return - eulergamma - log(beta + posterior) - logheight
    end

    return gamma( 0.0, (beta + posterior) * exp(logheight) )

end # tail_integral

"""
    tail_integral_grad(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
function tail_integral_grad(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

    if sigma == 0.0     # gamma CRM 
        return tail_integral_grad(logheight, beta, posterior = posterior)
    end

    if logheight < -100.0   # asymptotic behaviour
        return - exp( - sigma * logheight ) / gamma(1.0-sigma)
    end

    return - exp( - sigma * logheight - (beta + posterior) * exp(logheight) ) / gamma(1.0-sigma)

end # tail_integral_grad

"""
    tail_integral_grad(logheight::Float64, beta::Float64; posterior::Float64 = 0.0)

"""
function tail_integral_grad(logheight::Float64, beta::Float64; posterior::Float64 = 0.0)

    if logheight < -100.0   # asymptotic behaviour
        return - 1.0
    end

    return - exp( - (beta + posterior) * exp(logheight) )

end # tail_integral_grad

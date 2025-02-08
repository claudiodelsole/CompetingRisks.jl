"""
    kernel(x::Float64, t::Float64, eta::Float64)
    kernel(x::Float64, t::Float64, cp::Float64, eta::Float64)

"""
kernel(_::Float64, _::Float64, _::Float64) = NaN
kernel(_::Float64, _::Float64, _::Float64, _::Float64) = NaN

# Dykstra & Laud kernel
kernel_DL(x::Float64, t::Float64, _::Float64) = Float64(x <= t)
kernel_DL(x::Float64, t::Float64, cp::Float64, _::Float64) = (cp) * Float64(x <= t)

# rectangular kernel
kernel_rect(x::Float64, t::Float64, eta::Float64) = Float64(x <= t <= x + eta)
kernel_rect(x::Float64, t::Float64, cp::Float64, eta::Float64) = (cp) * Float64(x <= t <= x + eta)

# Ornstein–Uhlenbeck kernel
kernel_OU(x::Float64, t::Float64, eta::Float64) = Float64(x <= t) * sqrt(2.0 * eta) * exp( - eta * (t-x) )
kernel_OU(x::Float64, t::Float64, cp::Float64, eta::Float64) = (cp) * Float64(x <= t) * sqrt(2.0 * eta) * exp( - eta * (t-x) )

# power law kernel
# kernel_pl(x::Float64, t::Float64, eta::Float64) = Float64(x <= t) * eta / ( 1.0 + eta * (t-x) )
# kernel_pl(x::Float64, t::Float64, cp::Float64, eta::Float64) = (cp) * Float64(x <= t) * eta / ( 1.0 + eta * (t-x) )

"""
    KernelInt(x::Float64, t::Float64, eta::Float64)
    KernelInt(x::Float64, t::Float64, cp::Float64, eta::Float64)

"""
KernelInt(_::Float64, _::Float64, _::Float64) = NaN
KernelInt(_::Float64, _::Float64, _::Float64, _::Float64) = NaN

# Dykstra & Laud kernel
KernelInt_DL(x::Float64, t::Float64, _::Float64) = Float64(x <= t) * (t-x)
KernelInt_DL(x::Float64, t::Float64, cp::Float64, _::Float64) = (cp) * Float64(x <= t) * (t-x)

# rectangular kernel
KernelInt_rect(x::Float64, t::Float64, eta::Float64) = Float64(x <= t <= x + eta) * (t-x)
KernelInt_rect(x::Float64, t::Float64, cp::Float64, eta::Float64) = (cp) * Float64(x <= t <= x + eta) * (t-x)

# Ornstein–Uhlenbeck kernel
KernelInt_OU(x::Float64, t::Float64, eta::Float64) = Float64(x <= t) * sqrt(2.0 / eta) * (1.0 - exp( - eta * (t-x) ))
KernelInt_OU(x::Float64, t::Float64, cp::Float64, eta::Float64) = (cp) * Float64(x <= t) * sqrt(2.0 / eta) * (1.0 - exp( - eta * (t-x) ))

# power law kernel
# KernelInt_pl(x::Float64, t::Float64, eta::Float64) = (x <= t) ? log( 1.0 + eta * (t-x) ) : 0.0
# KernelInt_pl(x::Float64, t::Float64, cp::Float64, eta::Float64) = (x <= t) ? (cp) * log( 1.0 + eta * (t-x) ) : 0.0

"""
    KernelInt(x::Float64, T::Vector{Float64}, eta::Float64)

"""
function KernelInt(x::Float64, T::Vector{Float64}, eta::Float64) 

    # initialize KernelInt
    KInt = 0.0

    for t in T      # loop on times

        # !! for times in descending order !!
        if t <= x
            break 
        end

        # compute KernelInt
        KInt += KernelInt(x, t, eta)

    end
    
    return KInt
    
end # KernelInt

"""
    KernelInt(x::Float64, T::Vector{Float64}, CoxProd::Vector{Float64}, eta::Float64)

"""
function KernelInt(x::Float64, T::Vector{Float64}, CoxProd::Vector{Float64}, eta::Float64) 

    # initialize KernelInt
    KInt = 0.0

    for (t, cp) in zip(T, CoxProd)      # loop on times

        # !! for times in descending order !!
        if t <= x
            break 
        end

        # compute KernelInt
        KInt += KernelInt(x, t, cp, eta)

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
    logtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    tau_ratio(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    logtau_diff(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
tau(u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = (beta + posterior + u) ^ (sigma-1.0)
tau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = gamma(m-sigma) / gamma(1.0-sigma) * (beta + posterior + u) ^ (sigma-m)
logtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = loggamma(m-sigma) - loggamma(1.0-sigma) - (m-sigma) * log(beta + posterior + u)
tau_ratio(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = (m-sigma) / (beta + posterior + u)
logtau_diff(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = - (m-sigma) * ( log(beta + posterior + u) - log(beta + posterior) )

"""
    dlogtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)
    ddlogtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
dlogtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = - (m-sigma) / (beta + posterior + u)
ddlogtau(m::Int64, u::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = (m-sigma) / (beta + posterior + u)^2

"""
    sample_jump(m::Int64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
sample_jump(m::Int64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0) = rand( Gamma(m-sigma, 1.0 / (beta + posterior)) )

"""
    jumps_measure(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
function jumps_measure(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

    if sigma == 0.0     # gamma CRM

        if logheight < -500.0   # asymptotic behaviour
            return - eulergamma - log(beta + posterior) - logheight
        end
    
        return gamma( 0.0, (beta + posterior) * exp(logheight) )

    end

    if logheight < -500.0   # asymptotic behaviour
        return exp( - sigma * logheight ) / ( sigma * gamma(1.0-sigma) )
    end

    return (beta + posterior)^sigma * gamma( - sigma, (beta + posterior) * exp(logheight) ) / gamma(1.0-sigma)

end # jumps_measure

"""
    jumps_measure_grad(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

"""
function jumps_measure_grad(logheight::Float64, beta::Float64, sigma::Float64; posterior::Float64 = 0.0)

    if sigma == 0.0     # gamma CRM 

        if logheight < -500.0   # asymptotic behaviour
            return - 1.0
        end
    
        return - exp( - (beta + posterior) * exp(logheight) )

    end

    if logheight < -500.0   # asymptotic behaviour
        return - exp( - sigma * logheight ) / gamma(1.0-sigma)
    end

    return - exp( - sigma * logheight - (beta + posterior) * exp(logheight) ) / gamma(1.0-sigma)

end # jumps_measure_grad

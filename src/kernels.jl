# export structs
export DykstraLaudKernel, OrnsteinUhlenbeckKernel, RectKernel

# create abstract type
abstract type AbstractKernel end

"""
    struct DykstraLaudKernel <: AbstractKernel

Parameters of Dykstra-Laud kernel: `γ::Float64` is the kernel height.

See also [`OrnsteinUhlenbeckKernel`](@ref), [`RectKernel`](@ref).
"""
struct DykstraLaudKernel <: AbstractKernel

    # parameters
    γ::Float64      # height

end # struct

"""
    struct OrnsteinUhlenbeckKernel <: AbstractKernel

Parameters of Ornstein-Uhlenbeck kernel: `κ::Float64` is the kernel rate.

See also [`DykstraLaudKernel`](@ref), [`RectKernel`](@ref).
"""
struct OrnsteinUhlenbeckKernel <: AbstractKernel

    # parameters
    κ::Float64      # rate

end # struct

"""
    struct RectKernel <: AbstractKernel

Parameters of rectangular kernel: `γ::Float64` is the kernel height, `τ::Float64` is the bandwidth.

See also [`DykstraLaudKernel`](@ref), [`OrnsteinUhlenbeckKernel`](@ref).
"""
struct RectKernel <: AbstractKernel

    # parameters
    γ::Float64      # height
    τ::Float64      # bandwidth

end # struct

"""
    KernelType() where KernelType <: AbstractKernel

"""

# Dykstra & Laud kernel
DykstraLaudKernel() = DykstraLaudKernel(1.0)

# Ornstein–Uhlenbeck kernel
OrnsteinUhlenbeckKernel() = OrnsteinUhlenbeckKernel(1.0)

# rectangular kernel
RectKernel() = RectKernel(1.0, 1.0)

"""
    vectorize(kernelpars::KernelType) where KernelType <: AbstractKernel

"""

# Dykstra & Laud kernel
vectorize(kernelpars::DykstraLaudKernel) = [kernelpars.γ]

# Ornstein–Uhlenbeck kernel
vectorize(kernelpars::OrnsteinUhlenbeckKernel) = [kernelpars.κ]

# rectangular kernel
vectorize(kernelpars::RectKernel) = [kernelpars.γ, kernelpars.τ]

"""
    kernel(x::Float64, t::Float64, _::Nothing, kernelpars::KernelType) where KernelType <: AbstractKernel
    kernel(x::Float64, t::Float64, cp::Float64, kernelpars::KernelType) where KernelType <: AbstractKernel

"""
kernel(x::Float64, t::Float64, _::Nothing, kernelpars::KernelType) where KernelType <: AbstractKernel = kernel(x, t, kernelpars)
kernel(x::Float64, t::Float64, cp::Float64, kernelpars::KernelType) where KernelType <: AbstractKernel = (cp) * kernel(x, t, kernelpars)

"""
    KernelInt(x::Float64, t::Float64, _::Nothing, kernelpars::KernelType) where KernelType <: AbstractKernel
    KernelInt(x::Float64, t::Float64, cp::Float64, kernelpars::KernelType) where KernelType <: AbstractKernel

"""
KernelInt(x::Float64, t::Float64, _::Nothing, kernelpars::KernelType) where KernelType <: AbstractKernel = KernelInt(x, t, kernelpars)
KernelInt(x::Float64, t::Float64, cp::Float64, kernelpars::KernelType) where KernelType <: AbstractKernel = (cp) * KernelInt(x, t, kernelpars)

"""
    kernel(x::Float64, t::Float64, kernelpars::KernelType) where KernelType <: AbstractKernel
    KernelInt(x::Float64, t::Float64, kernelpars::KernelType) where KernelType <: AbstractKernel

"""

# Dykstra & Laud kernel
kernel(x::Float64, t::Float64, kernelpars::DykstraLaudKernel) = (x <= t) ? kernelpars.γ : 0.0
KernelInt(x::Float64, t::Float64, kernelpars::DykstraLaudKernel) = (x <= t) ? kernelpars.γ * (t-x) : 0.0

# Ornstein–Uhlenbeck kernel
kernel(x::Float64, t::Float64, kernelpars::OrnsteinUhlenbeckKernel) = (x <= t) ? sqrt(2.0 * kernelpars.κ) * exp( - kernelpars.κ * (t-x) ) : 0.0
KernelInt(x::Float64, t::Float64, kernelpars::OrnsteinUhlenbeckKernel) = (x <= t) ? sqrt(2.0 / kernelpars.κ) * (1.0 - exp( - kernelpars.κ * (t-x) )) : 0.0

# rectangular kernel
kernel(x::Float64, t::Float64, kernelpars::RectKernel) = (x <= t <= x + kernelpars.τ) ? kernelpars.γ : 0.0
KernelInt(x::Float64, t::Float64, kernelpars::RectKernel) = if (x <= t <= x + kernelpars.τ) return kernelpars.γ * (t-x) elseif (x + kernelpars.τ <= t) return kernelpars.γ * (kernelpars.τ - x) else return 0.0 end

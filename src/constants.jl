# export functions
export stdevs

# integration utility
const legendre = LegendreIntegral()     # quadrature nodes and weights

# relative standard deviation for resampling
const stdev_dishes = Ref(1.0)
const stdev_kernelpars = Ref(1.0)
const stdev_coefficients = Ref(1.0)

"""
    stdevs()

Set and print standard deviations in random walk Metropolis within Gibbs steps for dishes (i.e. locations), kernel parameters and regression coefficients.

# Optional arguments:
- `dishes::Float64`: standard deviation for each dish (i.e. location) in univariate random walks on logscale
- `kernelpars::Float64`: standard deviation for kernel parameters in multivariate random walk on logscale
- `coefficients::Float64`: standard deviation for each regression coefficient in univariate random walks
"""
function stdevs( ; dishes::Union{Nothing,Float64} = nothing, kernelpars::Union{Nothing,Float64} = nothing, coefficients::Union{Nothing,Float64} = nothing)

    # set stdevs
    if !isnothing(dishes) stdev_dishes[] = dishes end
    if !isnothing(kernelpars) stdev_kernelpars[] = kernelpars end
    if !isnothing(coefficients) stdev_coefficients[] = coefficients end

    # print stdevs
    println("# Stdevs in random walks")
    println("log-dishes:\t", stdev_dishes[])
    println("log-kernelpars:\t", stdev_kernelpars[])
    println("coefficients:\t", stdev_coefficients[])

end # stdevs

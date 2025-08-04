# integration utility
const legendre = LegendreIntegral()     # quadrature nodes and weights

# relative standard deviation for resampling
const stdev_dishes = Ref(1.0)
const stdev_alpha = Ref(1.0)
const stdev_kappa = Ref(1.0)
const stdev_coefficients = Ref(1.0)

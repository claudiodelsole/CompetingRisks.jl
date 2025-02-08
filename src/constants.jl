# integration utility
const legendre = LegendreIntegral()     # quadrature nodes and weights

# relative standard deviation for resampling
const mhdev_dishes = Ref(1.0)
const mhdev_alpha = Ref(1.0)
const mhdev_eta = Ref(1.0)
const mhdev_coefficients = Ref(1.0)

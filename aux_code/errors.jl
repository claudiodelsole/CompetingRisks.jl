"""
    error_survival(survival_post::Vector{Float64}, survival_true::Vector{Float64})
    
"""
function error_survival(survival_post::Vector{Float64}, survival_true::Vector{Float64})

    # survival posterior estimate absolute error
    error_post = abs.(survival_post - survival_true) 

    # maximum error
    max_error = maximum(error_post)

    return max_error

end # error_survival

"""
    error_incidence(incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64}, times::Vector{Float64})

"""
function error_incidence(incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64}, times::Vector{Float64})
    
    # true proportions
    props = [integrate_trapz(incidence_true[:,d], times) for d in axes(incidence_true, 2)]

    # incidence posterior estimates absolute error
    error_post = abs.(incidence_post - incidence_true)

    # integrated rescaled error
    int_error = [0.5 * integrate_trapz(error_post[:,d], times) / props[d] for d in axes(incidence_true, 2)]

    return int_error

end # error_incidence

"""
    error_cumincidence(cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64})

"""
function error_cumincidence(cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64})

    # true proportions
    props = cumincidence_true[end,:]

    # incidence posterior estimates absolute error
    error_post = abs.(cumincidence_post - cumincidence_true)

    # maximum rescaled error
    max_error = vec(maximum(error_post, dims = 1)) ./ props

    return max_error

end # error_cumincidence

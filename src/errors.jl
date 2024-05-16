# export functions
export survival_error, incidence_error, survival_errorplot, incidence_errorplot

"""
    survival_error(times::Vector{Float64}, survival_post::Vector{Float64}, survival_true::Vector{Float64}, density_true::Vector{Float64})
    
"""
function survival_error(times::Vector{Float64}, survival_post::Vector{Float64}, survival_true::Vector{Float64}, density_true::Vector{Float64})

    # survival posterior estimate absolute error
    error_post = abs.(survival_post - survival_true)

    # maximum error
    max_error = maximum(error_post)

    # integrated error
    int_error = integrate_trapz(density_true .* error_post, times)

    return (max_error, int_error)

end # survival_error

"""
    incidence_error(times::Vector{Float64}, incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64})

"""
function incidence_error(times::Vector{Float64}, incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64})

    # number of diseases
    diseases = 1:size(incidence_true, 2)

    # true proportions
    props = [integrate_trapz(incidence_true[:,d], times) for d in diseases]

    # incidence posterior estimates absolute error
    error_post = abs.(incidence_post - incidence_true)

    # integrated rescaled error
    int_error = 0.5 * [integrate_trapz(error_post[:,d], times) / props[d] for d in diseases]

    return int_error

end # incidence_error

"""
    incidence_error(times::Vector{Float64}, cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64}, incidence_true::Matrix{Float64})

"""
function incidence_error(times::Vector{Float64}, cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64}, incidence_true::Matrix{Float64})

    # number of diseases
    diseases = 1:size(cumincidence_true, 2)

    # true proportions
    props = cumincidence_true[end,:]

    # incidence posterior estimates absolute error
    error_post = abs.(cumincidence_post - cumincidence_true)

    # maximum rescaled error
    max_error = vec(maximum(error_post, dims = 1)) ./ props

    # integrated weighted rescaled error
    int_error = [integrate_trapz(incidence_true[:,d] .* error_post[:,d], times) / props[d]^2 for d in diseases]

    return (max_error, int_error)

end # incidence_error

"""
    survival_errorplot(times::Vector{Float64}, survival_post::Vector{Float64}, survival_true::Vector{Float64}, density_true::Vector{Float64};
            kaplan_meier::Union{Vector{Float64},Nothing} = nothing,
            lower::Union{Vector{Float64},Nothing} = nothing, upper::Union{Vector{Float64},Nothing} = nothing)

"""
function survival_errorplot(times::Vector{Float64}, survival_post::Vector{Float64}, survival_true::Vector{Float64}, density_true::Vector{Float64};
        kaplan_meier::Union{Vector{Float64},Nothing} = nothing,
        lower::Union{Vector{Float64},Nothing} = nothing, upper::Union{Vector{Float64},Nothing} = nothing)

    # initialize plot
    pl = plot(title = "Survival function errors")

    # plot survival posterior estimate error
    error_post = survival_post - survival_true
    plot!(pl, times, error_post, linecolor = 1, label = "estimated")

    # plot weighted error
    plot!(pl, times, density_true .* error_post, linecolor = 1, linestyle = :dash, primary = false)

    # plot true survival
    hline!(pl, [0.0], linecolor = 2, label = "true")

    if !isnothing(kaplan_meier)
        
        # plot Kaplan-Meier estimate error
        error_freq = kaplan_meier - survival_true
        plot!(pl, times, error_freq, linecolor = 3, label = "Kaplan-Meier")

    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, lower - survival_true, fillrange = upper - survival_true, 
                linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.3, primary = false)
    end

    # compute errors
    (max_error, int_error) = survival_error(times, survival_post, survival_true, density_true)

    # print posterior error
    println("--- Posterior survival function errors ---")
    println("Maximum error: ", string(max_error))
    println("Integrated error: ", string(int_error))

    if !isnothing(kaplan_meier)

        # compute errors
        (max_error, int_error) = survival_error(times, kaplan_meier, survival_true, density_true)

        # print frequentist error
        println()
        println("--- Kaplan-Meier survival function errors ---")
        println("Maximum error: ", string(max_error))
        println("Integrated error: ", string(int_error))
        println()

    end

    return pl

end # survival_error_plot

"""
    incidence_errorplot(times::Vector{Float64}, incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64};
            diseases::Union{Vector{Int64},Nothing} = nothing,
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function incidence_errorplot(times::Vector{Float64}, incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64};
        diseases::Union{Vector{Int64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = 1:size(incidence_true, 2)
    end

    # true proportions
    props = [integrate_trapz(incidence_true[:,d], times) for d in diseases]

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)

    # initialize plot
    pl = plot(title = "Incidence functions errors")

    # plot incidence posterior estimates error
    error_post = (incidence_post - incidence_true) ./ transpose(props)
    plot!(pl, times, error_post, linecolor = mycolors, label = mylabels)

    # plot true incidences
    hline!(pl, [0.0], linecolor = "black", linestyle = :dash, label = "true")

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, (lower - incidence_true) ./ transpose(props), fillrange = (upper - incidence_true) ./ transpose(props), 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    # compute errors
    int_error = incidence_error(times, incidence_post, incidence_true)

    # print error
    println("--- Posterior incidence functions errors ---")
    println("Integrated errors: ", string(int_error[diseases]))

    return pl

end # incidence_errorplot

"""
    incidence_errorplot(times::Vector{Float64}, cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64}, incidence_true::Matrix{Float64};
            aalen_johansen::Union{Matrix{Float64},Nothing} = nothing, diseases::Union{Vector{Int64},Nothing} = nothing,
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function incidence_errorplot(times::Vector{Float64}, cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64}, incidence_true::Matrix{Float64};
        aalen_johansen::Union{Matrix{Float64},Nothing} = nothing, diseases::Union{Vector{Int64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = 1:size(cumincidence_true, 2)
    end

    # true proportions
    props = cumincidence_true[end,:]

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)

    # initialize plot
    pl = plot(title = "Cumulative incidence functions errors")

    # plot cumincidence posterior estimates error
    error_post = (cumincidence_post - cumincidence_true) ./ transpose(props)
    plot!(pl, times, error_post, linecolor = mycolors, label = mylabels)

    # plot true incidences
    hline!(pl, [0.0], linecolor = "black", linestyle = :dash, label = "true")

    if !isnothing(aalen_johansen)
        
        # plot Aalen-Johansen estimate error
        error_freq = (aalen_johansen - cumincidence_true) ./ transpose(props)
        plot!(pl, times, error_freq, linecolor = mycolors, linestyle = :dashdot, primary = false)

    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, (lower - cumincidence_true) ./ transpose(props), fillrange = (upper - cumincidence_true) ./ transpose(props), 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    # compute errors
    (max_error, int_error) = incidence_error(times, cumincidence_post, cumincidence_true, incidence_true)

    # print error
    println("--- Posterior cumulative incidence functions errors ---")
    println("Maximum errors: ", string(max_error[diseases]))
    println("Integrated errors: ", string(int_error[diseases]))

    if !isnothing(aalen_johansen)

        # compute errors
        (max_error, int_error) = incidence_error(times, aalen_johansen, cumincidence_true, incidence_true)

        # print error
        println()
        println("--- Aalen-Johansen cumulative incidence functions errors ---")
        println("Maximum errors: ", string(max_error[diseases]))
        println("Integrated errors: ", string(int_error[diseases]))

    end

    return pl

end # incidence_errorplot

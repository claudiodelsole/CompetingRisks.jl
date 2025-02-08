# export functions
export survival_error, incidence_error, cumincidence_error, survival_errorplot, incidence_errorplot

"""
    survival_error(survival_post::Vector{Float64}, survival_true::Vector{Float64})
    
"""
function survival_error(survival_post::Vector{Float64}, survival_true::Vector{Float64})

    # survival posterior estimate absolute error
    error_post = abs.(survival_post - survival_true)

    # maximum error
    max_error = maximum(error_post)

    return max_error

end # survival_error

"""
    incidence_error(incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64}, times::Vector{Float64})

"""
function incidence_error(incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64}, times::Vector{Float64})

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
    cumincidence_error(cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64})

"""
function cumincidence_error(cumincidence_post::Matrix{Float64}, cumincidence_true::Matrix{Float64})

    # true proportions
    props = cumincidence_true[end,:]

    # incidence posterior estimates absolute error
    error_post = abs.(cumincidence_post - cumincidence_true)

    # maximum rescaled error
    max_error = vec(maximum(error_post, dims = 1)) ./ props

    return max_error

end # cumincidence_error

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
    max_error = survival_error(survival_post, survival_true)

    # print posterior error
    println("--- Survival function errors ---")
    println("BNP: ", string(max_error))

    if !isnothing(kaplan_meier)

        # compute errors
        max_error = survival_error(kaplan_meier, survival_true)

        # print frequentist error
        println("freq: ", string(max_error))

    end

    return pl

end # survival_error_plot

"""
    incidence_errorplot(times::Vector{Float64}, incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64};
            cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing,
            aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function incidence_errorplot(times::Vector{Float64}, incidence_post::Matrix{Float64}, incidence_true::Matrix{Float64};
        cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing,
        aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = 1:size(incidence_true, 2)
    end

    # true proportions
    if cum == false     # incidence estimates
        props = [integrate_trapz(incidence_true[:,d], times) for d in diseases]
    else    # cumulative incidence estimates
        props = incidence_true[end,:]
    end

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)

    # initialize plot
    if cum == false     # incidence estimates
        pl = plot(title = "Incidence functions errors")
    else    # cumulative incidence estimates
        pl = plot(title = "Cumulative incidence functions errors")
    end

    # plot incidence posterior estimates error
    error_post = (incidence_post - incidence_true) ./ transpose(props)
    plot!(pl, times, error_post, linecolor = mycolors, label = mylabels)

    # plot true incidences
    hline!(pl, [0.0], linecolor = "black", linestyle = :dash, label = "true")

    if cum == true && !isnothing(aalen_johansen)
        
        # plot Aalen-Johansen estimate error
        error_freq = (aalen_johansen - incidence_true) ./ transpose(props)
        plot!(pl, times, error_freq, linecolor = mycolors, linestyle = :dashdot, primary = false)

    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, (lower - incidence_true) ./ transpose(props), fillrange = (upper - incidence_true) ./ transpose(props), 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    # compute errors
    if cum == false     # incidence estimates
        int_error = incidence_error(incidence_post, incidence_true, times)
    else    # cumulative incidence estimates
        max_error = cumincidence_error(incidence_post, incidence_true)
    end

    # print posterior error
    if cum == false     # incidence estimates

        println("--- Incidence functions errors ---")
        println("BNP: ", string(int_error[diseases]))

    else    # cumulative incidence estimates

        println("--- Cumulative incidence functions errors ---")
        println("BNP: ", string(max_error[diseases]))

        if !isnothing(aalen_johansen)

            # compute errors
            max_error = cumincidence_error(aalen_johansen, incidence_true)
    
            # print error
            println("freq: ", string(max_error[diseases]))
    
        end

    end

    return pl

end # incidence_errorplot

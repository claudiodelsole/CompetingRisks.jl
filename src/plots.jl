# export functions
export plot_survival, plot_incidence, plot_proportions

"""
    plot_survival(times::Vector{Float64}, survival_post::Vector{Float64}; 
            survival_true::Vector{Float64} = nothing, kaplan_meier::Vector{Float64} = nothing, 
            lower::Vector{Float64} = nothing, upper::Vector{Float64} = nothing)

Plot survival functions with pointwise credible bands. 

Arguments `survival_post`, `lower` and `upper` are the outputs of [`estimate_survival`](@ref).

See also [`plot_incidence`](@ref), [`plot_proportions`](@ref).
"""
function plot_survival(times::Vector{Float64}, survival_post::Vector{Float64}; 
        survival_true::Union{Vector{Float64},Nothing} = nothing, kaplan_meier::Union{Vector{Float64},Nothing} = nothing,
        lower::Union{Vector{Float64},Nothing} = nothing, upper::Union{Vector{Float64},Nothing} = nothing)

    # plot survival posterior estimate
    pl = plot(ylim = (0.0, 1.0), xlabel = "\$t\$", ylabel = "\$S(t)\$") 
    plot!(pl, times, survival_post, linecolor = 1, label = "posterior")

    # plot true survival
    if !isnothing(survival_true)
        plot!(pl, times, survival_true, linecolor = 2, label = "true")
    end

    # plot Kaplan-Meier estimate
    if !isnothing(kaplan_meier)
        plot!(pl, times, kaplan_meier, linecolor = 3, label = "Kaplan-Meier")
    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, lower, fillrange = upper, linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.3, primary = false)
    end

    return pl

end # plot_survival

"""
    plot_incidence(times::Vector{Float64}, incidence_post::Matrix{Float64};
            cum::Bool = false, diseases::Vector{Int64} = axes(incidence_post, 2), 
            incidence_true::Matrix{Float64} = nothing, aalen_johansen::Matrix{Float64} = nothing,
            lower::Matrix{Float64} = nothing, upper::Matrix{Float64} = nothing,
            mycolors::Vector{Int64} = diseases, mylabels::Vector{String} = ["cause " * string(d) for d in diseases])

Plot incidence or cumulative incidence functions with pointwise credible bands.

Arguments `incidence_post`, `lower` and `upper` are the outputs of [`estimate_incidence`](@ref). Cumulative incidence functions need `cum = true`. Plots for a subset of diseases are obtained via the optional argument `diseases`. Diseases are arranged columnwise.

See also [`plot_survival`](@ref), [`plot_proportions`](@ref).
"""
function plot_incidence(times::Vector{Float64}, incidence_post::Matrix{Float64};
        cum::Bool = false, diseases::Vector{Int64} = Vector(axes(incidence_post, 2)), 
        incidence_true::Union{Matrix{Float64},Nothing} = nothing, aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing,
        mycolors::Vector{Int64} = diseases, mylabels::Vector{String} = ["cause " * string(d) for d in diseases])

    # labels and colors
    mycolors = reshape(mycolors, 1, :)
    mylabels = reshape(mylabels, 1, :)

    # plot incidence posterior estimates
    pl = plot(xlabel = "\$t\$", ylabel = (cum ? "\$F_\\delta(t)\$" : "\$f_\\delta(t)\$"))
    plot!(pl, times, incidence_post[:,diseases], linecolor = mycolors, label = mylabels)

    # plot true incidences
    if !isnothing(incidence_true)
        plot!(pl, times, incidence_true[:,diseases], linecolor = mycolors, linestyle = :dash, primary = false)
    end

    # plot Aalen-Johansen estimates
    if cum == true && !isnothing(aalen_johansen)
        plot!(pl, times, aalen_johansen[:,diseases], linecolor = mycolors, linestyle = :dashdot, primary = false)
    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, lower[:,diseases], fillrange = upper[:,diseases], 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    return pl

end # plot_incidence

"""
    plot_proportions(times::Vector{Float64}, proportions_post::Matrix{Float64};
            diseases::Vector{Int64} = axes(proportions_post, 2), proportions_true::Matrix{Float64} = nothing, 
            lower::Matrix{Float64} = nothing, upper::Matrix{Float64} = nothing,
            mycolors::Vector{Int64} = diseases, mylabels::Vector{String} = ["cause " * string(d) for d in diseases])

Plot prediction curves or relative hazard functions plots with pointwise credible bands.

Arguments `proportions_post`, `lower` and `upper` are the outputs of [`estimate_proportions`](@ref). Plots for a subset of diseases are obtained via the optional argument `diseases`. Diseases are arranged columnwise.

See also [`plot_survival`](@ref), [`plot_incidence`](@ref).
"""
function plot_proportions(times::Vector{Float64}, proportions_post::Matrix{Float64};
        diseases::Vector{Int64} = Vector(axes(proportions_post, 2)), proportions_true::Union{Matrix{Float64},Nothing} = nothing, 
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing,
        mycolors::Vector{Int64} = diseases, mylabels::Vector{String} = ["cause " * string(d) for d in diseases])

    # labels and colors
    mycolors = reshape(mycolors, 1, :)
    mylabels = reshape(mylabels, 1, :)

    # plot diseases proportions posterior estimates
    pl = plot(ylim = (0.0, 1.0), xlabel = "\$t\$", ylabel = "\$p_n(\\delta \\vert t)\$")
    plot!(pl, times, proportions_post[:,diseases], linecolor = mycolors, label = mylabels)

    # plot true proportions
    if !isnothing(proportions_true)
        plot!(pl, times, proportions_true[:,diseases], linecolor = mycolors, linestyle = :dash, primary = false)
    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, lower[:,diseases], fillrange = upper[:,diseases], 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    return pl

end # plot_proportions

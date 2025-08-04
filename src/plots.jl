# export functions
export plot_survival, plot_incidence, plot_proportions

"""
    plot_survival(times::Vector{Float64}, survival_post::Vector{Float64}; 
            survival_true::Union{Vector{Float64},Nothing} = nothing, kaplan_meier::Union{Vector{Float64},Nothing} = nothing,
            lower::Union{Vector{Float64},Nothing} = nothing, upper::Union{Vector{Float64},Nothing} = nothing)

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
            cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing, 
            incidence_true::Union{Matrix{Float64},Nothing} = nothing, aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function plot_incidence(times::Vector{Float64}, incidence_post::Matrix{Float64};
        cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing, 
        incidence_true::Union{Matrix{Float64},Nothing} = nothing, aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = axes(incidence_post, 2)
    end

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    # mycolors = reshape([d+1 for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)
    # mylabels = ["melanoma" "others"]
    # mylabels = ["GvHD" "death/relapse"]

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
            diseases::Union{Vector{Int64},Nothing} = nothing, 
            proportions_true::Union{Matrix{Float64},Nothing} = nothing, 
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function plot_proportions(times::Vector{Float64}, proportions_post::Matrix{Float64};
        diseases::Union{Vector{Int64},Nothing} = nothing, 
        proportions_true::Union{Matrix{Float64},Nothing} = nothing, 
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = axes(proportions_post, 2)
    end

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    # mycolors = reshape([d+1 for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)
    # mylabels = ["melanoma" "others"]
    # mylabels = ["GvHD" "death/relapse"]

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

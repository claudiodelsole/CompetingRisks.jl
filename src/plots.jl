# export functions
export summary_data, hazard_plot, survival_plot, incidence_plot, proportion_plot

"""
    summary_data(crd::CompetingRisksDataset)

"""
function summary_data(crd::CompetingRisksDataset)

    # count events by cause
    counts = [sum(crd.Delta .== d) for d in 1:crd.D]
    censored = sum(crd.Delta .== 0)

    # histogram
    plhist = plot(title = "Times to events", legend = false)
    histogram!(plhist, crd.T, normalize = :none, bins = :scott)

    # events by cause
    plcauses = plot(title = "Number of events", legend = false)
    if censored > 0
        bar!(plcauses, vcat(counts, censored), xticks = 1:crd.D, fillcolor = vcat(1:crd.D, "gray"))
    else
        bar!(plcauses, counts, xticks = 1:crd.D, fillcolor = 1:crd.D)
    end

    # combine plots
    pl = plot(plhist, plcauses, layout = (1,2))
    return pl

end # summary_data

"""
    hazard_plot(times::Vector{Float64}, hazard_post::Matrix{Float64};
            cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing, 
            hazard_true::Union{Matrix{Float64},Nothing} = nothing, hazard_prior::Union{Vector{Float64},Nothing} = nothing, 
            nelson_aalen::Union{Matrix{Float64},Nothing} = nothing,
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function hazard_plot(times::Vector{Float64}, hazard_post::Matrix{Float64}; 
        cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing, 
        hazard_true::Union{Matrix{Float64},Nothing} = nothing, hazard_prior::Union{Vector{Float64},Nothing} = nothing, 
        nelson_aalen::Union{Matrix{Float64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)
    
    # number of diseases
    if isnothing(diseases)
        diseases = 1:size(hazard_true, 2)
    end

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)

    if cum == false     # hazard estimates
        pl = plot(title = "Cause-specific hazard", legend = :topleft)
    else    # cumulative hazard estimates
        pl = plot(title = "Cause-specific cumulative hazard", legend = :topleft)
    end

    # plot hazards posterior estimates
    plot!(pl, times, hazard_post[:,diseases], linecolor = mycolors, label = mylabels)

    # plot true hazards
    if !isnothing(hazard_true)
        plot!(pl, times, hazard_true[:,diseases], linecolor = mycolors, linestyle = :dash, primary = false)
    end

    # plot hazards prior estimate
    if !isnothing(hazard_prior)
        plot!(pl, times, hazard_prior, linecolor = "gray", label = "prior")
    end

    # plot Nelson-Aalen estimates
    if cum == true && !isnothing(nelson_aalen)
        plot!(pl, times, nelson_aalen[:,diseases], linecolor = mycolors, linestyle = :dashdot, primary = false)
    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, lower[:,diseases], fillrange = upper[:,diseases], 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    return pl

end # hazard_plot

"""
    survival_plot(times::Vector{Float64}, survival_post::Vector{Float64}; 
            survival_true::Union{Vector{Float64},Nothing} = nothing, survival_prior::Union{Vector{Float64},Nothing} = nothing, 
            kaplan_meier::Union{Vector{Float64},Nothing} = nothing,
            lower::Union{Vector{Float64},Nothing} = nothing, upper::Union{Vector{Float64},Nothing} = nothing)

"""
function survival_plot(times::Vector{Float64}, survival_post::Vector{Float64}; 
        survival_true::Union{Vector{Float64},Nothing} = nothing, survival_prior::Union{Vector{Float64},Nothing} = nothing, 
        kaplan_meier::Union{Vector{Float64},Nothing} = nothing,
        lower::Union{Vector{Float64},Nothing} = nothing, upper::Union{Vector{Float64},Nothing} = nothing)

    # plot survival posterior estimate
    pl = plot(title = "Survival function", ylim = (0, 1.0))
    plot!(pl, times, survival_post, linecolor = 1, label = "posterior")

    # plot true survival
    if !isnothing(survival_true)
        plot!(pl, times, survival_true, linecolor = 2, label = "true")
    end

    # plot survival prior estimate
    if !isnothing(survival_prior)
        plot!(pl, times, survival_prior, linecolor = "gray", label = "prior")
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

end # survival_plot

"""
    incidence_plot(times::Vector{Float64}, incidence_post::Matrix{Float64};
            cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing, 
            incidence_true::Union{Matrix{Float64},Nothing} = nothing, incidence_prior::Union{Vector{Float64},Nothing} = nothing, 
            aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function incidence_plot(times::Vector{Float64}, incidence_post::Matrix{Float64};
        cum::Bool = false, diseases::Union{Vector{Int64},Nothing} = nothing, 
        incidence_true::Union{Matrix{Float64},Nothing} = nothing, incidence_prior::Union{Vector{Float64},Nothing} = nothing, 
        aalen_johansen::Union{Matrix{Float64},Nothing} = nothing,
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = 1:size(incidence_post, 2)
    end

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)

    if cum == false     # incidence estimates
        pl = plot(title = "Cause-specific incidence function")
    else    # cumulative incidence estimates
        pl = plot(title = "Cause-specific cumulative incidence function", legend = :topleft)
    end

    # plot incidence posterior estimates
    plot!(pl, times, incidence_post[:,diseases], linecolor = mycolors, label = mylabels)

    # plot true incidences
    if !isnothing(incidence_true)
        plot!(pl, times, incidence_true[:,diseases], linecolor = mycolors, linestyle = :dash, primary = false)
    end

    # plot incidence prior estimate
    if !isnothing(incidence_prior)
        plot!(pl, times, incidence_prior, linecolor = "gray", label = "prior")
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

end # incidence_plot

"""
    proportion_plot(times::Vector{Float64}, proportion_post::Matrix{Float64};
            diseases::Union{Vector{Int64},Nothing} = nothing, 
            proportion_true::Union{Matrix{Float64},Nothing} = nothing, 
            lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

"""
function proportion_plot(times::Vector{Float64}, proportion_post::Matrix{Float64};
        diseases::Union{Vector{Int64},Nothing} = nothing, 
        proportion_true::Union{Matrix{Float64},Nothing} = nothing, 
        lower::Union{Matrix{Float64},Nothing} = nothing, upper::Union{Matrix{Float64},Nothing} = nothing)

    # number of diseases
    if isnothing(diseases)
        diseases = 1:size(proportion_post, 2)
    end

    # labels and colors
    mycolors = reshape([d for d in diseases], 1, :)
    mylabels = reshape(["cause " * string(d) for d in diseases], 1, :)

    # plot diseases proportions posterior estimates
    pl = plot(title = "Diseases proportions function", ylim = (0, 1.0))
    plot!(pl, times, proportion_post[:,diseases], linecolor = mycolors, label = mylabels)

    # plot true proportions
    if !isnothing(proportion_true)
        plot!(pl, times, proportion_true[:,diseases], linecolor = mycolors, linestyle = :dash, primary = false)
    end

    # fill credible bands
    if !isnothing(lower) && !isnothing(upper)
        plot!(pl, times, lower[:,diseases], fillrange = upper[:,diseases], 
                linecolor = mycolors, linealpha = 0.0, fillcolor = mycolors, fillalpha = 0.3, primary = false)
    end

    return pl

end # proportion_plot

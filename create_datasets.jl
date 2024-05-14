# imports
import Distributions: UnivariateDistribution, DiscreteUniform

"""
    independent_dataset(N::Int64, models::Vector{Type}; 
            censoring::Union{TypeC,Nothing} = nothing) where {Type <: UnivariateDistribution, TypeC <: UnivariateDistribution}

"""
function independent_dataset(N::Int64, models::Vector{Type}; 
        censoring::Union{TypeC,Nothing} = nothing) where {Type <: UnivariateDistribution, TypeC <: UnivariateDistribution}

    # initialize observations
    T = zeros(Float64, N)
    Delta = zeros(Int64, N)

    # initialize time-to-event
    D = length(models)
    times_to_event = zeros(Float64, N, D)
    censoring_times = zeros(Float64, N)

    # sample independent data
    for cause in eachindex(models)
        times_to_event[:,cause] = rand(models[cause], N)
        if !isnothing(censoring) censoring_times = rand(censoring, N) end
    end

    # competing risks
    for i in 1:N

        # find minimum time-to-event
        (T[i], Delta[i]) = findmin(times_to_event[i,:])

        # censoring
        if !isnothing(censoring) && censoring_times[i] < T[i]
            (T[i], Delta[i]) = (censoring_times[i], 0)
        end

    end

    # create CompetingRisksDataset
    return CompetingRisksDataset(T, Delta)

end # independent_dataset

"""
    independent_dataset(N::Int64, models::Vector{Type}, L::Int64; 
            censoring::Union{TypeC,Nothing} = nothing) where {Type <: UnivariateDistribution, TypeC <: UnivariateDistribution}

"""
function independent_dataset(N::Int64, models::Vector{Type}, L::Int64; 
        censoring::Union{TypeC,Nothing} = nothing) where {Type <: UnivariateDistribution, TypeC <: UnivariateDistribution}

    # initialize observations
    T = zeros(Float64, N)
    Delta = zeros(Int64, N)

    # initialize time-to-event
    D = length(models)
    times_to_event = zeros(Float64, N, D)
    censoring_times = zeros(Float64, N)

    # sample independent data
    for cause in eachindex(models)
        times_to_event[:,cause] = rand(models[cause], N)
        if !isnothing(censoring) censoring_times = rand(censoring, N) end
    end

    # competing risks
    for i in 1:N
        # find minimum time-to-event
        (T[i], Delta[i]) = findmin(times_to_event[i,:])

        # censoring
        if !isnothing(censoring) && censoring_times[i] < T[i]
            (T[i], Delta[i]) = (censoring_times[i], 0)
        end

    end

    # random categorical predictor
    predictors = rand(DiscreteUniform(L), N)

    # create CompetingRisksDataset
    return CompetingRisksDataset(T, Delta, predictors)

end # independent_dataset

"""
    summary_models(models::Vector{Type}) where Type <: UnivariateDistribution

"""
function summary_models(models::Vector{Type}) where Type <: UnivariateDistribution

    # labels and colors
    mycolors = reshape(1:length(models), 1, :)
    mylabels = reshape(["cause " * string(d) for d in 1:length(models)], 1, :)

    # times vector
    upper_time = maximum([quantile(model, 0.9) for model in models])
    step = upper_time/1000
    times = Vector{Float64}(0.0:step:upper_time)

    # true hazard functions
    hazard_true = [hazard(model, t) for t in times, model in models]

    # true incidence functions
    incidence_true = [hazard(model, t) * prod([survival(model, t) for model in models]) for t in times, model in models]

    # hazard functions
    plhazard = plot(title = "Hazard functions", legend = :bottomright)
    plot!(plhazard, times, hazard_true, linecolor = mycolors, label = mylabels)
    plot!()

    # incidence functions
    plincidence = plot(title = "Incidence functions", legend = :topright)
    plot!(plincidence, times, incidence_true, linecolor = mycolors, label = mylabels)

    # combine plots
    pl = plot(plhazard, plincidence, layout = (1,2), size = (720,480))
    return pl

end # summary_models

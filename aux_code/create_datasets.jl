# import from Distributions
import Distributions: UnivariateDistribution
import DataFrames: DataFrame

"""
    independent_dataset(N::Int64, models::Vector{Type}; L::Int64 = 0, censoring::Union{TypeC,Nothing} = nothing) where {Type <: UnivariateDistribution, TypeC <: UnivariateDistribution}

"""
function independent_dataset(N::Int64, models::Vector{Type}; L::Int64 = 0, censoring::Union{TypeC,Nothing} = nothing) where {Type <: UnivariateDistribution, TypeC <: UnivariateDistribution}

    # initialize observations
    T = zeros(Float64, N)
    Delta = zeros(Int64, N)
    predictor = zeros(Int64, N)

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
    for i in eachindex(T)

        # find minimum time-to-event
        (T[i], Delta[i]) = findmin(times_to_event[i,:])

        # censoring
        if !isnothing(censoring) && censoring_times[i] < T[i]
            (T[i], Delta[i]) = (censoring_times[i], 0)
        end

    end

    # random categorical predictor
    if L > 0 predictor = rand(DiscreteUniform(L), N) end

    # create DataFrame
    return DataFrame(T = T, Delta = Delta, predictor = predictor)

end # independent_dataset

"""
    summary_models(models::Vector{Type}) where Type <: UnivariateDistribution

"""
function summary_models(models::Vector{Type}) where Type <: UnivariateDistribution

    # labels and colors
    mycolors = reshape(range(1, length(models)), 1, :)
    mylabels = reshape(["cause " * string(d) for d in range(1, length(models))], 1, :)

    # times vector
    upper_time = maximum([quantile(model, 0.9) for model in models])
    step = upper_time/1000
    times = Vector{Float64}(0.0:step:upper_time)

    # true incidence functions
    incidence_true = [hazard(model, t) * prod([survival(model, t) for model in models]) for t in times, model in models]

    # true proportions
    proportion_true = [hazard(model, t) / sum([hazard(model, t) for model in models]) for t in times, model in models]

    # incidence functions
    plincidence = plot(legend = :topright, xlabel = "\$t\$", ylabel = "\$f_δ(t)\$")
    plot!(plincidence, times, incidence_true, linecolor = mycolors, label = mylabels)

    # proportions
    plprop = plot(legend = :topright, xlabel = "\$t\$", ylabel = "\$p_n(δ \\vert t)\$")
    plot!(plprop, times, proportion_true, linecolor = mycolors, label = mylabels)

    # combine plots
    pl = plot(plincidence, plprop, layout = (1,2), size = (640,360))
    return pl

end # summary_models

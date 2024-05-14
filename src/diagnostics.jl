# import from Statistics, StatsBase, Plots
import Statistics: mean
import StatsBase: autocor
import Plots: plot, plot!, hline!, histogram!

# export struct and functions
export Diagnostics, summary_dishes, summary_theta, summary_alpha, summary_coefficients, summary_loglikelihood, traceplots

"""
    struct Diagnostics

"""
struct Diagnostics

    # latent structure
    dishes_number::Vector{Int64}
    accept_dishes::Vector{Float64}

    # hyperparameters
    theta::Vector{Float64}    
    logalpha::Vector{Float64}
    logeta::Vector{Float64}

    # acceptance probability
    accept_alpha::Vector{Float64}
    accept_eta::Vector{Float64}

    # regression coefficients
    xi::Vector{Float64}
    accept_coeffs::Vector{Float64}

    # loglikelihood
    loglik::Vector{Float64}

    # explicit constructor
    function Diagnostics()

        # initialize latent structure vector
        dishes_number = Array{Int64}(undef, 0)
        accept_dishes = Array{Float64}(undef, 0)

        # hyperparameters
        theta = Array{Float64}(undef, 0)
        logalpha = Array{Float64}(undef, 0)
        logeta = Array{Float64}(undef, 0)

        # acceptance probability
        accept_alpha = Array{Float64}(undef, 0)
        accept_eta = Array{Float64}(undef, 0)

        # regression coefficients
        xi = Array{Float64}(undef, 0)
        accept_coeffs = Array{Float64}(undef, 0)

        # loglikelihood
        loglik = Array{Float64}(undef, 0)

        # create Diagnostics
        return new(dishes_number, accept_dishes, theta, logalpha, logeta, accept_alpha, accept_eta, xi, accept_coeffs, loglik)

    end # Diagnostics

end # struct

"""
    append(dgn::Diagnostics, rf::Union{RestaurantFranchise,RestaurantArray}, 
            accept_dishes::Float64, accept_alpha::Float64, accept_eta::Float64)

"""
function append(dgn::Diagnostics, rf::Union{RestaurantFranchise,RestaurantArray}, 
        accept_dishes::Float64, accept_alpha::Float64, accept_eta::Float64)

    # dishes number
    dishes_number = sum(rf.n .> 0)

    # append dishes number and acceptance probability
    append!(dgn.dishes_number, dishes_number)
    append!(dgn.accept_dishes, accept_dishes / dishes_number)

    # append hyperparameters
    append!(dgn.theta, rf.theta)
    append!(dgn.logalpha, log(rf.alpha))
    append!(dgn.logeta, log(rf.eta))

    # append acceptance probability
    append!(dgn.accept_alpha, accept_alpha)
    append!(dgn.accept_eta, accept_eta)

    # append loglikelihood
    append!(dgn.loglik, loglikelihood(rf))

end # append

"""
    append(dgn::Diagnostics, rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, 
            accept_dishes::Float64, accept_alpha::Float64, accept_eta::Float64, accept_coeffs::Vector{Float64})

"""
function append(dgn::Diagnostics, rf::Union{RestaurantFranchise,RestaurantArray}, cm::CoxModel, 
        accept_dishes::Float64, accept_alpha::Float64, accept_eta::Float64, accept_coeffs::Vector{Float64})

    # dishes number
    dishes_number = sum(rf.n .> 0)

    # append dishes number and acceptance probability
    append!(dgn.dishes_number, dishes_number)
    append!(dgn.accept_dishes, accept_dishes / dishes_number)

    # append hyperparameters
    append!(dgn.theta, rf.theta)
    append!(dgn.logalpha, log(rf.alpha))
    append!(dgn.logeta, log(rf.eta))

    # append acceptance probability
    append!(dgn.accept_alpha, accept_alpha)
    append!(dgn.accept_eta, accept_eta)

    # append coefficients and acceptance probability
    append!(dgn.xi, cm.xi)
    append!(dgn.accept_coeffs, accept_coeffs)

    # append loglikelihood
    append!(dgn.loglik, loglikelihood(rf))

end # append

"""
    summary_dishes(dgn::Diagnostics, burn_in::Int64)

"""
function summary_dishes(dgn::Diagnostics, burn_in::Int64)

    # retrieve trace
    trace = dgn.dishes_number
    trace_start = trace[begin:burn_in]
    trace_post = trace[burn_in+1:end]

    # acceptance probabilities
    accept = dgn.accept_dishes[burn_in+1:end]

    # print output
    println("--- Dishes number ---")
    println("Posterior mean: ", string(mean(trace_post)))
    println("Acceptance probability: ", string(mean(accept)))

    # traceplot
    pltrace = plot(title = "Traceplot", legend = false)
    plot!(pltrace, 1:burn_in, trace_start, linecolor = "gray")
    plot!(pltrace, burn_in+1:length(trace), trace_post, linecolor = 1)
    hline!(pltrace, [mean(trace_post)], linecolor = 2)

    # histogram
    plhist = plot(title = "Posterior distribution", legend = false)
    histogram!(plhist, trace_post, normalize = :probability)

    # combine plots
    pl = plot(pltrace, plhist, layout = (1,2), size = (720,480))
    return pl

end # summary_dishes

"""
    summary_theta(dgn::Diagnostics, burn_in::Int64)

"""
function summary_theta(dgn::Diagnostics, burn_in::Int64)

    # retrieve trace
    trace = dgn.theta
    trace_start = trace[begin:burn_in]
    trace_post = trace[burn_in+1:end]

    # print output
    println("--- Base measure mass ---")
    println("Posterior mean: ", string(mean(trace_post)))

    # traceplot
    pltrace = plot(title = "Traceplot", legend = false)
    plot!(pltrace, 1:burn_in, trace_start, linecolor = "gray")
    plot!(pltrace, burn_in+1:length(trace), trace_post, linecolor = 1)
    hline!(pltrace, [mean(trace_post)], linecolor = 2)

    # histogram
    plhist = plot(title = "Posterior distribution", legend = false)
    histogram!(plhist, trace_post, normalize = :pdf)

    # combine plots
    pl = plot(pltrace, plhist, layout = (1,2), size = (720,480))
    return pl

end # summary_theta

"""
    summary_alpha(dgn::Diagnostics, burn_in::Int64)

"""
function summary_alpha(dgn::Diagnostics, burn_in::Int64)

    # retrieve trace
    trace = dgn.logalpha
    trace_start = trace[begin:burn_in]
    trace_post = trace[burn_in+1:end]

    # acceptance probability
    accept = dgn.accept_alpha[burn_in+1:end]

    # print output
    println("--- Kernel parameter ---")
    println("Posterior mean: ", string(exp(mean(trace_post))))
    println("Acceptance rate: ", string(mean(accept)))

    # traceplot
    pltrace = plot(title = "Traceplot", legend = false)
    plot!(pltrace, 1:burn_in, trace_start, linecolor = "gray")
    plot!(pltrace, burn_in+1:length(trace), trace_post, linecolor = 1)
    hline!(pltrace, [mean(trace_post)], linecolor = 2)

    # histogram
    plhist = plot(title = "Posterior distribution", legend = false)
    histogram!(plhist, trace_post, normalize = :pdf)

    # combine plots
    pl = plot(pltrace, plhist, layout = (1,2), size = (720,480))
    return pl

end # summary_alpha

"""
    summary_eta(dgn::Diagnostics, burn_in::Int64)

"""
function summary_eta(dgn::Diagnostics, burn_in::Int64)

    # retrieve trace
    trace = dgn.logeta
    trace_start = trace[begin:burn_in]
    trace_post = trace[burn_in+1:end]

    # acceptance probability
    accept = dgn.accept_eta[burn_in+1:end]

    # print output
    println("--- Kernel parameter ---")
    println("Posterior mean: ", string(exp(mean(trace_post))))
    println("Acceptance rate: ", string(mean(accept)))

    # traceplot
    pltrace = plot(title = "Traceplot", legend = false)
    plot!(pltrace, 1:burn_in, trace_start, linecolor = "gray")
    plot!(pltrace, burn_in+1:length(trace), trace_post, linecolor = 1)
    hline!(pltrace, [mean(trace_post)], linecolor = 2)

    # histogram
    plhist = plot(title = "Posterior distribution", legend = false)
    histogram!(plhist, trace_post, normalize = :pdf)

    # combine plots
    pl = plot(pltrace, plhist, layout = (1,2), size = (720,480))
    return pl

end # summary_eta

"""
    summary_coefficient(dgn::Diagnostics, burn_in::Int64; l::Int64 = 1)

"""
function summary_coefficients(dgn::Diagnostics, burn_in::Int64; l::Int64 = 1)

    # retrieve traces
    trace = reshape(dgn.xi, length(dgn.accept_alpha), :)
    trace_start = trace[begin:burn_in, l]
    trace_post = trace[burn_in+1:end, l]

    # acceptance probability
    accept = reshape(dgn.accept_coeffs, length(dgn.accept_alpha), :)
    accept = dgn.accept_coeffs[burn_in+1:end, l]

    # print output
    println("--- Regression coefficient: Group ", string(l), " ---")
    println("Posterior mean: ", string(mean(trace_post)))
    println("Acceptance rate: ", string(mean(accept)))

    # traceplot
    pltrace = plot(title = "Traceplot", legend = false)
    plot!(pltrace, 1:burn_in, trace_start, linecolor = "gray")
    plot!(pltrace, burn_in+1:length(trace), trace_post, linecolor = l)
    hline!(pltrace, [mean(trace_post)], linecolor = l+1)

    # histogram
    plhist = plot(title = "Posterior distribution", legend = false)
    histogram!(plhist, trace_post, normalize = :pdf)

    # combine plots
    pl = plot(pltrace, plhist, layout = (1,2), size = (720,480))
    return pl

end # summary_coefficient

"""
    summary_loglikelihood(dgn::Diagnostics, burn_in::Int64)

"""
function summary_loglikelihood(dgn::Diagnostics, burn_in::Int64)

    # retrieve trace
    trace = dgn.loglik
    trace_start = trace[begin:burn_in]
    trace_post = trace[burn_in+1:end]

    # traceplot
    pltrace = plot(title = "Log-Likelihood", legend = false)
    plot!(pltrace, 1:burn_in, trace_start, linecolor = "gray")
    plot!(pltrace, burn_in+1:length(trace), trace_post, linecolor = 1)

    # combine plots
    pl = plot(pltrace, size = (720,480))
    return pl

end # summary_loglikelihood

"""
    ess(trace::Vector{Float64})

"""
ess(trace::Vector{Float64}) = length(trace) / (2 * sum(autocor(trace)) - 1)

"""
    traceplots(estimator::HazardMarginal, time::Float64; l::Int64 = 0)

"""
function traceplots(estimator::HazardMarginal, time::Float64; l::Int64 = 0)

    # labels and colors
    mycolors = reshape([d for d in 1:estimator.D], 1, :)
    mylabels = reshape(["cause " * string(d) for d in 1:estimator.D], 1, :)

    # times vector length
    num_times = length(estimator.times)

    # reshape samples
    post_samples = reshape(estimator.post_samples, num_times, estimator.L + 1, estimator.D, :)

    # retrieve trace
    t = sum(time .>= estimator.times)
    trace = transpose(post_samples[t,l+1,:,:])

    # traceplot
    pltrace = plot(title = "Traceplots")
    plot!(pltrace, trace, linecolor = mycolors, label = mylabels)

    # autocorrelation plot
    plcor = plot(title = "Autocorrelations")
    plot!(plcor, autocor(trace), ylims = (-0.05, 1.0), linecolor = mycolors, label = mylabels)
    hline!(plcor, [0.0], linecolor = "black", linestyle = :dash, label = false)

    # print output
    println("ESS: ", string([ess(trace[:,d]) for d in 1:estimator.D]))

    # combine plots
    pl = plot(pltrace, plcor, layout = (1,2), size = (720,480))
    return pl

end # traceplots

"""
    traceplots(estimator::SurvivalMarginal, time::Float64; l::Int64 = 0)

"""
function traceplots(estimator::SurvivalMarginal, time::Float64; l::Int64 = 0)

    # times vector length
    num_times = length(estimator.times)

    # reshape samples
    post_samples = reshape(estimator.post_samples, num_times, estimator.L + 1, :)

    # retrieve trace
    t = sum(time .>= estimator.times)
    trace = post_samples[t,l+1,:]

    # traceplot
    pltrace = plot(title = "Traceplot", legend = false)
    plot!(pltrace, trace, linecolor = 1)

    # autocorrelation plot
    plcor = plot(title = "Autocorrelation", legend = false)
    plot!(plcor, autocor(trace), ylims = (-0.05, 1.0), linecolor = 2)
    hline!(plcor, [0.0], linecolor = "black", linestyle = :dash)

    # print output
    println("ESS: ", string(ess(trace)))

    # combine plots
    pl = plot(pltrace, plcor, layout = (1,2), size = (720,480))
    return pl

end # traceplots

"""
    traceplots(hazard_estimator::HazardMarginal, survival_estimator::SurvivalMarginal, time::Float64; l::Int64 = 0)

"""
function traceplots(hazard_estimator::HazardMarginal, survival_estimator::SurvivalMarginal, time::Float64; l::Int64 = 0)

    # labels and colors
    mycolors = reshape([d for d in 1:hazard_estimator.D], 1, :)
    mylabels = reshape(["cause " * string(d) for d in 1:hazard_estimator.D], 1, :)

    # reshape samples
    post_samples = reshape_samples(hazard_estimator, survival_estimator)

    # retrieve trace
    t = sum(time .>= hazard_estimator.times)
    trace = transpose(post_samples[t,l+1,:,:])

    # traceplot
    pltrace = plot(title = "Traceplots")
    plot!(pltrace, trace, linecolor = mycolors, label = mylabels)

    # autocorrelation plot
    plcor = plot(title = "Autocorrelations")
    plot!(plcor, autocor(trace), ylims = (-0.05, 1.0), linecolor = mycolors, label = mylabels)
    hline!(plcor, [0.0], linecolor = "black", linestyle = :dash, label = false)

    # print output
    println("ESS: ", string([ess(trace[:,d]) for d in 1:hazard_estimator.D]))

    # combine plots
    pl = plot(pltrace, plcor, layout = (1,2), size = (720,480))
    return pl

end # traceplots

"""
    loglikelihood(rf::RestaurantFranchise)

"""
function loglikelihood(rf::RestaurantFranchise)

    # initialize loglikelihood
    loglik = 0.0

    # loop on dishes
    for (dish, rdish) in enumerate(rf.r)

        if rdish == 0       # no tables at dish
            continue
        end

        # retrieve indices
        customers = (rf.X .== dish)         # customers eating dish
        tables = (rf.table_dish .== dish)   # tables eating dish

        # retrieve dish value
        dish_value = rf.Xstar[dish]

        # precompute KernelInt
        KInt = rf.alpha * rf.KInt[dish]

        # compute loglikelihood
        if isnothing(rf.CoxProd)     # exchangeable model
            loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.eta)))
        else    # regression model
            loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.CoxProd[customers], rf.eta)))
        end

        # compute loglikelihood
        loglik += sum(logtau.(rf.q[tables], KInt, rf.beta, rf.sigma))
        loglik += logtau(rdish, rf.D * psi(KInt, rf.beta, rf.sigma), rf.beta0, rf.sigma0) 
        loglik += log(rf.theta) + pdf(rf.base_measure, dish_value)

    end

    if isnothing(rf.CoxProd)     # exchangeable model

        # integrand function
        function f(x::Float64)
            
            # compute integrand
            return psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        function g(x::Float64)
            
            # compute integrand
            return psi(rf.D * psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta), rf.beta, rf.sigma), rf.beta0, rf.sigma0) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    return loglik

end # loglikelihood

"""
    loglikelihood(rf::RestaurantArray)

"""
function loglikelihood(rf::RestaurantArray)

    # initialize loglikelihood
    loglik = 0.0

    # loop on tables
    for (table, ntable) in enumerate(rf.n)

        if ntable == 0      # no customers at dish
            continue
        end

        # retrieve indices
        customers = (rf.X .== table)         # customers seated at table

        # retrieve dish value
        dish_value = rf.Xstar[table]

        # compute loglikelihood
        if isnothing(rf.CoxProd)     # exchangeable model
            loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.eta)))
        else    # regression model
            loglik += sum(log.(rf.alpha * kernel.(dish_value, rf.T[customers], rf.CoxProd[customers], rf.eta)))
        end

        # compute loglikelihood
        loglik += logtau(ntable, rf.alpha * rf.KInt[table], rf.beta, rf.sigma)
        loglik += log(rf.theta) + pdf(rf.base_measure, dish_value)

    end

    if isnothing(rf.CoxProd)     # exchangeable model

        # integrand function
        function f(x::Float64)
            
            # compute integrand
            return psi(rf.alpha * KernelInt(x, rf.T, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * rf.D * integrate(f, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    else    # regression model

        # integrand function
        function g(x::Float64)
            
            # compute integrand
            return psi(rf.alpha * KernelInt(x, rf.T, rf.CoxProd, rf.eta), rf.beta, rf.sigma) * pdf(rf.base_measure, x)

        end

        # compute loglikelihood
        loglik -= rf.theta * rf.D * integrate(g, rf.legendre; lower = 0.0, upper = maximum(rf.T))

    end

    return loglik

end # loglikelihood

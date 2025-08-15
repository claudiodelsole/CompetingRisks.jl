# load module
using CompetingRisks

# imports
import Random: seed!
import DataFrames: DataFrame
import StatsBase: counts
import CSV: CSV
import Plots: plot, plot!, vline!, savefig

##########
# Melanoma survival dataset
##########

# set seed for reproducibility
seed!(42)

# include auxiliary files
include("../aux_code/functions.jl")
include("../aux_code/rbart.jl")

# load dataset
data = CSV.File("data/melanoma.txt")

# retrieve observed variables
T = Float64.(data.days)         # observations, time-to-event
Delta = data.status             # event causes
predictor = data.sex            # categorical predictors

# map event causes codes
Delta[Delta.==2] .= 0       # censored observations
Delta[Delta.==3] .= 2       # deaths from other causes

# convert days to years
T = T./365.25

# create synthetic dataset
data = DataFrame(T = T, Delta = Delta, predictor = predictor)

# data summary
println("# Data summary")
println("Subjects per competing event: ", counts(data.Delta, 2))
println("Censored observations: ", count(data.Delta .== 0))

# times vector
# upper_time = quantile(data.T, 0.99)
upper_time = 12.0
times = collect(0.0:0.05:upper_time)

##########
# Setup R
##########

# setup RCall
using RCall

# import required libraries
R"library(survival)"
R"library(cmprsk)"

# import data
@rput data
@rput times

##########
# Gibbs sampling algorithm
##########

# create CompetingRisksModel
# cmprsk = CompetingRisksModel(OrnsteinUhlenbeckKernel)
cmprsk = CompetingRisksModel(OrnsteinUhlenbeckKernel, sigma = 0.25, sigma0 = 0.25, regression = true)

# setup acceptance rates
stdevs(dishes = 0.5, kernelpars = 2.0)

# run chain
marginal_estimator, conditional_estimator, params = posterior_sampling(data, cmprsk, nsamples = 2000, times = times, burn_in = 5000)

##########
# Diagnostics
##########

println("# Model hyperparameters")

# number of dishes
plot(summary_dishes(params, burn_in = 5000)[2], size = (480,360), xlim = (0.0, 13.0))
savefig("figures_supp/melanoma_dishes.svg")

# base measure mass
plot(summary_theta(params, burn_in = 5000)[2], size = (480,360), xlim = (0.0, 0.4))
savefig("figures_supp/melanoma_theta.svg")

# kernel shape parameter
plot(summary_kernelpars(params, :κ, burn_in = 5000)[2], size = (480,360), xlim = (-8.0, 1.0))
savefig("figures_supp/melanoma_kappa.svg")

##########
# Posterior estimates: hazard rate ratio
##########

# frequentist estimate - Cox proportional hazard model
R"coxfit <- coxph(Surv(T, Delta > 0) ~ predictor, data = data)"
R"summary(coxfit)"

# regression coefficient
(pltrace, hist) = summary_coefficients(params, 1, burn_in = 5000)
# plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (-0.5, 1.5))
plot(hist, size = (480,360), xlim = (-0.5, 1.5))
savefig("figures_supp/melanoma_coeffs.svg")

# estimate hazard ratio
# (pltrace, hist) = summary_coefficients(params, 1, burn_in = 5000, hazard_ratio = true)
# plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (0.5, 4.0))
# plot(hist, size = (480,360), xlim = (0.5, 4.0))

##########
# Posterior estimates: survival function
##########

# frequentist estimate
R"fit <- survfit(coxfit, newdata = data.frame(predictor = c(0,1)))"
R"km <- summary(fit, times)$surv"

# retrieve estimate
survival_freq = rcopy(R"km")

# posterior estimate
(survival_post, _, _) = estimate_survival(marginal_estimator)
(_, survival_lower, survival_upper) = estimate_survival(conditional_estimator)

# plots for females
plot_survival(times, survival_post[:,1], kaplan_meier = survival_freq[:,1], lower = survival_lower[:,1], upper = survival_upper[:,1])
plot!(xlim = (0.0,8.5), size = (480,360), xlabel = "\$t\$ (years)")
savefig("figures_supp/melanoma_survival_female.svg")

# plots for males
plot_survival(times, survival_post[:,2], kaplan_meier = survival_freq[:,2], lower = survival_lower[:,2], upper = survival_upper[:,2])
plot!(xlim = (0.0,8.5), size = (480,360), xlabel = "\$t\$ (years)")
savefig("figures_supp/melanoma_survival_male.svg")

##########
# Posterior estimates: cumulative incidence functions
##########

# frequentist estimate for primary cause (melanoma)
R"finegray <- crr(data$T, data$Delta, data$predictor)"
R"cif.melanoma <- predict(finegray, cov1 = as.matrix(data.frame(predictor = c(0,1))))"

# frequentist estimate for competing cause (other)
R"finegray <- crr(data$T, data$Delta, data$predictor, failcode = 2)"
R"cif.other <- predict(finegray, cov1 = as.matrix(data.frame(predictor = c(0,1))))"

# retrieve estimates
cumincidence_freq = cat(mapslices(values -> timepoints(rcopy(R"cif.melanoma[,1]"), values, times), rcopy(R"cif.melanoma[,c(2,3)]"), dims = 1), mapslices(values -> timepoints(rcopy(R"cif.other[,1]"), values, times), rcopy(R"cif.other[,c(2,3)]"), dims = 1), dims = 3)

# posterior estimates
(cumincidence_post, _, _) = estimate_incidence(marginal_estimator, cum = true)
(_, cumincidence_lower, cumincidence_upper) = estimate_incidence(conditional_estimator, cum = true)

# plots for females
plot_incidence(times, cumincidence_post[:,1,:], cum = true, aalen_johansen = cumincidence_freq[:,1,:], lower = cumincidence_lower[:,1,:], upper = cumincidence_upper[:,1,:], mycolors = [2, 3], mylabels = ["melanoma", "others"])
plot!(size = (480,360), xlim = (0.0,8.5), ylim = (0.0,0.6), xlabel = "\$t\$ (years)", legend = :topleft)
savefig("figures_supp/melanoma_cumincidence_female.svg")

# plots for males
plot_incidence(times, cumincidence_post[:,2,:], cum = true, aalen_johansen = cumincidence_freq[:,2,:], lower = cumincidence_lower[:,2,:], upper = cumincidence_upper[:,2,:], mycolors = [2, 3], mylabels = ["melanoma", "others"])
plot!(size = (480,360), xlim = (0.0,8.5), ylim = (0.0,0.6), xlabel = "\$t\$ (years)", legend = :topleft)
savefig("figures_supp/melanoma_cumincidence_male.svg")

##########
# Posterior estimates: prediction curves
##########

# posterior estimates
(proportions_post, proportions_lower, proportions_upper) = estimate_proportions(marginal_estimator)

# plots
plot_proportions(times, proportions_post, lower = proportions_lower, upper = proportions_upper, mycolors = [2, 3], mylabels = ["melanoma", "others"])
plot!(size = (480,360), xlim = (0.0,8.5), xlabel = "\$t\$ (years)")
vline!([maximum(data.T[data.Delta.!=0])], linestyle = :dashdot, linecolor = :black, linealpha = 0.5, label = false)
savefig("figures/melanoma_prediction.svg")

##########
# BART for competing risks
##########

# import BART library
R"library(BART)"

# discretize times
R"data$T <- pmax(1, round(data$T * 52)) / 52"

# BART competing risks model
R"post <- crisk.bart(x.train = data$predictor, times = data$T, delta = data$Delta, x.test = matrix(c(0, 1), nrow = 2, ncol = 1))"

##########
# Posterior estimates: survival function
##########

# posterior estimate
(survival_post_bart, survival_lower_bart, survival_upper_bart) = estimate_survival_bart(rcopy(R"post$times"), rcopy(R"post$surv.test"), times)

# plots for females
plot_survival(times, survival_post_bart[:,1], kaplan_meier = survival_freq[:,1], lower = survival_lower_bart[:,1], upper = survival_upper_bart[:,1])
plot!(xlim = (0.0, 8.5), size = (480,360), xlabel = "\$t\$ (years)")

# plots for males
plot_survival(times, survival_post_bart[:,2], kaplan_meier = survival_freq[:,2], lower = survival_lower_bart[:,2], upper = survival_upper_bart[:,2])
plot!(xlim = (0.0, 8.5), size = (480,360), xlabel = "\$t\$ (years)")

##########
# Posterior estimates: cumulative incidence functions
##########

# posterior estimates
(cumincidence_post_bart, cumincidence_lower_bart, cumincidence_upper_bart) = estimate_incidence_bart(rcopy(R"post$times"), rcopy(R"post$cif.test"), rcopy(R"post$cif.test2"), times)

# plots for females
plot_incidence(times, cumincidence_post_bart[:,1,:], cum = true, aalen_johansen = cumincidence_freq[:,1,:], lower = cumincidence_lower_bart[:,1,:], upper = cumincidence_upper_bart[:,1,:])
plot!(size = (480,360), xlim = (0.0,8.5), ylim = (0.0,0.6), xlabel = "\$t\$ (years)")

# plots for males
plot_incidence(times, cumincidence_post_bart[:,2,:], cum = true, aalen_johansen = cumincidence_freq[:,2,:], lower = cumincidence_lower_bart[:,2,:], upper = cumincidence_upper_bart[:,2,:])
plot!(size = (480,360), xlim = (0.0,8.5), ylim = (0.0,0.6), xlabel = "\$t\$ (years)")

##########
# Comparison: survival function
##########

# plots for females
begin

    # initialize plot
    pl = plot(ylim = (0.0, 1.0), xlabel = "\$t\$", ylabel = "\$S(t)\$") 
    plot!(xlim = (0.0,8.5), size = (480,360), xlabel = "\$t\$ (years)")

    # plot frequentist estimate
    plot!(pl, times, survival_freq[:,1], linecolor = :black, label = "freq")

    # plot hCRM estimate
    plot!(pl, times, survival_post[:,1], linecolor = 1, label = "hCRM")
    plot!(pl, times, survival_lower[:,1], fillrange = survival_upper[:,1], linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(pl, times, survival_post_bart[:,1], linecolor = 2, label = "BART")
    plot!(pl, times, survival_lower_bart[:,1], fillrange = survival_upper_bart[:,1], linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_melanoma_survival_female.svg")

end

# plots for males
begin

    # initialize plot
    pl = plot(ylim = (0.0, 1.0), xlabel = "\$t\$", ylabel = "\$S(t)\$") 
    plot!(xlim = (0.0,8.5), size = (480,360), xlabel = "\$t\$ (years)")

    # plot frequentist estimate
    plot!(pl, times, survival_freq[:,2], linecolor = :black, label = "freq")

    # plot hCRM estimate
    plot!(pl, times, survival_post[:,2], linecolor = 1, label = "hCRM")
    plot!(pl, times, survival_lower[:,2], fillrange = survival_upper[:,2], linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(pl, times, survival_post_bart[:,2], linecolor = 2, label = "BART")
    plot!(pl, times, survival_lower_bart[:,2], fillrange = survival_upper_bart[:,2], linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_melanoma_survival_male.svg")

end

##########
# Comparison: cumulative incidence functions
##########

# plots for females
begin

    # initialize plot
    pl = plot(xlabel = "\$t\$", ylabel = "\$F_\\delta(t)\$")
    plot!(size = (480,360), xlim = (0.0,8.5), ylim = (0.0,0.6), xlabel = "\$t\$ (years)", legend = :topleft)

    # plot frequentist estimate
    plot!(pl, times, cumincidence_freq[:,1,:], linecolor = :black, linestyle = [:solid :dash], label = ["freq" false])

    # plot hCRM estimate
    plot!(pl, times, cumincidence_post[:,1,:], linecolor = 1, linestyle = [:solid :dash], label = ["hCRM" false])
    plot!(pl, times, cumincidence_lower[:,1,:], fillrange = cumincidence_upper[:,1,:], 
                linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(pl, times, cumincidence_post_bart[:,1,:], linecolor = 2, linestyle = [:solid :dash], label = ["BART" false])
    plot!(pl, times, cumincidence_lower_bart[:,1,:], fillrange = cumincidence_upper_bart[:,1,:], 
                linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_melanoma_cumincidence_female.svg")

end

# plots for males
begin

    # initialize plot
    pl = plot(xlabel = "\$t\$", ylabel = "\$F_\\delta(t)\$")
    plot!(size = (480,360), xlim = (0.0,8.5), ylim = (0.0,0.6), xlabel = "\$t\$ (years)", legend = :topleft)

    # plot frequentist estimate
    plot!(pl, times, cumincidence_freq[:,2,:], linecolor = :black, linestyle = [:solid :dash], label = ["freq" false])

    # plot hCRM estimate
    plot!(pl, times, cumincidence_post[:,2,:], linecolor = 1, linestyle = [:solid :dash], label = ["hCRM" false])
    plot!(pl, times, cumincidence_lower[:,2,:], fillrange = cumincidence_upper[:,2,:], 
                linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(pl, times, cumincidence_post_bart[:,2,:], linecolor = 2, linestyle = [:solid :dash], label = ["BART" false])
    plot!(pl, times, cumincidence_lower_bart[:,2,:], fillrange = cumincidence_upper_bart[:,2,:], 
                linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_melanoma_cumincidence_male.svg")

end

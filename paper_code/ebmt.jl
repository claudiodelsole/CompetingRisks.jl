# load module
using CompetingRisks

# imports
import Random: seed!
import DataFrames: DataFrame
import StatsBase: counts
import CSV: CSV
import Plots: plot, plot!, vline!, savefig

##########
# Multicenter bone marrow transplantation dataset
##########

# set seed for reproducibility
seed!(42)

# include auxiliary files
include("../aux_code/functions.jl")
include("../aux_code/rbart.jl")

# load dataset
data = CSV.File("data/ebmt.txt")

# retrieve observed variables
T = Float64.(data.ftime)        # observations, time-to-event
Delta = data.fstatus            # event causes
predictors = data.cells         # categorical predictors

# convert days to years
T = T./365.25

# create CompetingRisksDataset
data = DataFrame(T = T, Delta = Delta, predictor = predictors)

# data summary
println("# Data summary")
println("Subjects per competing event: ", counts(data.Delta, 2))
println("Censored observations: ", count(data.Delta .== 0))

# times vector
# upper_time = quantile(crd.T, 0.99)
upper_time = 10.0
times = collect(0.0:0.04:upper_time)

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
stdevs(dishes = 0.2, kernelpars = 0.25, coefficients = 0.5)

# run chain
marginal_estimator, conditional_estimator, params = posterior_sampling(data, cmprsk, nsamples = 2000, times = times, burn_in = 5000)

##########
# Diagnostics
##########

println("# Model hyperparameters")

# number of dishes
plot(summary_dishes(params, burn_in = 5000)[2], size = (480,360), xlim = (0.0, 25.0))
savefig("figures_supp/ebmt_dishes.pdf")

# base measure mass
plot(summary_theta(params, burn_in = 5000)[2], size = (480,360), xlim = (0.0, 0.8))
savefig("figures_supp/ebmt_theta.pdf")

# kernel shape parameter
plot(summary_kernelpars(params, :κ, burn_in = 5000)[2], size = (480,360), xlim = (0.0,2.0))
savefig("figures_supp/ebmt_kappa.pdf")

##########
# Posterior estimates: hazard rate ratio
##########

# frequentist estimate - Cox proportional hazard model
R"coxfit <- coxph(Surv(T, Delta > 0) ~ predictor, data = data)"
R"summary(coxfit)"

# regression coefficient
(pltrace, hist) = summary_coefficients(params, 1, burn_in = 5000)
# plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (-0.55, 0.55))
plot(hist, size = (480,360), xlim = (-0.55, 0.55))
savefig("figures_supp/ebmt_coeffs.pdf")

# estimate hazard ratio
# (pltrace, hist) = summary_coefficients(params, 1, burn_in = 5000, hazard_ratio = true)
# plot(pltrace, size = (480,360), xlim = (0,2500), ylim = (0.5, 1.5))
# plot(hist, size = (480,360), xlim = (0.5, 1.5))

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

# plots for bone marrow cells
plot_survival(times, survival_post[:,1], kaplan_meier = survival_freq[:,1], lower = survival_lower[:,1], upper = survival_upper[:,1])
plot!(size = (480,360), xlim = (0.0,5.5), xlabel = "\$t\$ (years)")
savefig("figures/ebmt_survival.pdf")

# plots for blood cells
# plot_survival(times, survival_post[:,2], kaplan_meier = survival_freq[:,2], lower = survival_lower[:,2], upper = survival_upper[:,2])
# plot!(size = (480,360), xlim = (0.0,5.5), xlabel = "\$t\$ (years)")

##########
# Posterior estimates: cumulative incidence functions
##########

# frequentist estimate for primary cause (GvHD)
R"finegray <- crr(data$T, data$Delta, data$predictor)"
R"cif.GvHD <- predict(finegray, cov1 = as.matrix(data.frame(predictor = c(0,1))))"

# frequentist estimate for competing cause (others)
R"finegray <- crr(data$T, data$Delta, data$predictor, failcode = 2)"
R"cif.other <- predict(finegray, cov1 = as.matrix(data.frame(predictor = c(0,1))))"

# retrieve estimates
cumincidence_freq = cat(mapslices(values -> timepoints(rcopy(R"cif.GvHD[,1]"), values, times), rcopy(R"cif.GvHD[,c(2,3)]"), dims = 1), mapslices(values -> timepoints(rcopy(R"cif.other[,1]"), values, times), rcopy(R"cif.other[,c(2,3)]"), dims = 1), dims = 3)

# posterior estimates
(cumincidence_post, _, _) = estimate_incidence(marginal_estimator, cum = true)
(_, cumincidence_lower, cumincidence_upper) = estimate_incidence(conditional_estimator, cum = true)

# plots for bone marrow cells
plot_incidence(times, cumincidence_post[:,1,:], cum = true, aalen_johansen = cumincidence_freq[:,1,:], lower = cumincidence_lower[:,1,:], upper = cumincidence_upper[:,1,:], mycolors = [2, 3], mylabels = ["GvHD", "death/relapse"])
plot!(size = (480,360), xlim = (0.0,5.5), ylim = (0.0,0.65), xlabel = "\$t\$ (years)")
savefig("figures/ebmt_cumincidence.pdf")

# plots for blood cells
# plot_incidence(times, cumincidence_post[:,2,:], cum = true, aalen_johansen = cumincidence_freq[:,2,:], lower = cumincidence_lower[:,2,:], upper = cumincidence_upper[:,2,:], mycolors = [2, 3], mylabels = ["GvHD", "death/relapse"])
# plot!(size = (480,360), xlim = (0.0,5.5), ylim = (0.0,0.65), xlabel = "\$t\$ (years)")

##########
# Posterior estimates: prediction curves
##########

# posterior estimates
(proportions_post, proportions_lower, proportions_upper) = estimate_proportions(marginal_estimator)

# plots
plot_proportions(times, proportions_post, lower = proportions_lower, upper = proportions_upper, mycolors = [2, 3], mylabels = ["GvHD", "death/relapse"])
plot!(size = (480,360), xlim = (0.0,5.5), xlabel = "\$t\$ (years)")
vline!([maximum(data.T[data.Delta.!=0])], linestyle = :dashdot, linecolor = :black, linealpha = 0.5, label = false)
savefig("figures/ebmt_prediction.pdf")

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

# plots for bone marrow cells
plot_survival(times, survival_post_bart[:,1], kaplan_meier = survival_freq[:,1], lower = survival_lower_bart[:,1], upper = survival_upper_bart[:,1])
plot!(size = (480,360), xlim = (0.0,5.5), xlabel = "\$t\$ (years)")

# plots for blood cells
plot_survival(times, survival_post_bart[:,2], kaplan_meier = survival_freq[:,2], lower = survival_lower_bart[:,2], upper = survival_upper_bart[:,2])
plot!(size = (480,360), xlim = (0.0,5.5), xlabel = "\$t\$ (years)")

##########
# Posterior estimates: cumulative incidence functions
##########

# posterior estimates
(cumincidence_post_bart, cumincidence_lower_bart, cumincidence_upper_bart) = estimate_incidence_bart(rcopy(R"post$times"), rcopy(R"post$cif.test"), rcopy(R"post$cif.test2"), times)

# plots for bone marrow cells
plot_incidence(times, cumincidence_post_bart[:,1,:], cum = true, aalen_johansen = cumincidence_freq[:,1,:], lower = cumincidence_lower_bart[:,1,:], upper = cumincidence_upper_bart[:,1,:])
plot!(size = (480,360), xlim = (0.0,5.5), ylim = (0.0,0.65), xlabel = "\$t\$ (years)")

# plots for blood cells
plot_incidence(times, cumincidence_post_bart[:,2,:], cum = true, aalen_johansen = cumincidence_freq[:,2,:], lower = cumincidence_lower_bart[:,2,:], upper = cumincidence_upper_bart[:,2,:])
plot!(size = (480,360), xlim = (0.0,5.5), ylim = (0.0,0.65), xlabel = "\$t\$ (years)")

##########
# Comparison: survival function
##########

# plots for bone marrow cells
begin

    # initialize plot
    plot(size = (480,360), xlabel = "\$t\$ (years)", ylabel = "\$S(t)\$", xlim = (0.0,5.5), ylim = (0.0, 1.0))

    # plot frequentist estimate
    plot!(times, survival_freq[:,1], linecolor = :black, label = "freq")

    # plot hCRM estimate
    plot!(times, survival_post[:,1], linecolor = 1, label = "hCRM")
    plot!(times, survival_lower[:,1], fillrange = survival_upper[:,1], linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(times, survival_post_bart[:,1], linecolor = 2, label = "BART")
    plot!(times, survival_lower_bart[:,1], fillrange = survival_upper_bart[:,1], linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_ebmt_survival_bm.pdf")

end

# plots for blood cells
begin

    # initialize plot
    plot(size = (480,360), xlabel = "\$t\$ (years)", ylabel = "\$S(t)\$", xlim = (0.0,5.5), ylim = (0.0, 1.0))

    # plot frequentist estimate
    plot!(times, survival_freq[:,2], linecolor = :black, label = "freq")

    # plot hCRM estimate
    plot!(times, survival_post[:,2], linecolor = 1, label = "hCRM")
    plot!(times, survival_lower[:,2], fillrange = survival_upper[:,2], linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(times, survival_post_bart[:,2], linecolor = 2, label = "BART")
    plot!(times, survival_lower_bart[:,2], fillrange = survival_upper_bart[:,2], linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_ebmt_survival_blood.pdf")

end

##########
# Comparison: cumulative incidence functions
##########

# plots for bone marrow cells
begin

    # initialize plot
    plot(size = (480,360), xlabel = "\$t\$ (years)", ylabel = "\$F_\\delta(t)\$", xlim = (0.0,5.5), ylim = (0.0,0.65), legend = :topleft)

    # plot frequentist estimate
    plot!(times, cumincidence_freq[:,1,:], linecolor = :black, linestyle = [:solid :dash], label = ["freq" false])

    # plot hCRM estimate
    plot!(times, cumincidence_post[:,1,:], linecolor = 1, linestyle = [:solid :dash], label = ["hCRM" false])
    plot!(times, cumincidence_lower[:,1,:], fillrange = cumincidence_upper[:,1,:], 
                linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(times, cumincidence_post_bart[:,1,:], linecolor = 2, linestyle = [:solid :dash], label = ["BART" false])
    plot!(times, cumincidence_lower_bart[:,1,:], fillrange = cumincidence_upper_bart[:,1,:], 
                linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_ebmt_cumincidence_bm.pdf")

end

# plots for blood cells
begin

    # initialize plot
    plot(size = (480,360), xlabel = "\$t\$", ylabel = "\$F_\\delta(t)\$", xlim = (0.0,5.5), ylim = (0.0,0.65), legend = :topleft)

    # plot frequentist estimate
    plot!(times, cumincidence_freq[:,2,:], linecolor = :black, linestyle = [:solid :dash], label = ["freq" false])

    # plot hCRM estimate
    plot!(times, cumincidence_post[:,2,:], linecolor = 1, linestyle = [:solid :dash], label = ["hCRM" false])
    plot!(times, cumincidence_lower[:,2,:], fillrange = cumincidence_upper[:,2,:], 
                linecolor = 1, linealpha = 0.0, fillcolor = 1, fillalpha = 0.2, primary = false)

    # plot BART estimate
    plot!(times, cumincidence_post_bart[:,2,:], linecolor = 2, linestyle = [:solid :dash], label = ["BART" false])
    plot!(times, cumincidence_lower_bart[:,2,:], fillrange = cumincidence_upper_bart[:,2,:], 
                linecolor = 2, linealpha = 0.0, fillcolor = 2, fillalpha = 0.2, primary = false)

    # save figure
    savefig("figures_supp/compare_ebmt_cumincidence_blood.pdf")

end

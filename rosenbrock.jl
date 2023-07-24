using LinearAlgebra, Random
using Distributions, Plots

include("eki_mini.jl")
include("gradientdescent.jl")

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

rosenbrock(x, a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2 
N_steps = 45

initials = [[-1,-1],[1.5,-0.5],[0,2]]

## minimize rosenbrock using gradient descent
alpha_ = 5*1e-3 # step size
conv_gd = zeros(length(initials),N_steps+1)
for i in 1:3
    final_gd, conv_gd[i,:]= run_gd(initials[i], rosenbrock, alpha_, N_steps)
end

## now with momentum
beta_ = 0.9
alpha_ = 5*1e-3
conv_gdm = zeros(length(initials), N_steps+1)
for i in 1:3
    final_gdm, conv_gdm[i,:] = run_gd_momentum(initials[i], rosenbrock, alpha_, beta_, N_steps)
end

## now with nesterov variation on momentum
conv_gdn = zeros(length(initials), N_steps+1)
for i in 1:3
    final_gdn, conv_gdn[i,:] = run_gd_nesterov(initials[i], rosenbrock, alpha_, beta_, N_steps)
end



## reinterpret as an inverse problem

dim_output = 1
dim_input = 2
Γ = I(dim_output)*0.1  # we don't add noise to the observation, Γ^-1 informs EKI algorithm similar to step size
noise_dist = MvNormal(zeros(dim_output), Γ)
prior = MvNormal(zeros(dim_input), I*0.1) ## variance on prior?
theta_true = [1.0, 1.0]  ## known location of minimum
G(theta) = rosenbrock(theta)*I(1) #  quick fix for scalar issue

N_trials = 100
N_ensemble = 5
N_iterations = 15
convs = zeros(N_trials, N_iterations+1)
convs_m = zeros(N_trials, N_iterations+1)
convs_m1 = zeros(N_trials, N_iterations+1)
convs_m2 = zeros(N_trials, N_iterations+1)
r = [3,5,10]

# for one trial only, we will visualize in parameter space
global ens_historical = zeros(N_iterations+1, dim_input, N_ensemble)
global ens_historical_m = zeros(N_iterations+1, dim_input, N_ensemble)


for trial in 1:N_trials
    local y = G(theta_true) + rand(noise_dist) # each trial has new random noise

    function loss_eki(theta)  # used for plotting
        return norm((G(theta) - y))
    end

    # sample initial ensemble and perform EKI
    local initial_ensemble = draw_initial(prior, N_ensemble)
    global ens_historical, conv_eki = run_eki_tracked(initial_ensemble, G, y, Γ, N_iterations, loss_eki)

    global ens_historical_m, conv_eki_m = run_eki_momentum_tracked(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[1])
    local final_ensemble_m1, conv_eki_m1 = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[2])
    local final_ensemble_m2, conv_eki_m2 = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[3])

    convs[trial,:] = mean(conv_eki, dims=2) # mean over ensemble members
    convs_m[trial, :] = mean(conv_eki_m, dims=2)  
    convs_m1[trial, :] = mean(conv_eki_m1, dims=2)
    convs_m2[trial, :] = mean(conv_eki_m2, dims=2)
end

plots = []

## code for individual IC plots:
# for k in 1:length(initials)
#     plot_gd = plot([1:N_steps+1], conv_gd[k,:], c = :black, label = "GD", legend = :topright, linewidth = 2)
#     plot!([1:N_steps+1], conv_gdm[k,:], c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
#     plot!([1:N_steps+1], conv_gdn[k,:], c = :green, label = "GD with Nesterov momentum", legend = :topright, linewidth = 2)
#     xlabel!("Iteration")
#     ylabel!("Loss")
#     push!(plots,plot_gd)
# end

## EKI CONV PLOT
plota = plot([1:N_iterations+1], mean(log.(convs), dims=1)', c = :black, label="traditional EKI")
plot!([1:N_iterations+1], mean(log.(convs_m), dims=1)', c = :red, label="r = "*string(r[1]))
plot!([1:N_iterations+1], mean(log.(convs_m1), dims=1)', c = :blue, label="r = "*string(r[2]))
plot!([1:N_iterations+1], mean(log.(convs_m2), dims=1)', c = :green, label="r = "*string(r[3]))

xlabel!("EKI iteration, N_ensemble = "*string(N_ensemble))
ylabel!("log(Loss)")
display(plota)

## EKI PARAMETER SPACE
plot_param = plot()
xlims!(-1,1.5)
ylims!(-1,1.2)
for i in 1:N_iterations+1
    plot!(ens_historical[i,1,:],ens_historical[i,2,:],seriestype=:scatter, label="", ms=2)
end
plot!(theta_true, seriestype=:scatter, ms = 7, label="truth") 
xlabel!("theta_1")
ylabel!("theta_2")
title!("Parameter evolution (no momentum)")
display(plot_param)

## EKI PARAMETER SPACE
plot_param = plot()
xlims!(-1,1.5)
ylims!(-1,1.2)
for i in 1:N_iterations+1
    plot!(ens_historical_m[i,1,:],ens_historical_m[i,2,:],seriestype=:scatter, label="", ms=2)
end
plot!(theta_true, seriestype=:scatter, ms = 7, label="truth") 
xlabel!("theta_1")
ylabel!("theta_2")
title!("Parameter evolution (momentum)")
display(plot_param)

## GRAD DESCENT CONV PLOT
plot_means = plot([1:N_steps+1], mean(log.(conv_gd[i,:]) for i in 1:length(initials)), c = :black, label = "GD", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], mean(log.(conv_gdm[i,:]) for i in 1:3), c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], mean(log.(conv_gdn[i,:]) for i in 1:3), c = :green, label = "GD with Nesterov momentum", legend = :topright, linewidth = 2)
xlabel!("GD step")
ylabel!("log(Loss)")
push!(plots,plot_means)

plot(plots...)

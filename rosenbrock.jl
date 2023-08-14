using LinearAlgebra, Random
using Distributions
using Plots; pythonplot()

include("eki_mini.jl")
include("gradientdescent.jl")

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

rosenbrock(x, a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2 
N_steps = 35

initials = [[-1,-1],[1.5,-0.5],[0,2]]

## minimize rosenbrock using gradient descent
alpha_ = 5*1e-3 # step size
conv_gd = zeros(length(initials),N_steps+1)
for i in 1:3
    final_gd, conv_gd[i,:]= run_gd(initials[i], rosenbrock, alpha_, N_steps)
end

## now with momentum
beta_ = 0.5
alpha_ = 5*1e-3
conv_gdm = zeros(length(initials), N_steps+1)
for i in 1:3
    final_gdm, conv_gdm[i,:] = run_gd_momentum(initials[i], rosenbrock, alpha_, beta_, N_steps)
end

## now with nesterov variation on momentum
beta_ = 0.75
conv_gdn = zeros(length(initials), N_steps+1)
for i in 1:3
    final_gdn, conv_gdn[i,:] = run_gd_nesterov(initials[i], rosenbrock, alpha_, beta_, N_steps)
end


## reinterpret as an inverse problem

dim_output = 1
dim_input = 2
Γ = I(dim_output)*0.001 ## small error since this is not a real inverse problem
noise_dist = MvNormal(zeros(dim_output), Γ)
prior = MvNormal([3,2], I(2)*0.1) #MvNormal([0,0],I(2))
theta_true = [1.0, 1.0]  ## known location of minimum
G(theta) = rosenbrock(theta)*I(1) #  quick fix for scalar issue

N_trials = 100
N_ensemble = 5
N_iterations = 100
convs = zeros(N_trials, N_iterations+1)
convs_m = zeros(N_trials, N_iterations+1)
convs_m_means = zeros(N_trials, N_iterations+1)
convs_m1 = zeros(N_trials, N_iterations+1)
convs_m2 = zeros(N_trials, N_iterations+1)
r = [3,5,10]

# for one trial only, we will visualize in parameter space
global ens_historical = zeros(N_iterations+1, dim_input, N_ensemble)
global ens_historical_m = zeros(N_iterations+1, dim_input, N_ensemble)
lambda = 0.5

for trial in 1:N_trials
    local y = G(theta_true) #+ rand(noise_dist) # each trial has new random noise

    # function loss_eki(theta)  # used for plotting
    #     return norm((G(theta) - y))
    # end

    function loss_eki(theta)
        return norm(inv(Γ).^0.5 * (G(theta) .- y).^2)
    end

    # sample initial ensemble and perform EKI
    local initial_ensemble = draw_initial(prior, N_ensemble)
    global ens_historical, conv_eki = run_eki_tracked(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1)

    global ens_historical_m, conv_eki_m = run_eki_momentum_tracked(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[1])
    local final_ensemble_m1, conv_eki_m1 = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[2])
    local final_ensemble_m2, conv_eki_m2 = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[3])
    local final_ensemble_means, conv_eki_means = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_eki, 1,r[1],true)

    convs[trial,:] = mean(conv_eki, dims=2) # mean over ensemble members
    convs_m[trial, :] = mean(conv_eki_m, dims=2)  
    convs_m1[trial, :] = mean(conv_eki_m1, dims=2)
    convs_m2[trial, :] = mean(conv_eki_m2, dims=2)
    convs_m_means[trial, :] = mean(conv_eki_means, dims=2)
end

plots = []

# ## EKI CONV PLOT
plota = plot([1:N_iterations+1], mean(log.(convs), dims=1)', c = :black, label="traditional EKI")
plot!([1:N_iterations+1], mean(log.(convs_m), dims=1)', c = :red, label="r = "*string(r[1]))
# plot!([1:N_iterations+1], mean(log.(convs_m1), dims=1)', c = :blue, label="r = "*string(r[2]))
# plot!([1:N_iterations+1], mean(log.(convs_m2), dims=1)', c = :green, label="r = "*string(r[3]))
plot!([1:N_iterations+1], mean(log.(convs_m_means), dims=1)', c = :blue, label="ensemble mean momentum r="*string(r[1]))
xlabel!("EKI iteration, N_ensemble = "*string(N_ensemble)*", N_trials = "*string(N_trials))
ylabel!("log(Loss)")
display(plota)
savefig(plota, "rosen_conv.pdf")

## EKI single-trial conv plot (corresponding to param space plots)
plota = plot([1:N_iterations+1], (log.(convs)[N_trials,:]), c = :black, label="traditional EKI")
plot!([1:N_iterations+1], (log.(convs_m)[N_trials,:]), c = :red, label="r = "*string(r[1]))
plot!([1:N_iterations+1], (log.(convs_m_means)[N_trials,:]), c = :blue, label="ensemble mean momentum r="*string(r[1]))
xlabel!("EKI iteration, N_ensemble = "*string(N_ensemble)*", final trial")
ylabel!("log(Loss)")
display(plota)

## EKI PARAMETER SPACE
plot_param = plot()
xlims!(-5,5)
ylims!(-5,5)
xlabel!("theta_1")
ylabel!("theta_2")
title!("Rosenbrock IP parameter evolution (no momentum)")
xrange = collect(LinRange(-5,5,200)) #-1:0.1:1.5
yrange = collect(LinRange(-5,5,300))
a=1
b=5
z = reshape([@.log((a-x)^2 + b*(y - x.^2)^2) for x in xrange for y in yrange],300,200)
contour!(yrange, xrange, z', levels=15, clabels=false, color=[:black])
ens_mean_historical = mean(ens_historical, dims=3)
for i in 1:N_iterations
    plot!(ens_historical[i,1,:],ens_historical[i,2,:],seriestype=:scatter, label="", marker_z=i, cbar=false)
   # plot!([ens_mean_historical[i,1]], [ens_mean_historical[i,2]], seriestype=:scatter, label="", marker_z=i, cbar=false)
end
plot!([1], [1], ms = 7, label="truth", markershape=:star5, markeralpha=1, mc=:red) 
display(plot_param)
savefig(plot_param, "param_traditional.pdf")

## EKI PARAMETER SPACE
plot_param = plot()
xlims!(-5,5)
ylims!(-5,5)
contour!(yrange, xrange, z', levels=15, clabels=false, color=[:black])
ens_mean_historical = mean(ens_historical_m, dims=3)
for i in 1:N_iterations+1
    plot!(ens_historical_m[i,1,:],ens_historical_m[i,2,:],seriestype=:scatter, label="", marker_z=i, cbar=false)
    #plot!([ens_mean_historical[i,1]], [ens_mean_historical[i,2]], seriestype=:scatter, label="", marker_z=i, cbar=false)
end
plot!([1], [1], ms = 7, label="truth", markershape=:star5, markeralpha=1, mc=:red) 
xlabel!("theta_1")
ylabel!("theta_2")
title!("Rosenbrock IP parameter evolution (momentum)")
display(plot_param)
savefig(plot_param, "param_momentum.pdf")

# ## GRAD DESCENT CONV PLOT
# plot_means = plot([1:N_steps+1], mean((conv_gd[i,:]) for i in 1:length(initials)), c = :black, label = "GD", legend = :topright, linewidth = 2)
# #plot!([1:N_steps+1], mean(log.(conv_gdm[i,:]) for i in 1:3), c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
# plot!([1:N_steps+1], mean((conv_gdn[i,:]) for i in 1:3), c = :green, label = "GD with Nesterov momentum", legend = :topright, linewidth = 2)
# xlabel!("GD step")
# ylabel!("Loss")
# title!("Gradient Descent Convergence on the Rosenbrock Function")
# push!(plots,plot_means)

# plot(plots...)

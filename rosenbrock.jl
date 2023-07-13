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

plots = []

for k in 1:length(initials)
    plot_gd = plot([1:N_steps+1], conv_gd[k,:], c = :black, label = "GD", legend = :topright, linewidth = 2)
    plot!([1:N_steps+1], conv_gdm[k,:], c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
    plot!([1:N_steps+1], conv_gdn[k,:], c = :green, label = "GD with Nesterov momentum", legend = :topright, linewidth = 2)
    xlabel!("Iteration")
    ylabel!("Loss")
    push!(plots,plot_gd)
end

plot_means = plot([1:N_steps+1], mean(conv_gd[i,:] for i in 1:3), c = :black, label = "GD", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], mean(conv_gdm[i,:] for i in 1:3), c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], mean(conv_gdn[i,:] for i in 1:3), c = :green, label = "GD with Nesterov momentum", legend = :topright, linewidth = 2)
xlabel!("Iteration")
ylabel!("Loss")
push!(plots,plot_means)

plot(plots...)




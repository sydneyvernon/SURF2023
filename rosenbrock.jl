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
prior = MvNormal(zeros(dim_input), I) ## variance on prior?
theta_true = [1.0, 1.0]  ## known location of minimum
G(theta) = rosenbrock(theta)*I(1) #  quick fix for scalar issue
y = G(theta_true)

function loss_eki(theta)  # used for plotting; y = 0 so it is simply function value
    return norm((G(theta) - y))
end

# sample initial ensemble and perform EKI
N_ensemble = 5
N_iterations = 10
initial_ensemble = draw_initial(prior, N_ensemble)
final_ensemble, conv_eki = run_eki(initial_ensemble, G, y, Γ, N_iterations, loss_eki)

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

plota = plot([1:N_iterations+1], log.(mean(conv_eki, dims=2)), c = :black, label="")
xlabel!("EKI iteration, N_ensemble = "*string(N_ensemble))
ylabel!("log(Loss)")
display(plota)

plot_means = plot([1:N_steps+1], mean(log.(conv_gd[i,:]) for i in 1:length(initials)), c = :black, label = "GD", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], mean(log.(conv_gdm[i,:]) for i in 1:3), c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], mean(log.(conv_gdn[i,:]) for i in 1:3), c = :green, label = "GD with Nesterov momentum", legend = :topright, linewidth = 2)
xlabel!("GD step")
ylabel!("log(Loss)")
push!(plots,plot_means)

plot(plots...)

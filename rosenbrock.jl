using LinearAlgebra, Random
using Distributions, Plots

include("eki_mini.jl")
include("gradientdescent.jl")

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

rosenbrock(x, a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2 


## minimize rosenbrock using gradient descent
initial = [-1,-1]
N_steps = 45
alpha_ = 5*1e-3 # step size
final_gd, conv_gd= run_gd(initial, rosenbrock, alpha_, N_steps)

## now with momentum
beta_ = 0.9
alpha_ = 2*1e-3
final_gdm, conv_gdm = run_gd_momentum(initial, rosenbrock, alpha_, beta_, N_steps)

plot([1:N_steps+1], conv_gd, c = :black, label = "GD", legend = :topright, linewidth = 2)
plot!([1:N_steps+1], conv_gdm, c = :blue, label = "GD with momentum", legend = :topright, linewidth = 2)
xlabel!("Iteration")
ylabel!("Loss")






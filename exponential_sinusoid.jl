using LinearAlgebra, Random
using Distributions, Plots

include("eki_mini.jl")
include("gradientdescent.jl")


dt = 0.01
trange = 0:dt:(2 * pi + dt)

# define model
function G(u)
    theta, vert_shift = u
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end

function model(amplitude, vert_shift)
    phi = 2 * pi * rand(rng)
    return exp.(amplitude*sin.(trange .+ phi) .+ vert_shift)
end

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)


function main()
    dim_output = 2
    Γ = I(2)*0.1
    noise_dist = MvNormal(zeros(dim_output), Γ)
    prior = MvNormal([2,0], diagm([1,5]))

    theta_true = [1.0, 0.8]

    #N_trials = 5

    #for trial in 1:N_trials:
    y = G(theta_true) .+ rand(noise_dist)
  
    # very simple loss function
    function loss_fn( 
        theta::Any,
    )
        return mean((theta - theta_true).^2) ## MSE on parameters
    end

    # sample initial ensemble and perform EKI
    N_ensemble = 5
    N_iterations = 7
    initial_ensemble = draw_initial(prior, N_ensemble)
    final_ensemble, conv_eki = run_eki(initial_ensemble, G, y, Γ, N_iterations, loss_fn)


    # PLOT MODEL EVALUATIONS, INITIAL/FINAL (all)
    plot_a = plot(trange, model(theta_true...), c = :black, label = "Truth", legend = :bottomright, linewidth = 2)
    plot!(
        trange,
        [model(initial_ensemble[:, i]...) for i in 1:N_ensemble],
        c = :red,
        label = ["Initial ensemble" "" "" "" ""],
    )
    plot!(trange, [model(final_ensemble[:, i]...) for i in 1:N_ensemble], c = :blue, label = ["Final ensemble" "" "" "" ""])
    #plot!(trange, model(final_gd...), c=:green, label = "Final after GD")
    xlabel!("Time")
    display(plot_a)

    # PLOT CONVERGENCE (EKI)
    plot_b = plot([1:N_iterations+1], [conv_eki[:,j] for j in 1:N_ensemble], c = :black, label=["Loss" "" "" "" ""], legend = :topright, linewidth = 2)
    xlabel!("EKI Iteration")
    display(plot_b)

    # PLOT CONVERGENCE (EKI)
    plot_c = plot([1:N_iterations+1], [log.(conv_eki[:,j]) for j in 1:N_ensemble], c = :black, label=["log(Loss)" "" "" "" ""], legend = :topright, linewidth = 2)
    xlabel!("EKI Iteration")
    display(plot_c)

    # PLOT CONVERGENCE (GD)
    #plot_c = plot([1:N_steps+1], conv_gd, c = :black, label = "Loss", legend = :topright, linewidth = 2)
    #xlabel!("Gradient Descent Iteration")
    #display(plot_c)

end

main()


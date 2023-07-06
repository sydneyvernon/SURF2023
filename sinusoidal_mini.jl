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
    return amplitude * sin.(trange .+ phi) .+ vert_shift
end

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)


function main()
    dim_output = 2
    Γ = I(2)*0.1
    noise_dist = MvNormal(zeros(dim_output), Γ)
    prior = MvNormal(zeros(dim_output), I)

    theta_true = [1.0, 7.0]
    y = G(theta_true) .+ rand(noise_dist)

    # perform gradient descent
    alpha_ = 1e-1 # step size
    N_steps = 20
    theta_0 = mean(prior) # start with an initial guess informed by the prior
    
    # very simple loss function
    function loss_fn( 
        theta::Any,
    )
        #return 0.5*(norm(G(theta) .- y).^2 + norm(theta .- theta_0).^2) # for EKI members, theta_0 is not the mean of the prior
        return norm(G(theta) .- y).^2
    end

    final_gd, conv_gd = run_gd(theta_0, loss_fn, alpha_, N_steps)

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
    plot!(trange, model(final_gd...), c=:green, label = "Final after GD")
    xlabel!("Time")
    display(plot_a)

    # PLOT CONVERGENCE (EKI)
    plot_b = plot([1:N_iterations+1], [conv_eki[:,j] for j in 1:N_ensemble], c = :black, label=["Loss" "" "" "" ""], legend = :topright, linewidth = 2)
    xlabel!("EKI Iteration")
    display(plot_b)

    # PLOT CONVERGENCE (GD)
    plot_c = plot([1:N_steps+1], conv_gd, c = :black, label = "Loss", legend = :topright, linewidth = 2)
    xlabel!("Gradient Descent Iteration")
    display(plot_c)

end

main()


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
    prior = MvNormal([2,0], diagm([1,25]))

    theta_true = [1.0, 0.8]

    function loss_fn(   ##  squared norm on parameters
        theta::Any,
    )
        return norm(theta-theta_true).^2
    end

    N_trials = 100
    N_iterations = 30  # EKI iterations in each trial
    N_ensemble = 50

    # store convergence for each trial
    convs = zeros(N_trials, N_iterations+1)

    for trial in 1:N_trials
        y = G(theta_true) .+ rand(noise_dist)  # each trial has new noise on obs
    
        # sample initial ensemble and perform EKI        
        initial_ensemble = draw_initial(prior, N_ensemble)
        final_ensemble, conv_eki = run_eki(initial_ensemble, G, y, Γ, N_iterations, loss_fn)

        convs[trial, :] = mean(conv_eki, dims=2)  # mean loss of all ensemble members
    end

    # PLOT CONVERGENCE (EKI)
    plot_b = plot([0:N_iterations], mean(convs, dims=1)[:], c = :black, label=["Loss" "" "" "" ""], legend = :topright, linewidth = 2)
    xlabel!("EKI Iteration")
    display(plot_b)

    ## PLOT CONVERGENCE (EKI)
    plot_c = plot([0:N_iterations],log.(mean(convs, dims=1)[:]), c = :black, label="") #, legend = :topright, linewidth = 2)
    xlabel!("EKI Iteration")
    ylabel!("log(Loss)")
    display(plot_c)
    savefig(plot_c, "exp_sin.pdf")


end

main()


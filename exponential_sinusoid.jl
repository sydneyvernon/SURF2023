using LinearAlgebra, Random
using Distributions, Plots

include("eki_mini.jl")
include("gradientdescent.jl")


dt = 0.01
trange = 0:dt:(2 * pi + dt)

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

# define model
function G(u)
    theta, vert_shift = u
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end

function model(amplitude, vert_shift)
    phi = 2 * pi #* rand(rng)
    return exp.(amplitude*sin.(trange .+ phi) .+ vert_shift)
end

function main()
    dim_output = 2
    Γ = I(2)*1
    noise_dist = MvNormal(zeros(dim_output), Γ)
    prior = MvNormal([2,0], diagm([1,25]))

    theta_true = [1.0, 0.8]

    N_trials = 10
    N_iterations = 30  # EKI iterations in each trial
    N_ensemble = 50

    N_steps = 1000 # for GD
    alpha = 5*1e-2

    # store convergence for each trial
    convs = zeros(N_trials, N_iterations+1)
    convs_gd = zeros(N_trials, N_steps+1)

    for trial in 1:N_trials
        y = G(theta_true) .+ rand(noise_dist)  # each trial has new noise on obs

        function loss_fn(   ##  use log loss to deal with exponential
            theta
        )
            #return (G(theta) .- y)' * inv(Γ) * (G(theta) .- y) # zygote cannot differentiate this
            return log.(norm(inv(Γ).^0.5 * (G(theta) .- y)))
        end
    
        function loss_fn_clean(
            theta
        )
            return log.(norm((G(theta) .- G(theta_true)))) # no noise on obs AND no noise variance scaling
        end

        # sample initial ensemble and perform EKI        
        initial_ensemble = draw_initial(prior, N_ensemble)
        final_ensemble, conv_eki = run_eki(initial_ensemble, G, y, Γ, N_iterations, loss_fn)

        initial_gd = [3.5,0.8] #rand(prior) # each run has different IC
        final_gd, conv_gd = run_gd(initial_gd, loss_fn_clean, alpha, N_steps)

        convs[trial, :] = mean(conv_eki, dims=2)  # mean loss of all ensemble members
        convs_gd[trial, :] = conv_gd
    end

    ## PLOT log CONVERGENCE (EKI)
    plot_c = plot([0:N_iterations], (mean(convs, dims=1)[:]), c = :black, label="")
    xlabel!("EKI Iteration, N_ensemble = "*string(N_ensemble))
    ylabel!("log(Loss)")
    display(plot_c)
    savefig(plot_c, "exp_sin.pdf")

    # PLOT log CONVERGENCE (GD)
    plot_b = plot([0:N_steps], (mean((convs_gd), dims=1)[:]), c = :black, label="",linewidth=2)
    xlabel!("GD Iteration")
    ylabel!("log(Loss)")
    display(plot_b)

end

main()


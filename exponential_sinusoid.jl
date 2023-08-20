using LinearAlgebra, Random
using Distributions, Plots
ENV["GKSwstype"]="100"

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
    phi = 2 * pi * rand(rng)
    return exp.(amplitude*sin.(trange .+ phi) .+ vert_shift)
end

function main()
    dim_output = 2
    Γ = I(2)*0.01
    noise_dist = MvNormal(zeros(dim_output), Γ)
    prior = MvNormal([2,0], diagm([0.1,0.5]))

    theta_true = [1.0, 0.8]

    N_trials = 50
    N_iterations = 100  # EKI iterations in each trial
    N_ensemble = 10
    dt = 1

    N_steps = 100 # for GD
    alpha = 5*1e-2

    # store convergence for each trial
    convs = zeros(N_trials, N_iterations+1)
    convs_m = zeros(N_trials, N_iterations+1)
    convs_m_v = zeros(N_trials, N_iterations+1)
    convs_m_highorder = zeros(N_trials, N_iterations+1)
    convs_m_means = zeros(N_trials, N_iterations+1)
    convs_m_highorder_means = zeros(N_trials, N_iterations+1)
    convs_m_means_con = zeros(N_trials, N_iterations+1)
    convs_m_con = zeros(N_trials, N_iterations+1)
    global ens_final_eki = 0
    global ens_final_eki_m = 0
    # convs_gd = zeros(N_trials, N_steps+1)
    # global ens_historical = zeros(N_iterations+1, dim_input, N_ensemble)
    # global ens_historical_m = zeros(N_iterations+1, dim_input, N_ensemble)
    # global gd_historical = zeros(N_steps+1, dim_input)

    beta = 0.5 ## for nesterov gd

    for trial in 1:N_trials
        y = G(theta_true) .+ rand(noise_dist)  # each trial has new noise on obs

        function loss_fn(   ##  use log loss to deal with exponential
            theta
        )
            return log.(norm(inv(Γ).^0.5 * (G(theta) .- y)).^2)
        end
    
        function loss_fn_clean(
            theta
        )
            return log.(norm((G(theta) .- G(theta_true))).^2) # no noise on obs AND no noise variance scaling
        end

        # sample initial ensemble and perform EKI        
        initial_ensemble = draw_initial(prior, N_ensemble)
        global ens_final_eki, conv_eki = run_eki(initial_ensemble, G, y, Γ, N_iterations, loss_fn, dt)
        global ens_final_eki_m, conv_eki_m, conv_eki_m_v = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_fn,dt,3)
        #global ens_final_eki_m, conv_eki_means = run_eki_momentum(initial_ensemble, G, y, Γ, N_iterations, loss_fn,dt,3,true)
        # global ens_final, conv_eki_high = run_eki_momentum_highorder(initial_ensemble, G, y, Γ, N_iterations, loss_fn,dt)
        # global ens_final, conv_eki_high_means = run_eki_momentum_highorder(initial_ensemble, G, y, Γ, N_iterations, loss_fn,dt,4,0.51,true)

        #global ens_final_means, conv_eki_means_con = run_eki_momentum_means_constrained(initial_ensemble, G, y, Γ, N_iterations, loss_fn,1,3)
        #global ens_final_means, conv_eki_con = run_eki_momentum_constrained(initial_ensemble, G, y, Γ, N_iterations, loss_fn,1,3)

        # initial_gd = [3.5,0.8] #rand(prior) # each run has different IC
        # global gd_historical, conv_gd = run_gd_nesterov_tracked(initial_gd, loss_fn_clean, alpha, beta, N_steps)

        convs[trial, :] = conv_eki  # mean loss of all ensemble members
        convs_m[trial, :] = conv_eki_m
        convs_m_v[trial, :] = conv_eki_m_v

        # convs_m_means[trial,:] = conv_eki_means
        # convs_m_highorder[trial,:] = conv_eki_high
        # convs_m_highorder_means[trial,:] = conv_eki_high_means
        # convs_m_con[trial,:] = mean(conv_eki_con, dims=2)
        # convs_m_means_con[trial,:] = mean(conv_eki_means_con, dims=2)
        # convs_gd[trial, :] = conv_gd
    end

    # # PLOT log CONVERGENCE (GD)
    # plot_b = plot([0:N_steps], (mean((convs_gd), dims=1)[:]), c = :black, label="",linewidth=2)
    # xlabel!("GD Iteration")
    # ylabel!("log(Loss)")
    # display(plot_b)

    # # PLOT PARAM SPACE GD
    # plot_param = plot()
    # xlims!(-0.5,2.5)
    # ylims!(-1,2)
    # for i in 1:N_iterations
    #     plot!(gd_historical[i,1,:],gd_historical[i,2,:],seriestype=:scatter, label="", marker_z = i)
    # end
    # plot!([1], [0.8], ms = 7, label="truth", markershape=:star5,) 
    # xlabel!("theta_1")
    # ylabel!("theta_2")
    # title!("Exp sin GD parameter evolution")
    # display(plot_param)

    ## PLOT log CONVERGENCE (EKI)
    plot_c = plot([0:N_iterations], (mean(convs, dims=1)[:]), c = :black, label="EKI, traditional")
    plot!([0:N_iterations], (mean(convs_m, dims=1)[:]), c = :red, label="EKI with momentum")
    plot!([0:N_iterations], (mean(convs_m_v, dims=1)[:]), c = :green, label="EKI with momentum, v plot")

    # plot!([0:N_iterations], (mean(convs_m_means, dims=1)[:]), c = :blue, label="EKI+momentum (ens-mean)")
    # plot!([0:N_iterations], (mean(convs_m_highorder, dims=1)[:]), c = :green, label="high order")
    # plot!([0:N_iterations], (mean(convs_m_highorder_means, dims=1)[:]), c = :orange, label="high order (ens-mean)")

    #plot!([0:N_iterations], (mean(convs_m_con, dims=1)[:]), c = :green, label="EKI+momentum (con)")
    #plot!([0:N_iterations], (mean(convs_m_means_con, dims=1)[:]), c = :gray, label="EKI+ ens-mean momentum (con)")

    xlabel!("EKI Iteration, N_ensemble = "*string(N_ensemble)*", N_trials = "*string(N_trials)*", dt = "*string(dt))
    ylabel!("log(Loss)")
    display(plot_c)
    savefig(plot_c, "exp_sin.pdf")

    # ## PLOT PARAMETER SPACE (FROM ONE EKI TRIAL)
    # plot_param = plot()
    # xlims!(-0.5,2.5)
    # ylims!(-1,2)
    # for i in 1:N_iterations
    #     plot!(ens_historical[i,1,:],ens_historical[i,2,:],seriestype=:scatter, label="", marker_z = i)
    # end
    # plot!([1], [0.8], ms = 7, label="truth", markershape=:star5,) 
    # xlabel!("theta_1")
    # ylabel!("theta_2")
    # title!("Exp sin parameter evolution (no momentum)")
    # display(plot_param)

    # plot_param = plot()
    # xlims!(-0.5,2.5)
    # ylims!(-1,2)
    # for i in 1:N_iterations
    #     plot!(ens_historical_m[i,1,:],ens_historical_m[i,2,:],seriestype=:scatter, label="", marker_z = i)
    # end
    # plot!([1], [0.8], ms = 7, label="truth", markershape=:star5,) 
    # xlabel!("theta_1")
    # ylabel!("theta_2")
    # title!("Exp sin parameter evolution (momentum)")
    # display(plot_param)

    # # PLOT MODEL EVALUATIONS, INITIAL/FINAL (all)
    # println(size(convs))
    # initial_ensemble = 
    # initial_ensemble_m = 
    # final_ensemble = ens_final_eki
    # final_ensemble_m = ens_final_eki_m
    # plot_a = plot(trange, model(theta_true...), c = :black, label = "Truth", legend = :bottomright, linewidth = 2)
    # plot!(
    #     trange,
    #     [model(initial_ensemble[:, i]...) for i in 1:N_ensemble],
    #     c = :red,
    #     label = ["Initial ensemble" "" "" "" ""],
    # )
    # plot!(trange, [model(final_ensemble[:, i]...) for i in 1:N_ensemble], c = :blue, label = ["Final ensemble" "" "" "" ""])
    # xlabel!("Time")
    # display(plot_a)
    # savefig(plot_a,"exp_sinusoidal_modeled.pdf") 

end

main()


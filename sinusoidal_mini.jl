using LinearAlgebra, Random
using Distributions, Plots

# put this in eki_mini file
function draw_initial(
    prior::Distribution{},
    num::Int
)
    a = rand(prior,num)
    #for i=1:num     # quick way to enure amplitude positive
    #    if a[1,i] < 0
    #        a[1,i] = -a[1,i]
    #    end
    #end
    return a
end

function eki_update(
    ens::AbstractMatrix{},
    G_, #, #, # function
    y_, # target
    Γ_ #::Real{}
)
    N = size(ens)[2] # number of ensemble members
    ens_eval = G_(ens[:,1]) # first output
    for i in 2:N    # other outputs if applicable   # gross !!
        ens_eval = hcat(ens_eval, G_(ens[:,i]))
    end

    t_mean = vcat(mean(ens[1,:]), mean(ens[2,:]))
    g_mean = vcat(mean(ens_eval[1,:]), mean(ens_eval[2,:])) # hard coded dimensions :(

    # compute empirical covariance matrices
    # C_tt = 1/N * sum((ens[:,i] - t_mean)*(ens[:,i] - t_mean)' for i in 1:N)
    C_tg = 1/N * sum((ens[:,i] .- t_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
    C_gg = 1/N * sum((ens_eval[:,i] .- g_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
    
    Γ_ = Matrix(I, 2, 2)*0.1  # better way to handle I-type argument ??
    # construct array of updated ensemble members
    ens_new = ens[:,1] .+ C_tg * inv(Γ_ .+ C_gg) * (y_ .- ens_eval[:,1])  # dimension issues here
    for i in 2:N
        ens_new = hcat(ens_new, ens[:,i] .+ C_tg * inv(Γ_ .+ C_gg) * (y_ .- ens_eval[:,i]))
    end

    return ens_new
end

# solving the inverse problem
function run_eki(
    initial_ensemble::Any,
    G::Any,
    y::Any,
    Γ::Any,
    N_iterations::Int
    )
    # We now define prior distributions on the two parameters. For the amplitude,
    # we define a prior with mean 2 and standard deviation 1. It is
    # additionally constrained to be nonnegative. For the vertical shift we define
    # a Gaussian prior with mean 0 and standard deviation 5.
        prior = MvNormal(zeros(dim_output), I)
    
        #initial_ensemble = draw_initial(prior, N_ensemble)
    
        ensemble = initial_ensemble
        # carry out inversion
        for i in 1:N_iterations
            ensemble_new = eki_update(ensemble, G, y, Γ)
            ensemble = ensemble_new
        end
    
        return ensemble
end


dt = 0.01
trange = 0:dt:(2 * pi + dt)

function model(amplitude, vert_shift)
    phi = 2 * pi * rand(rng)
    return amplitude * sin.(trange .+ phi) .+ vert_shift
end

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

# define model G. Takes parameter vector, returns measured "observables"
function G(u)
    theta, vert_shift = u
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end

dim_output = 2
# noise on both measurements (max-min and mean) is dist ~N(0,0.1), INDEPENDENT of each other.
Γ = 0.1 * I
noise_dist = MvNormal(zeros(dim_output), Γ)

theta_true = [1.0, 7.0]
y = G(theta_true) .+ rand(noise_dist)

# We now generate the initial ensemble and set up the ensemble Kalman inversion.
N_ensemble = 5
N_iterations = 3
initial_ensemble = draw_initial(prior, N_ensemble)
final_ensemble = run_eki(initial_ensemble, G, y, Γ, N_iterations)



# To visualize the success of the inversion, we plot model with the true
# parameters, the initial ensemble, and the final ensemble.
plot(trange, model(theta_true...), c = :black, label = "Truth", legend = :bottomright, linewidth = 2)
plot!(
    trange,
    [model(initial_ensemble[:, i]...) for i in 1:N_ensemble],
    c = :red,
    label = ["Initial ensemble" "" "" "" ""],
)
plot!(trange, [model(final_ensemble[:, i]...) for i in 1:N_ensemble], c = :blue, label = ["Final ensemble" "" "" "" ""])

xlabel!("Time")


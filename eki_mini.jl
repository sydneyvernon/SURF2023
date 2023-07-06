using LinearAlgebra, Random
using Distributions

function draw_initial(
    prior::Distribution{},
    num
)
    a = rand(prior,num) # note 'amplitude' draw may be negative (conflicts w prior knowledge) but since this effectively causes a phase shift, doesnt matter here
    return a
end

function eki_update(
    ens::AbstractMatrix{},
    G_, 
    y_,
    Γ_
)
    N = size(ens)[2] # number of ensemble members
    N_param = size(ens)[1] # number of parameters (dim theta)
    Γ_ = I(2)*0.1

    # run G on ensemble members
    ens_eval = G_(ens[:,1]) # first output
    for i in 2:N 
        ens_eval = hcat(ens_eval, G_(ens[:,i]))
    end
    N_out = size(ens_eval)[1] # number of (summary) outputs (dim G(theta))

    t_mean = vcat([mean(ens[i,:]) for i in 1:N_param])
    g_mean = vcat([mean(ens_eval[i,:]) for i in 1:N_out])

    # compute empirical covariance matrices
    C_tg = 1/N * sum((ens[:,i] .- t_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
    C_gg = 1/N * sum((ens_eval[:,i] .- g_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
    
    # construct array of updated ensemble members
    ens_new = ens[:,1] .+ C_tg * inv(Γ_ .+ C_gg) * (y_ .- ens_eval[:,1])
    for i in 2:N
        ens_new = hcat(ens_new, ens[:,i] .+ C_tg * inv(Γ_ .+ C_gg) * (y_ .- ens_eval[:,i]))
    end
    return ens_new
end

# solving the inverse problem
function run_eki(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int
    ) 
        ensemble = initial_ensemble
        for i in 1:N_iterations
            ensemble_new = eki_update(ensemble, G, y, Γ)
            ensemble = ensemble_new
        end
        return ensemble
end
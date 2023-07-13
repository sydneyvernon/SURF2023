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
    
    # run G on ensemble members
    ens_eval_0 = G_(ens[:,1]) # first output
    N_out = size(ens_eval_0)[1] # number of (summary) outputs (dim G(theta))

    ens_eval = zeros(N_out, N)
    ens_eval[:,1] = ens_eval_0
    for i in 2:N 
        ens_eval[:,i] = G_(ens[:,i])
    end

    t_mean = mean(ens, dims=2)
    g_mean = mean(ens_eval, dims=2)

    # compute empirical covariance matrices
    C_tg = 1/N * sum((ens[:,i] .- t_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
    C_gg = 1/N * sum((ens_eval[:,i] .- g_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
    
    # construct array of updated ensemble members
    ens_new = zeros(N_param, N)
    for i in 1:N
        ens_new[:,i] = ens[:,i] .+ C_tg * inv(Γ_ .+ C_gg) * (y_ .- ens_eval[:,i])
    end
    return ens_new
end

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

# keep track of convergence
function run_eki(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn
    ) 
        conv = zeros(N_iterations+1, size(initial_ensemble)[2])
        for j in 1:size(initial_ensemble)[2]
            conv[1,j] = loss_fn(initial_ensemble[:,j])
        end

        ensemble = initial_ensemble
        for i in 1:N_iterations
            ensemble_new = eki_update(ensemble, G, y, Γ)
            ensemble = ensemble_new
            for j in 1:size(initial_ensemble)[2]
                conv[i+1,j] = loss_fn(ensemble[:,j])
            end
        end
        return ensemble, conv
end

# function run_eki_updatedloss(
#     initial_ensemble,
#     G, # model
#     y, # target or observed data
#     Γ, # covariance of measurement noise
#     N_iterations::Int,
#     loss_fn
#     ) 
#         conv = zeros(N_iterations+1, size(initial_ensemble)[2])
#         for j in 1:size(initial_ensemble)[2]
#             conv[1,j] = loss_fn(initial_ensemble[:,j])
#         end

#         ensemble = initial_ensemble
#         for i in 1:N_iterations
#             ensemble_new = eki_update(ensemble, G, y, Γ)
#             ensemble = ensemble_new
#             for j in 1:size(initial_ensemble)[2]
#                 conv[i+1,j] = loss_fn(ensemble[:,j])
#             end
#         end
#         return ensemble, conv
# end

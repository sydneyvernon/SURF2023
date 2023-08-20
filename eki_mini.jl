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
    Γ_,
    dt ## timestep
)
    N = size(ens)[2] # number of ensemble members
    N_param = size(ens)[1] # number of parameters (dim theta)
    
    # run G on ensemble members
    ens_eval_0 = G_(ens[:,1])
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
        ens_new[:,i] = ens[:,i] .+ dt * C_tg * inv(Γ_ .+ C_gg) * (y_ .- ens_eval[:,i])
    end
    return ens_new
end

function eki_update_momentum(
    ens::AbstractMatrix{},
    ens_prev,
    G_, 
    y,
    Γ_,
    k::Int, ## iteration number
    s, ## dt^2
    r,
    mean_update::Bool = false
)
    N = size(ens)[2] # number of ensemble members
    N_param = size(ens)[1] # number of parameters (dim theta)

    v = zeros(N_param, N)
    if mean_update
        for i in 1:N
            v[:,i] = ens[:,i] .+ (1-r/k)*(mean(ens .- ens_prev, dims=2))
        end
    else
        for i in 1:N
            v[:,i] = ens[:,i] .+ (1-r/k)*(ens[:,i] .- ens_prev[:,i])
        end
    end
    
    # run G on v ("momentum ensemble")
    v_eval_0 = G_(v[:,1]) # first output 
    N_out = size(v_eval_0)[1] # get dims of output
    v_eval = zeros(N_out, N)
    v_eval[:,1] = v_eval_0
    for i in 2:N 
        v_eval[:,i] = G_(v[:,i])
    end

    # compute empirical covariance matrices
    t_mean = mean(v, dims=2)
    g_mean = mean(v_eval, dims=2)
    C_tg = 1/N * sum((v[:,i] .- t_mean)*(v_eval[:,i] .- g_mean)' for i in 1:N)
    C_gg = 1/N * sum((v_eval[:,i] .- g_mean)*(v_eval[:,i] .- g_mean)' for i in 1:N)

    # construct array of updated ensemble members
    ens_new = zeros(N_param, N)
    if mean_update
        for i in 1:N
            ens_new[:,i] = v[:,i] .+ s*C_tg * inv(Γ_ .+ C_gg) * (y .- v_eval[:,i])
        end
    else
        for i in 1:N
            ens_new[:,i] = v[:,i] .+ s*C_tg * inv(Γ_ .+ C_gg) * (y .- v_eval[:,i])
        end
    end
    return ens_new, v
end

## DuJorShiSu22
function eki_update_momentum_highorder(
    ens::AbstractMatrix{},
    ens_prev,
    v_prev,
    G_, 
    y,
    Γ_,
    k::Int, ## iteration number
    s, ## dt^2
    alpha=4,
    beta=0.51,
    mean_update::Bool=false
)
    N = size(ens)[2] # number of ensemble members
    N_param = size(ens)[1] # number of parameters (dim theta)

    # run G on v_prev
    v_eval_0 = G_(v_prev[:,1])
    N_out = size(v_eval_0)[1] # number of (summary) outputs (dim G(theta))
    v_eval = zeros(N_out, N)
    v_eval[:,1] = v_eval_0
    for i in 2:N 
        v_eval[:,i] = G_(v_prev[:,i])
    end

    # compute empirical covariance matrices
    t_mean = mean(v_prev, dims=2)
    g_mean = mean(v_eval, dims=2)
    C_tg = 1/N * sum((v_prev[:,i] .- t_mean)*(v_eval[:,i] .- g_mean)' for i in 1:N)
    C_gg = 1/N * sum((v_eval[:,i] .- g_mean)*(v_eval[:,i] .- g_mean)' for i in 1:N)

    # gradients evaluated at current+past positions
    grad_f_current = zeros(N_param,N)  ## delta f(x_k+1)
    for i in 1:N
        grad_f_current[:,i] = (C_tg * inv(Γ_ .+ C_gg) * (y .- v_eval[:,i]))
    end

    # update u
    ens_new = zeros(N_param, N)
    for i in 1:N
        ens_new[:,i] = v_prev[:,i] .+ beta*s*grad_f_current[:,i] 
    end

    # update v
    v = zeros(size(ens))
    for i in 1:N
        v[:,i] = v_prev[:,i] .+ s*grad_f_current[:,i] .+ (k/(k+alpha))*(ens_new[:,i] - ens[:,i])
    end

    return ens_new, v
end

function run_eki(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn,
    dt ## timestep
    ) 
        conv = zeros(N_iterations+1)
        conv[1] = loss_fn(mean(initial_ensemble,dims=2))

        ensemble = initial_ensemble
        for i in 1:N_iterations
            ensemble_new = eki_update(ensemble, G, y, Γ, dt)
            ensemble = ensemble_new
            conv[i+1] = loss_fn(mean(ensemble,dims=2))
        end
        return ensemble, conv
end

function run_eki_momentum(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn,
    dt=1,
    r=3,
    mean_update=false, #  toggle "ensemble-mean" momentum approach
    give_mean_loss=true  #  if false: calculate mean loss (not loss of ensemble mean)
    ) 
        s = dt^2
        conv = zeros(N_iterations+1, size(initial_ensemble)[2])
        
        if give_mean_loss
            conv = zeros(N_iterations+1)
            conv_v = zeros(N_iterations+1) ## temporary experiment
        end
        if give_mean_loss
            conv[1] = loss_fn(mean(initial_ensemble, dims=2))
            conv_v[1] = loss_fn(mean(initial_ensemble, dims=2))  ## from our IC, v_0 = u_0.
        else
            for j in 1:size(initial_ensemble)[2]
                conv[1,j] = loss_fn(initial_ensemble[:,j])
            end
        end

        ensemble = initial_ensemble
        ens_prev = initial_ensemble
        for i in 1:N_iterations
            ensemble_new, v_new = eki_update_momentum(ensemble, ens_prev, G, y, Γ, i, s,r, mean_update)
            ens_prev = ensemble
            ensemble = ensemble_new

            # slightly different options for tracking convergence
            if give_mean_loss
                conv[i+1] = loss_fn(mean(ensemble, dims=2)) # loss of ens mean
                conv_v[i] = loss_fn(mean(v_new, dims=2)) # loss of "v" ens mean # i think the index shift makes sense?
            else
                for j in 1:size(initial_ensemble)[2]
                    conv[i+1,j] = loss_fn(ensemble[:,j])
                end
            end
        end
        return ensemble, conv, conv_v
end

function run_eki_momentum_highorder(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn,
    dt=1,
    alpha=4,
    beta=0.51,
    mean_update=false  # toggle "ensemble-mean" momentum approach
    ) 
        s = dt^2
        conv = zeros(N_iterations+1)
        conv[1] = loss_fn(mean(initial_ensemble, dims=2))

        ensemble = initial_ensemble
        ens_prev = initial_ensemble
        ens_new = initial_ensemble
        v_prev = initial_ensemble ## I.C. u_0 = v_0
        for i in 1:N_iterations
            ens_new, v = eki_update_momentum_highorder(ensemble, ens_prev, v_prev, G, y, Γ, i-1, s, alpha, beta, mean_update)
            ens_prev = ensemble
            ensemble = ens_new
            v_prev = v
            conv[i+1] = loss_fn(mean(ensemble, dims=2)) # loss of ens mean
        end
        return ensemble, conv
end

## variants to TRACK PARAM VALUES
function run_eki_momentum_tracked(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn,
    dt=1,
    r=3,
    mean_update::Bool=false
    ) 
        s = dt^2
        conv = zeros(N_iterations+1)

        ens_historical = zeros(N_iterations+1, size(initial_ensemble)[1], size(initial_ensemble)[2])
        ens_historical[1,:,:] = initial_ensemble

        conv[1] = loss_fn(mean(initial_ensemble, dims=2))
        
        ensemble = initial_ensemble
        ens_prev = initial_ensemble 
        for i in 1:N_iterations
            ensemble_new = eki_update_momentum(ensemble, ens_prev, G, y, Γ, i, s, r, mean_update)
            ens_prev = ensemble
            ensemble = ensemble_new
            conv[i+1] = loss_fn(mean(ensemble, dims=2))
            ens_historical[i+1,:,:] = ensemble
        end
        return ens_historical, conv
end

function run_eki_tracked(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn,
    dt ## timestep
    ) 
        ens_historical = zeros(N_iterations+1, size(initial_ensemble)[1], size(initial_ensemble)[2])
        ens_historical[1,:,:] = initial_ensemble
        conv = zeros(N_iterations+1)
        conv[1] = loss_fn(mean(initial_ensemble, dims=2))

        ensemble = initial_ensemble
        for i in 1:N_iterations
            ensemble_new = eki_update(ensemble, G, y, Γ, dt)
            ensemble = ensemble_new
            conv[i+1] = loss_fn(mean(ensemble, dims=2))
            ens_historical[i+1,:,:] = ensemble
        end
        return ens_historical, conv
end

function run_eki_momentum_constrained(
    initial_ensemble,
    G, # model
    y, # target or observed data
    Γ, # covariance of measurement noise
    N_iterations::Int,
    loss_fn,
    dt=1,
    r=3,
    mean_update::Bool=false
    ) 
        s = dt^2
        conv = zeros(N_iterations+1)
        conv[1] = loss_fn(mean(initial_ensemble, dims=2))
        
        ensemble = initial_ensemble
        ens_prev = initial_ensemble
        for i in 1:N_iterations
            if i<r ## CONSTRAIN MOMENTUM APPLICATION
                ensemble_new = eki_update(ensemble, G, y, Γ, dt)
            else
                ensemble_new = eki_update_momentum(ensemble, ens_prev, G, y, Γ, i, s, r, mean_update)
            end           
            ens_prev = ensemble
            ensemble = ensemble_new
            conv[i+1] = loss_fn(mean(ensemble, dims=2))
        end
        return ensemble, conv
end


# function run_eki(  # doesn't track convergence.
#     initial_ensemble,
#     G, # model
#     y, # target or observed data
#     Γ, # covariance of measurement noise
#     N_iterations::Int,
#     dt
#     ) 
#         ensemble = initial_ensemble
#         for i in 1:N_iterations
#             ensemble_new = eki_update(ensemble, G, y, Γ, dt)
#             ensemble = ensemble_new
#         end
#         return ensemble
# end


# function eki_update_momentum_means(
#     ens::AbstractMatrix{},
#     ens_prev,
#     G_, 
#     y,
#     Γ_,
#     k::Int, ## iteration number
#     s, ## dt^2,
#     r
# )
#     N = size(ens)[2] # number of ensemble members
#     N_param = size(ens)[1] # number of parameters (dim theta)
    
#     # run G on ensemble members
#     ens_eval_0 = G_(ens[:,1] .+ (1-r/k)*(mean(ens, dims=2) .- mean(ens_prev, dims=2))) # first output 
#     N_out = size(ens_eval_0)[1] # number of (summary) outputs (dim G(theta))
#     ens_eval = zeros(N_out, N)
#     ens_eval[:,1] = ens_eval_0
#     for i in 2:N 
#         ens_eval[:,i] = G_(ens[:,i] .+ (1-r/k)*(mean(ens, dims=2) .- mean(ens_prev, dims=2)))
#     end

#     # compute empirical covariance matrices
#     t_mean = mean(ens, dims=2)
#     g_mean = mean(ens_eval, dims=2)
#     C_tg = 1/N * sum((ens[:,i] .- t_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
#     C_gg = 1/N * sum((ens_eval[:,i] .- g_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)

#     # momentum follows the ensemble means
#     ens_mean = mean(ens, dims=2)
#     ens_mean_prev = mean(ens_prev, dims=2)

#     # construct array of updated ensemble members
#     v = zeros(N_param, N)
#     ens_new = zeros(N_param, N)

#     for i in 1:N
#         v[:,i] = ens[:,i] .+ (1-r/k)*(ens_mean .- ens_mean_prev)
#         ens_new[:,i] = v[:,i] .+ s*C_tg * inv(Γ_ .+ C_gg) * (y .- ens_eval[:,i])  ## C_gg should be evaluated WHERE
#         # if k < r
#         #     ens_new[:,i] = ens[:,i] .+ C_tg * inv(Γ_ .+ C_gg) * (y .- ens_eval[:,i]) # normal update step
#         # end
#     end
#     return ens_new
# end


# function run_eki_momentum_const(
#     initial_ensemble,
#     G, # model
#     y, # target or observed data
#     Γ, # covariance of measurement noise
#     N_iterations::Int,
#     loss_fn,
#     lambda,
#     s=1,
#     r=3
#     ) 
#         conv = zeros(N_iterations+1, size(initial_ensemble)[2])
#         for j in 1:size(initial_ensemble)[2]
#             conv[1,j] = loss_fn(initial_ensemble[:,j])
#         end
        
#         ensemble = initial_ensemble
#         ens_prev = zeros(size(initial_ensemble))
#         for i in 1:N_iterations  # const update doesnt actually need iteration tracker
#             ensemble_new = eki_update_momentum_const(ensemble, ens_prev, G, y, Γ, i, lambda, s,r)
#             ens_prev = ensemble
#             ensemble = ensemble_new
#             for j in 1:size(initial_ensemble)[2]
#                 conv[i+1,j] = loss_fn(ensemble[:,j])
#             end
#         end
#         return ensemble, conv
# end

# function eki_update_momentum_const(
#     ens::AbstractMatrix{},
#     ens_prev,
#     G_, 
#     y,
#     Γ_,
#     k::Int, ## iteration number
#     lambda, 
#     s, ## dt^2,
#     r
# )
#     N = size(ens)[2] # number of ensemble members
#     N_param = size(ens)[1] # number of parameters (dim theta)
    
#     # run G on ensemble members
#     ens_eval_0 = G_(ens[:,1]) # first output 
#     N_out = size(ens_eval_0)[1] # number of (summary) outputs (dim G(theta))
#     ens_eval = zeros(N_out, N)
#     ens_eval[:,1] = ens_eval_0
#     for i in 2:N 
#         ens_eval[:,i] = G_(ens[:,i])
#     end

#     # compute empirical covariance matrices
#     t_mean = mean(ens, dims=2)
#     g_mean = mean(ens_eval, dims=2)
#     C_tg = 1/N * sum((ens[:,i] .- t_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)
#     C_gg = 1/N * sum((ens_eval[:,i] .- g_mean)*(ens_eval[:,i] .- g_mean)' for i in 1:N)

#     # construct array of updated ensemble members
#     v = zeros(N_param, N)
#     ens_new = zeros(N_param, N)

#     for i in 1:N
#         v[:,i] = ens[:,i] .+ lambda*(ens[:,i] .- ens_prev[:,i])
#         ens_new[:,i] = v[:,i] .+ s*C_tg * inv(Γ_ .+ C_gg) * (y .- ens_eval[:,i])
#         if k < r
#             ens_new[:,i] = ens[:,i] .+ C_tg * inv(Γ_ .+ C_gg) * (y .- ens_eval[:,i]) # normal update step
#         end
#     end

#     return ens_new
# end
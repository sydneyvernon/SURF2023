using LinearAlgebra
using Zygote

## there are some hardcoded dimensions, need to fix

function gd_step(  # computes gradient using zygote
    current,
    loss_f,
    alpha
    )
    return current .- alpha .* gradient((t1,t2) -> loss_f([t1,t2]), current[1],current[2])
end

function gd_step(  # specified gradient
    current,
    loss_f,
    grad_f,
    alpha
    )
    return current .- alpha .* grad_f(current)
end

function run_gd(
    initial,
    loss_fn,
    alpha,
    N_iterations)
        conv = zeros(N_iterations+1) # keep track of convergence
        conv[1] = loss_fn(initial)

        current = initial
        for i in 1:N_iterations
            next = gd_step(current, loss_fn, alpha)
            current = next
            conv[i+1] = loss_fn(current) # save cost value at current point 
        end
        return (current, conv)

end

function run_gd(  # specified gradient
    initial,
    loss_fn,
    alpha,
    N_iterations,
    grad_fn
    )
        conv = zeros(N_iterations+1) # keep track of convergence
        conv[1] = loss_fn(initial)

        current = initial
        for i in 1:N_iterations
            next = gd_step(current, loss_fn, grad_fn, alpha)
            current = next
            conv[i+1] = loss_fn(current) # save cost value at current point 
        end
        return (current, conv)

end

function gd_step_momentum(  # computes gradient using zygote
    current,
    loss_f,
    alpha,
    beta,
    v,
    )
    v_new = beta*v .- alpha .* gradient((t1,t2) -> loss_f([t1,t2]), current[1],current[2])[:]
    return current .+ v_new, v_new 
end

function run_gd_momentum(
    initial,
    loss_fn,
    alpha,
    beta,
    N_iterations)
        conv = zeros(N_iterations+1) # keep track of convergence
        conv[1] = loss_fn(initial)

        current = initial
        v = zeros(size(initial))
        for i in 1:N_iterations
            next, v = gd_step_momentum(current, loss_fn, alpha, beta, v)
            current = next
            conv[i+1] = loss_fn(current) # save cost value at current point 
        end
        return (current, conv)
end

function gd_step_nesterov(  # computes gradient using zygote
    current,
    loss_f,
    alpha,
    beta,
    v,
    )
    v_new = beta*v .- alpha .* gradient((t1,t2) -> loss_f([t1,t2]), current[1]+beta*v[1],current[2]+beta*v[2])[:]
    return current .+ v_new, v_new 
end

function run_gd_nesterov(
    initial,
    loss_fn,
    alpha,
    beta,
    N_iterations)
        conv = zeros(N_iterations+1) # keep track of convergence
        conv[1] = loss_fn(initial)

        current = initial
        v = zeros(size(initial))
        for i in 1:N_iterations
            next, v = gd_step_nesterov(current, loss_fn, alpha, beta, v)
            current = next
            conv[i+1] = loss_fn(current) # save cost value at current point 
        end
        return (current, conv)
end

function run_gd_nesterov_tracked(  # tracks history of values
    initial,
    loss_fn,
    alpha,
    beta,
    N_iterations)
        historical = zeros(N_iterations, length(initial))
        conv = zeros(N_iterations+1) # keep track of convergence
        conv[1] = loss_fn(initial)

        current = initial
        v = zeros(size(initial))
        for i in 1:N_iterations
            historical[i, :] = current
            next, v = gd_step_nesterov(current, loss_fn, alpha, beta, v)
            current = next
            conv[i+1] = loss_fn(current) # save cost value at current point 
        end
        return (historical, conv)
end
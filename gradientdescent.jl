using LinearAlgebra
using Zygote

function gd_step(
    current,
    loss_f,
    alpha
    )
    return current .- alpha .* gradient((t1,t2) -> loss_f([t1,t2]), current[1],current[2])
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
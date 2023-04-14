module kooc
using DifferentialEquations, LinearAlgebra, ControlSystems, Plots
using Statistics
using Flux, Zygote, ForwardDiff
using Distributions, Random, Distributed
using AbstractDifferentiation

# dynamics
μ = -1.0
λ = 1.0
A = [μ 0.0;0.0 λ]
B = [0.0; 1.0]

function main()
    x0 = [-5.0, 5.0]
    s0 = [x0; 0.0]
    tspan = (0.0, 50.0)
    Q = I(2)
    R = 1
    K=lqr(A, B, Q, R)
    function ode_lqr!(ds, s, p, t)
        x = s[1:2]
        input = -K*x
        u = input[1]
        dx = A*x + [0.0;-λ*x[1]^2] + B*u
        dJ = x'*Q*x + u'*R*u
        ds .= [dx; dJ]
    end
    prob = ODEProblem(ode_lqr!, s0, tspan)
    sol=solve(prob, Tsit5())
    J = sol[end][end]
    @show J

    p1 = plot(layout=(2,2))
    plot!(p1, sol, idxs=1:2, xlabel="Time [s]", subplot=1)
    plot!(p1, sol, idxs=(1,2), subplot=2, xlims=(-5,5), ylims=(-5,5))

    Aa = [μ 0.0 0.0;0.0 λ -λ;0.0 0.0 2.0μ]
    Ba = [0.0;1.0;0.0]
    Qa =  [1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 0.0]
    Ra = 1.0
    Ka = lqr(Aa, Ba, Qa, Ra)
    function ode_klqr!(ds, s, p, t)
        x = s[1:3]
        input = -Ka*x
        u = input[1]
        dx = Aa*x + Ba*u
        dJ = x'Qa*x + u'*Ra*u
        ds .=[dx; dJ]
    end
    xa0 = [-5.0, 5.0, 25.0]
    sa0 = [xa0; 0.0]
    prob_a = ODEProblem(ode_klqr!, sa0, tspan)
    sol_a = solve(prob_a, Tsit5())
    plot!(p1, sol_a, idxs=1:2, xlabel="Time [s]", subplot=3)
    plot!(p1, sol_a, idxs=(1,2), subplot=4, xlims=(-5,5), ylims=(-5,5))
    Ja = sol_a[end][end]
    @show Ja
    return p1
end

function uniform(lb, ub, n; type=Float64)
    @assert ub >= lb
    return (ub-lb).*rand(type, n).+lb
end
function generate_train_data(n, nₓ ;lb=[-1.,-1.,-1.], ub=[1.,1.,1.], type=Float64)
    @assert length(lb) == length(ub)
    len = length(lb)
    x_data = zeros(len, n)
    y_data = zeros(type,(nₓ, n))    
    for i = 1:len
        x_data[i, :] = uniform(lb[i], ub[i], n, type=type)
    end    

    for i in 1:n
        x = x_data[1:2, i]
        u = x_data[3, i]
        y_data[:, i] = A*x + [0.0;-λ*x[1]^2] + B*u
    end
    
    return (type.(x_data), type.(y_data))
end


function jvp(func, primal, tangent)
    g(t) = func(primal + t * tangent)
    jvp_result = ForwardDiff.derivative(g, 0.0)
    return jvp_result
end


struct Observables{S<:AbstractArray}
    s::S
end
(m::Observables)(x) = m.s*x[1]^2
function train(;n=10::Integer, epoches=3::Integer, batchsize=1::Integer, lr=0.01, type=Float32, seed=2023)
    Random.seed!(seed)
    N = 1

    f(x, u) = A*x + [0.0f0; -λ*x[1]^2] + B*u
    nₓ = 2
    m = 1
    # generate data
    X, Y = generate_train_data(n, nₓ, type=type)
    train_set = Flux.DataLoader((X, Y), batchsize=batchsize, shuffle=true)

    Flux.@functor Observables # makes trainable
    
    phi = Observables([0.1f0])
    
    W = rand(Float32, nₓ+N, nₓ+N+m)
    ps = Flux.params(phi, W)
    
    function loss(xus, ys, phi, W, batchsize)
        
        function loss_each_batch(xandu, y, phi, W)
            x = xandu[1:2]
            u = xandu[3]        
            # dϕdt = jvp(phi, x, y)
            dt = 0.01
            dϕdt = (phi(x + dt*y) - phi(x))/dt
            er = [y; dϕdt] .- W*[x; phi(x); u]
            return sum(abs, er)
        end
        return sum(loss_each_batch.(xus, ys, [phi], [W]))/batchsize
    end
    
    opt_state = Flux.setup(Adam(lr), (phi, W))
    losses = Float32[]
    params = []
    for epoch in 1:epoches
        
        for (i, data) in enumerate(train_set)
            input, label = data            
            input = [input[:, i] for i = 1:batchsize]
            label = [label[:, i] for i = 1:batchsize]

            # calculate loss
            val = loss(input, label, phi, W, batchsize)
            push!(losses, val)
            push!(params, (phi.s[1], W[1,1], W[2,2], W[3,3], W[2,3]))
                    
            grads = Flux.gradient(phi, W) do ϕ, Weights
                loss(input, label, ϕ, Weights, batchsize)
            end
            
            
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end
            Flux.update!(opt_state, (phi, W), grads)
        end
        loss_val = losses[end]        
        println("Epoch :", epoch,"  Loss :",loss_val)
    end
    return losses, params, phi, W, X, Y
end
    function test(phi, W, rtol=1e-6, atol=1e-6)
        At = W[:, 1:3]
        Bt = W[:, 4]
        Qa =  [1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 0.0]
        Ra = 1.0
        K = lqr(At, Bt, Qa, Ra)

        x0 = [0.0; 0.0]
        s0 = [x0; 0.0]
        tspan = (0.0, 50.0)

        function ode_kooc!(ds, s, p, t)
            x = s[1:2]
            input = -K*[x;phi(x)]
            u = input[1]
            dx = A*x + [0.0;-λ*x[1]^2] + B*u
            dJ = x'*Qa[1:2,1:2]*x + u'*Ra*u
            ds .= [dx; dJ]
        end
        prob = ODEProblem(ode_kooc!, s0, tspan)

        # initial values
        x1 = -5.0:0.5:5.0
        x2 = -5.0:0.5:5.0
        x12 = Iterators.product(x1, x2)
        xis = vec(collect.(x12))
        n_traj = length(xis)
        function prob_func(prob, i, repeat)
            remake(prob, u0 = [xis[i];0.0])
        end
        # output_func(sol, i) = (sol[end, end], false)
        ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
        sim_kooc = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories= n_traj)

        Q = I(2)
        R = 1
        K=lqr(A, B, Q, R)
        function ode_lqr!(ds, s, p, t)
            x = s[1:2]
            input = -K*x
            u = input[1]
            dx = A*x + [0.0;-λ*x[1]^2] + B*u
            dJ = x'*Q*x + u'*R*u
            ds .= [dx; dJ]
        end
        prob = ODEProblem(ode_lqr!, x0, tspan)
        ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
        sim_lqr = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=n_traj)

        return (sim_kooc, sim_lqr)
    end
    
end
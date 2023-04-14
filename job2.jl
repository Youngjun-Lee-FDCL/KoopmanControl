module job2
using DifferentialEquations, LinearAlgebra, ControlSystems, Plots
using Statistics
using Flux, Zygote, ForwardDiff
using Distributions, Random, Distributed
# aerodynamic coefficients
aₐ = 0.3f0
aₘ = 40.44f0
aₙ = 19.373f0
bₘ = -64.015f0
bₙ = -31.023f0
cₘ = 2.922f0
cₙ = -9.717f0
dₘ = -11.803f0
dₙ = -1.948f0
eₘ = -1.719f0

Ca = aₐ
Cz(α, mach, δ) = aₙ*α^3 + bₙ* α * abs(α) + cₙ * (2.0f0 - mach/3.0f0)*α +dₙ *δ
Cm(α, mach, δ, q) = aₘ*α^3 + bₘ* α * abs(α) + cₘ * (-7.0f0 + 8.0f0*mach/3.0f0) *α +dₘ *δ + eₘ*q

S_ref = 0.0409f0 # m^2
d_ref = 0.2286f0 # m
mass = 284.02f0 # kg
I_yy = 247.439f0 # kg * m^2

ω = 150.0f0 # rad/s
ξ = 0.7f0 
δ_lim = 30.0f0 # deg

T0 = 288.16f0      # Temp. at Sea Level [K]
rho0 = 1.225f0     # Density [Kg/m^3]
L = 0.0065f0       # Lapse Rate [K/m]
R = 287.26f0       # Gas Constant J/Kg/K
gam = 1.403f0      # Ratio of Specific Heats
h_trop = 11000.0f0 # Height of Troposphere [m]
grav = 9.81f0      # Gravity [m/s/s]

d2r = π/180.0f0

function temp(h)
    h = min(max(h, 0.0f0), h_trop)
    return T0 - L * h
end

function density(h)
    T = temp(h)
    TT = T/T0
    PP = TT^(grav /L/R)
    if h < h_trop
        u = 1
    else
        u = exp(grav/R*(h_trop-mimimum(h,h_trop))/T)
    end
    return PP / TT * rho0 * u
end

function sonicSpeed(h)
    T = temp(h)
    return sqrt(gam * R * T)
end

function dyn(α, q, V, δ, h)
    Vₛ = sonicSpeed(h)
    ρ = density(h)    
    γ = 0.f0
    mach = V/Vₛ
    α_dot = ρ * V * S_ref / (2.0f0 * mass) * (Cz(α * d2r, mach, δ) * cos(α * d2r) + Ca * sin(α *d2r)) + grav/(V) * cos(γ)+ q
    α_dot = rad2deg(α_dot)
    q_dot = ρ * V^2 * S_ref * d_ref / (2.0 * I_yy) * Cm(α * d2r, mach, δ, q)
    V_dot = ρ * V^2 * S_ref / (2.0 * mass) * (Cz(α * d2r, mach, δ) * sin(α * d2r) - Ca * cos(α * d2r)) - grav * sin(γ)
    return [α_dot;q_dot;V_dot]
end

function dyn_full(α, q, mach, γ, h, δ)
    # α: deg, q: rad/s, δ: rad, mach : ND, h : m
    Vₛ = sonicSpeed(h)
    ρ = density(h)    
    
    α_dot = ρ * Vₛ * mach * S_ref / (2.0f0 * mass) * (Cz(α * d2r, mach, δ) * cos(α * d2r) + Ca * sin(α *d2r)) + grav/(Vₛ * mach) *cos(γ)+ q
    α_dot = rad2deg(α_dot)
    q_dot = ρ * Vₛ^2 * mach^2 * S_ref * d_ref / (2.0 * I_yy) * Cm(α * d2r, mach, δ, q)
    mach_dot = ρ * Vₛ * mach^2 * S_ref / (2.0 * mass) * (Cz(α * d2r, mach, δ) * sin(α * d2r) - Ca * cos(α * d2r)) - grav/Vₛ * sin(γ)
    γ_dot = -ρ * Vₛ * mach * S_ref / (2.0 * mass) * (Cz(α * d2r, mach, δ) * cos(α * d2r) + Ca * sin(α * d2r)) - grav/(Vₛ * mach) * cos(γ)
    h_dot = mach * Vₛ * sin(γ)
    return [α_dot;q_dot;mach_dot;γ_dot;h_dot]
end

function get_linearized_model(x0, u0, h)
    A = Zygote.jacobian((x)->dyn(x[1], x[2], x[3], u0, h), x0)[1]
    B = Zygote.jacobian(δ -> dyn(x0[1], x0[2], x[3], δ, h), u0)[1]
    return A, B
end

function uniform(lb, ub, n; type=Float64)
    @assert ub >= lb
    return (ub-lb).*rand(type, n).+lb
end

function generate_train_data(n, nₓ, h;lb=[-1.,-1.,2.,-1.], ub=[1.,1.,4.,1.], type=Float64, noise_std=nothing)    @assert length(lb) == length(ub)
    len = length(lb)
    x_data = zeros(len, n)
    y_data = zeros(type,(nₓ, n))    
    for i = 1:len
        x_data[i, :] = uniform(lb[i], ub[i], n, type=type)
    end    

    for i in 1:n
        x = x_data[1:nₓ, i]
        u = x_data[nₓ+1, i]
        y_data[:, i] = dyn(x[1], x[2], x[3], u[1], h)
    end
    if noise_std === nothing
        return (type.(x_data), type.(y_data))
    else
        y_data .= y_data .+ noise_std*abs.(y_data).*randn(size(y_data))
        return (type.(x_data), type.(y_data))
    end
end

function test_dyn()
    h = 10000.0f0
    ode_fn(ds, s, p, t) = begin
        α, q, V = s
        δ = deg2rad(10.f0)
        ds .= dyn(α, q, V, δ, h)
    end
    Vₛ = sonicSpeed(h)
    mach = 3.0f0
    V = Vₛ * mach
    x0 = [10.f0; 0.0f0; V]
    tspan = (0.0f0, 10.f0)
    prob = ODEProblem(ode_fn, x0, tspan)
    sol = solve(prob, Tsit5())
end

function sat(x, lim)
    return max(min(x, lim),-lim)
end
struct obs{S<:AbstractArray}
    s::S
end
function (m::obs)(x) 
    Vₛ = sonicSpeed(10000.f0)
    x_n = x .* [d2r; 1.0f0; 1/Vₛ]
    return [x_n; sin(x_n[1]); cos(x_n[1]); x_n[1]^3; x_n[1]*abs(x_n[1]); x_n[3]^2; x_n[3]*cos(x_n[1]); x_n[3]*sin(x_n[2])]
end

function train(;n=10::Integer, epoches=3::Integer, batchsize=1, lr=0.05, type=Float32, seed=2023, hidden_layer=[64;32;16], N=3, act=relu)
    Random.seed!(seed)
    nₓ = 3
    nᵤ = 1

    # initial conditions
    h = 10000.0f0
    Vₛ = sonicSpeed(h)

    # generate data
    lb = [-30.0f0; -deg2rad(10.0f0); 2.0f0*Vₛ; -deg2rad(30.0f0)] # α: deg, q: rad/s, mach: ND, δ: rad/s
    ub = [30.0f0; deg2rad(10.0f0); 4.0f0*Vₛ; deg2rad(30.0f0)]
    X, Y = generate_train_data(n, nₓ, h, lb=lb, ub=ub, type=type)
    train_set = Flux.DataLoader((X,Y), batchsize=batchsize, shuffle=true)

    #generate lifting function
    Flux.@functor obs
    ϕ₀ = obs([1.0f0])
    ϕ = Chain(ϕ₀, Dense(10 => hidden_layer[1], act),
              Dense(hidden_layer[1] => hidden_layer[2], act),
              Dense(hidden_layer[2] => hidden_layer[3], act),
              Dense(hidden_layer[3] => N),
              x->200.0f0.*x)
    W = randn(Float32, nₓ+N, nₓ+nᵤ+N)

    function loss(xus, ys, ϕ, W, batchsize)
        function loss_each_batch(xu, y, phi, W)
            x = xu[1:nₓ]
            u = xu[nₓ+nᵤ]
            dt = 0.05            
            dϕdt = (phi(x + y*dt) - phi(x))/dt
            er = [y;dϕdt] .- W*[x;phi(x);u]
            return sum(abs, er)
        end
        pa, = Flux.destructure(ϕ)
        λ1 = 0.0001f0
        
        return sum(loss_each_batch.(xus, ys, [ϕ], [W]))/batchsize + λ1*sum(abs, pa)
    end

    opt_state = Flux.setup(Adam(lr), (ϕ, W))
    losses = Float32[]
    params = []
    for epoch in 1:epoches
        for (i, data) in enumerate(train_set)
            input, label = data
            input = [input[:, i] for i = 1:batchsize]
            label = [label[:, i] for i = 1:batchsize]
            # calculate loss
            val = loss(input, label, ϕ, W, batchsize)
            push!(losses, val)
            push!(params, W[1,2])

            grads = Flux.gradient(ϕ, W) do ϕ, weights
                loss(input, label, ϕ, weights, batchsize)
            end
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end
            
            Flux.update!(opt_state, (ϕ,W), grads)
        end
        loss_val = losses[end]
        println("Epoch :", epoch, "  Loss :",loss_val)
    end
    return losses, params, ϕ, W, X, Y
end

function test(ϕ,W,α_cmd)
    q_cmd = 0.0f0
    h = 10000.0f0
    Vₛ = sonicSpeed(h)
    A = W[:, 1:end-1]
    B = W[:, end]
    Q = zeros(size(A))
    Q[1,1] = 10
    Q[2,2] = 30
    R = 20
    K = lqr(A,B,Q,R)
    @show K
    ode_fn(ds, s, p, t) = begin 
        x = s[1:3]
        α, q, mach = x
        e = [α-α_cmd; q-q_cmd; mach]
        δ = sat(-(K*[e;ϕ(e)])[1], deg2rad(δ_lim))
        dx = dyn(α, q, mach, δ, h)
        dJ = e'*Q[1:2,1:2]*e +δ'*R*δ
        ds .= [dx;dJ]
    end
    x0 = [0.f0; 1.0f0; 3.0f0*Vₛ]
    s0 = [x0;0.0]
    tspan = (0.0f0, 2.f0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    e_hist = sol[1:3,:] .- [α_cmd;q_cmd;0.0f0]
    e_hist = [e_hist[:,i] for i = 1:size(e_hist)[2]]
    ϕ_e_hist = [ [e_hist[i];ϕ(e_hist[i])] for i = 1:length(e_hist)]
    u_hist = sat.(dot.([K], ϕ_e_hist), deg2rad(δ_lim))
    println("Cost :", sol.u[end][end])
    return sol, rad2deg.(vec(u_hist))
end

end
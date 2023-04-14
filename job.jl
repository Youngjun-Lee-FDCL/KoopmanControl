module job
using DifferentialEquations, LinearAlgebra, ControlSystems, Plots
using Statistics, MatrixEquations
using Flux, Zygote, ForwardDiff
using Distributions, Random, Distributed
using DataFrames, DelimitedFiles
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

Ca(;bias=1.0f0) = bias*aₐ
Cz(α, mach, δ;bias=1.0f0) = (bias)*(aₙ*α^3 + bₙ* α * abs(α) + cₙ * (2.0f0 - mach/3.0f0)*α +dₙ *δ)
Cm(α, mach, δ, q;bias=1.0f0) = (bias)*(aₘ*α^3 + bₘ* α * abs(α) + cₘ * (-7.0f0 + 8.0f0*mach/3.0f0) *α +dₘ *δ + eₘ*q)

S_ref = 0.0409f0 # m^2
d_ref = 0.2286f0 # m
mass(;bias=1.0f0) = 284.02f0*bias # kg
I_yy(;bias=1.0f0) = 247.439f0*bias # kg * m^2

ω = 150.0f0 # rad/s
ξ = 0.7f0 
δ_lim = 30.0f0 # fin deflection limit [deg]
δ_dot_lim = 5000.0f0 # fin deflection rate limit [deg/s]

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
        u = exp(grav/R*(h_trop-min(h,h_trop))/T)
    end
    return PP / TT * rho0 * u
end

function sonicSpeed(h)
    T = temp(h)
    return sqrt(gam * R * T)
end

function dyn(α, q, δ, mach, h)
    # α: deg, q: rad/s, δ: rad, mach : ND, h : m
    Vₛ = sonicSpeed(h)
    ρ = density(h)    
    α_dot = ρ * Vₛ * mach * S_ref / (2.0f0 * mass()) * 
            (Cz(α * d2r, mach, δ) * cos(α * d2r) + Ca() * sin(α *d2r)) + grav/(Vₛ * mach) + q;
    α_dot = rad2deg(α_dot)
    q_dot = ρ * Vₛ^2 * mach^2 * S_ref * d_ref / (2.0 * I_yy()) * Cm(α * d2r, mach, δ, q)
    return [α_dot;q_dot]
end

function dyn_mach(α, q, mach, δ, h)
    Vₛ = sonicSpeed(h)
    ρ = density(h)    
    α_dot = ρ * Vₛ * mach * S_ref / (2.0f0 * mass()) * (Cz(α * d2r, mach, δ) * cos(α * d2r) + Ca() * sin(α *d2r)) + grav/(Vₛ * mach) *cos(γ)+ q
    α_dot = rad2deg(α_dot)
    q_dot = ρ * Vₛ^2 * mach^2 * S_ref * d_ref / (2.0 * I_yy()) * Cm(α * d2r, mach, δ, q)
    mach_dot = ρ * Vₛ * mach^2 * S_ref / (2.0 * mass()) * (Cz(α * d2r, mach, δ) * sin(α * d2r) - Ca() * cos(α * d2r)) - grav/Vₛ * sin(γ)
    return [α_dot;q_dot;mach_dot]
end

function dyn_full(α, q, mach, γ, h, δ_cmd, δ, δ_dot; aero_bias=[1.0f0,1.0f0,1.0f0], phy_bias=[1.0f0,1.0f0])
    # α: deg, q: rad/s, δ: rad, mach : ND, h : m
    Vₛ = sonicSpeed(h)
    ρ = density(h)    
    bias_ca = aero_bias[1]
    bias_cn = aero_bias[2]
    bias_cm = aero_bias[3]
    bias_ma = phy_bias[1]
    bias_in = phy_bias[2]
    α_dot = ρ * Vₛ * mach * S_ref / (2.0f0 * mass(bias=bias_ma)) * (Cz(α * d2r, mach, δ,bias=bias_cn) * cos(α * d2r) + Ca(bias=bias_ca) * sin(α *d2r)) + grav/(Vₛ * mach) *cos(γ)+ q
    α_dot = rad2deg(α_dot)
    q_dot = ρ * Vₛ^2 * mach^2 * S_ref * d_ref / (2.0 * I_yy(bias=bias_in)) * Cm(α * d2r, mach, δ, q,bias=bias_cm)
    mach_dot = ρ * Vₛ * mach^2 * S_ref / (2.0 * mass(bias=bias_ma)) * (Cz(α * d2r, mach, δ,bias=bias_cn) * sin(α * d2r) - Ca(bias=bias_ca) * cos(α * d2r)) - grav/Vₛ * sin(γ)
    γ_dot = -ρ * Vₛ * mach * S_ref / (2.0 * mass(bias=bias_ma)) * (Cz(α * d2r, mach, δ,bias=bias_cn) * cos(α * d2r) + Ca(bias=bias_ca) * sin(α * d2r)) - grav/(Vₛ * mach) * cos(γ)
    h_dot = mach * Vₛ * sin(γ)
    (δ_dot, δ_ddot) = actuator(δ, δ_dot, δ_cmd)
    return [α_dot;q_dot;mach_dot;γ_dot;h_dot;δ_dot;δ_ddot]
end

function actuator(δ, δ_dot, δ_cmd)
    ω = 150.0f0
    ξ = 0.707f0
    δ_ddot = -ω^2*δ -2.0f0*ξ*ω*δ_dot + ω^2*δ_cmd
    return (δ_dot, δ_ddot)
end
function get_linearized_model(x0,u0,mach, h)
    A = Zygote.jacobian((x)->dyn(x[1], x[2], u0, mach, h), x0)[1]
    B = Zygote.jacobian(δ -> dyn(x0[1], x0[2], δ, mach, h), u0)[1]
    return A, B
end

function uniform(lb, ub, n; type=Float64)
    @assert ub >= lb
    return (ub-lb).*rand(type, n).+lb
end

function generate_train_data(n, nₓ, mach, h;lb=[-1.,-1.,-1.], ub=[1.,1.,1.], type=Float64, noise_std=nothing)
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
        y_data[:, i] = dyn(x[1], x[2], u[1], mach, h)
        # dyn_full()
    end
    if noise_std === nothing
        return (type.(x_data), type.(y_data))
    else
        y_data .= y_data .+ noise_std*abs.(y_data).*randn(size(y_data))
        return (type.(x_data), type.(y_data))
    end
end

function test_dyn()
    ode_fn(ds, s, p, t) = begin
        α, q = s
        δ = deg2rad(10.f0)
        h = 10000.0f0
        mach = 3.f0
        ds .= dyn(α, q, δ, mach, h)
    end
    x0 = [10.f0; 0.0f0]
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
    x_n = x .* [d2r; 1.0f0]
    return [sin(x_n[1]); cos(x_n[1]); x_n[1]^3; x_n[1]*abs(x_n[1])]
end

function stepCmd(t, stepTimes, stepValues)
    cmd = 0
    idx = sum(t .>= stepTimes)
    if idx == 0
        return cmd
    end
    cmd = stepValues[idx]
    return cmd
end
function train(;n=10::Integer, epoches=3::Integer, batchsize=1, lr=0.01, type=Float32, seed=2023, hidden_layer=[64;32;16], N=5, act=relu)
    # ls, pa, ϕ, W, X, Y = job.train(n=1000, epoches=1000, batchsize=200, N=2, lr=0.05, hidden_layer=[128,64,32])
    Random.seed!(seed)
    nₓ = 2 # state dimension
    nᵤ = 1 # input dimension

    # initial conditions
    mach = 3.0f0
    h = 10000.0f0

    # generate data
    lb = [-30.0f0; -deg2rad(10.0f0); -deg2rad(30.0f0)] # α: deg, q: rad/s, δ: rad/s
    ub = [30.0f0; deg2rad(10.0f0); deg2rad(30.0f0)]
    X, Y = generate_train_data(n, nₓ, mach, h, lb=lb, ub=ub, type=type, noise_std=0.1f0)
    train_set = Flux.DataLoader((X,Y), batchsize=batchsize, shuffle=true)
    
    # generate lifting function
    Flux.@functor obs
    ϕ₀ = obs([1.0f0])
    ϕ = Chain(ϕ₀, Dense(4 => hidden_layer[1], act, bias=true),
              Dense(hidden_layer[1] => hidden_layer[2], act, bias=true), 
              Dense(hidden_layer[2] => hidden_layer[3], act, bias=true),
              Dense(hidden_layer[3] => N, bias=true))
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
    mach = 3.f0
    A = W[:, 1:end-1]
    B = W[:, end]
    Q = zeros(size(A))
    Q[1,1] = 10
    Q[2,2] = 30
    R = 20
    K = lqr(A,B,Q,R)
    @show K
    ode_fn(ds, s, p, t) = begin 
        x = s[1:2]
        α, q = x
        e = [α-α_cmd; q-q_cmd]
        δ = sat(-(K*[e;ϕ(e)])[1], deg2rad(δ_lim))
        dx = dyn(α, q, δ, mach, h)
        dJ = e'*Q[1:2,1:2]*e +δ'*R*δ
        ds .= [dx;dJ]
    end
    x0 = [0.f0; 1.0f0]
    s0 = [x0;0.0]
    tspan = (0.0f0, 2.f0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    e_hist = sol[1:2,:] .- [α_cmd;q_cmd]
    e_hist = [e_hist[:,i] for i = 1:size(e_hist)[2]]
    ϕ_e_hist = [ [e_hist[i];ϕ(e_hist[i])] for i = 1:length(e_hist)]
    u_hist = sat.(dot.([K], ϕ_e_hist), deg2rad(δ_lim))
    println("Cost :", sol.u[end][end])
    return sol, rad2deg.(vec(u_hist))
end

function test_lqr(α_cmd)
    q_cmd = 0.0f0
    x0 = [0.0f0, 0.0f0]
    u0 = 0.0f0
    mach = 3.0f0
    h = 10000.0f0
    A, B = get_linearized_model(x0, u0, mach, h)
    Q = [10.0 0.0;0.0 30.0]
    R= 20.0
    K = lqr(A,B,Q,R)
    @show K 
    ode_fn(ds, s, p, t) = begin
        x = s[1:2]
        α, q = x
        e = [α-α_cmd;q-q_cmd]
        δ = sat(-(K*e)[1], deg2rad(δ_lim))
        dx = dyn(α, q, δ, mach, h)
        dJ = e'*Q*e + δ'*R*δ
        ds .= [dx;dJ]
    end
    x0 = [0.0f0; 1.0f0]
    s0 = [x0; 0.0]
    tspan = (0.0f0, 2.0f0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    u_hist = sat.(K*(sol[1:2,:] .- [α_cmd;q_cmd]), deg2rad(δ_lim))
    println("Cost :", sol.u[end][end])
    return sol,rad2deg.(vec(u_hist))
end

function test_lqr_full(;h₀=10000.f00, mach₀=3.0f0, aero_bias=[1.0f0,1.0f0,1.0f0], phy_bias=[1.0f0, 1.0f0], plot_on=true)
    α₀ = 0.0f0; q₀ = 0.0f0; γ₀ = 0.0f0;
    δ₀ = 0.0f0; δdot₀ = 0.0f0;

    α_cmd(t) = stepCmd(t, [1.0, 3,0,  4.0, 5.0,  6.0,   9.0,   10.5,  13.0,  15.0,  16.0],
                          [-10.0, -10., 15.0, 15.0, -15.0, -15.0, 0.0, 0.0, -15.0, 0.0])
    q_cmd(t) = 0.0f0
    
    A, B = get_linearized_model([α₀, q₀], δ₀, mach₀, h₀) 
    Q = [10.0 0.0;0.0 30.0]
    R= 100.0
    K = lqr(A,B,Q,R)
    ode_fn(ds, s, p, t) =begin
        x = s[1:7]
        α, q, mach, γ, h, δ, δ_dot = x
        e =[α-α_cmd(t);q-q_cmd(t)]
        δ_cmd = sat(-(K*e)[1], deg2rad(δ_lim))
        dx = dyn_full(α, q, mach, γ, h, δ_cmd, δ, δ_dot, aero_bias=aero_bias,phy_bias=phy_bias)
        dJ = e'*Q*e + δ'*R*δ
        ds .= [dx;dJ]
    end
    x0 = [α₀, q₀, mach₀, γ₀, h₀, δ₀, δdot₀]
    s0 = [x0; 0.0f0]
    tspan = (0.0f0, 12.0f0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    α_cmd_hist = α_cmd.(sol.t)'
    q_cmd_hist = q_cmd.(sol.t)'
    u_hist = sat.(-K*(sol[1:2,:] .- [α_cmd_hist; q_cmd_hist]), deg2rad(δ_lim))
    u_hist = rad2deg.(vec(u_hist))
    println("Cost :", sol.u[end][end])
    if plot_on == true
        p11 = plot(sol.t,α_cmd_hist', layout=(3,1), subplot=1, label="cmd")
        plot!(p11, sol, idxs=(0,1), label="act")
        p12 = plot(p11, sol, idxs=(0, 3), subplot=2)
        p13 = plot(p12, sol.t, u_hist, subplot=3)
        plot!(p13, sol.t, rad2deg.(sol[6,:]), subplot=3)
        display(p13)
    end
    return sol, u_hist, α_cmd_hist'
end

function test_full(ϕ,W, α_cmd)
    α₀ = 0.0f0; q₀ = 0.0f0; mach₀ = 3.0f0; γ₀ = 0.0f0;  h₀ = 10000.0f0;
    δ₀ = 0.0f0; δdot₀ = 0.0f0;

    q_cmd = 0.0f0
    A = W[:, 1:end-1]
    B = W[:, end]
    Q = zeros(size(A))
    Q[1,1] = 10
    Q[2,2] = 30
    R = 20
    K = lqr(A,B,Q,R)
    @show K
    ode_fn(ds, s, p, t)= begin
        x = s[1:7]
        α, q, mach, γ, h, δ, δ_dot = x
        e = [α - α_cmd; q - q_cmd]
        δ_cmd = sat(-(K*[e;ϕ(e)])[1], deg2rad(δ_lim))
        dx = dyn_full(α, q, mach, γ, h, δ_cmd, δ, δ_dot)
        dJ = e'*Q[1:2,1:2]*e +δ'*R*δ
        ds .= [dx;dJ]
    end
    x0 = [α₀, q₀, mach₀, γ₀, h₀, δ₀, δdot₀]
    s0 = [x0; 0.0f0]
    tspan = (0.0f0, 2.0f0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    e_hist = sol[1:2,:] .- [α_cmd;q_cmd]
    e_hist = [e_hist[:,i] for i = 1:size(e_hist)[2]]
    ϕ_e_hist = [ [e_hist[i];ϕ(e_hist[i])] for i = 1:length(e_hist)]
    u_hist = sat.(dot.([K], ϕ_e_hist), deg2rad(δ_lim))
    println("Cost :", sol.u[end][end])
    return sol, rad2deg.(vec(u_hist))
end

function test_lqrpi_full(α_cmd)
    α₀ = 0.0f0; q₀ = 0.0f0; mach₀ = 3.0f0; γ₀ = 0.0f0;  h₀ = 10000.0f0;
    δ₀ = 0.0f0; δdot₀ = 0.0f0;
    
    A, B = get_linearized_model([α₀, q₀], δ₀, mach₀, h₀) 
    Q = [20.f0 0.0f0 0.0f0; 
         0.0f0 0.1f0 0.0f0; 
         0.0f0 0.0f0 10.0f0]
    R = 20.0
    C = [1.0f0; 0.0f0]
    D = 0.0f0;
    Aa = [0.0f0 C';zeros(2) A]
    Ba = [D; B]
    K = lqr(Aa,Ba,Q,R)
    @show K
    ode_fn(ds, s, p, t) =begin
        e_αI = s[1]
        x = s[2:8]
        α, q, mach, γ, h, δ, δ_dot = x
        e =[e_αI; α; q]
        δ_cmd = sat(-(K*e)[1], deg2rad(δ_lim))
        de = α - α_cmd
        dx = dyn_full(α, q, mach, γ, h, δ_cmd ,δ, δ_dot)
        dJ = e'*Q*e + δ'*R*δ
        ds .= [de; dx; dJ]
    end
    x0 = [α₀, q₀, mach₀, γ₀, h₀, δ₀, δdot₀]
    s0 = [0.0f0; x0; 0.0f0]
    tspan = (0.0f0, 2.0f0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    u_hist = sat.(K*(sol[1:3,:]), deg2rad(δ_lim))
    println("Cost :", sol.u[end][end])
    return sol, rad2deg.(vec(u_hist))
end

function test_koopman_full(ϕ,W;h₀ = 10000.0f0, mach₀ = 3.5f0, aero_bias=[1.0,1.0,1.0], phy_bias=[1.0, 1.0], plot_on=true)
    α₀ = 0.0; q₀ = 0.0;  γ₀ = 0.0;  
    δ₀ = 0.0; δdot₀ = 0.0;

    α_cmd(t) = stepCmd(t, [1.0, 3,0,  4.0, 5.0,  6.0,   9.0,   10.5,  13.0,  15.0,  16.0],
                          [-10.0, -10., 15.0, 15.0, -15.0, -15.0, 0.0, 0.0, -15.0, 0.0])
    A = W[:, 1:end-1]
    B = W[:, end]
    (r,c) = size(A)
    Q = zeros(r+1, c+1)
    C = zeros(c)
    C[1] = 1.0
    D = 0.0f0
    Q[1,1] = 20.0
    Q[2,2] = 0.0
    Q[3,3] = 10.0
    R = 20.0

    Aa = [0.0 C';zeros(r) A]
    Ba = [D; B]

    K = lqr(Aa,Ba,Q,R)
    ode_fn(ds, s, p, t)= begin
        e_αI = s[1]
        x = s[2:8]
        α, q, mach, γ, h, δ, δdot = x
        e = [e_αI; α; q]
        δ_cmd = sat(-(K*[e;ϕ(e[2:3])])[1], deg2rad(δ_lim))
        de = α - α_cmd(t)
        dx = dyn_full(α, q, mach, γ, h, δ_cmd, δ, δdot, aero_bias=aero_bias, phy_bias=phy_bias)
        dJ = e'*Q[1:3,1:3]*e +δ'*R*δ
        ds .= [de; dx; dJ]
    end
    x0 = [α₀, q₀, mach₀, γ₀, h₀, δ₀, δdot₀]
    s0 = [0.0; x0; 0.0]
    tspan = (0.0, 12.0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    e_hist = sol[1:3,:]
    e_hist = [e_hist[:,i] for i = 1:size(e_hist)[2]]
    ϕ_e_hist = [ [e_hist[i];ϕ(e_hist[i][2:3])] for i = 1:length(e_hist)]
    u_hist = sat.(dot.([-K], ϕ_e_hist), deg2rad(δ_lim))
    u_hist = rad2deg.(vec(u_hist))
    α_cmd_hist = α_cmd.(sol.t)
    println("Cost :", sol.u[end][end])
    if plot_on == true
        p11 = plot(sol.t,α_cmd_hist, layout=(3,1), subplot=1, label="cmd")
        plot!(p11, sol, idxs=(0,2), label="act")
        p12 = plot(p11, sol, idxs=(0, 4), subplot=2)
        p13 = plot(p12, sol.t, u_hist, subplot=3)
        plot!(p13, sol.t, rad2deg.(sol[7,:]), subplot=3)
        display(p13)
    end
    return sol, u_hist, α_cmd_hist
end

function test_mrac_full()
    α₀ = 0.0; q₀ = 0.0; mach₀ = 3.5; γ₀ = 0.0;  h₀ = 10000.0;
    δ₀ = 0.0; δdot₀ = 0.0
    α_cmd(t) = stepCmd(t, [0.0, 1.0, 2.0,  3.0, 4.0,  5.0,   6.0,   7.0,  8.0,  9.0,  10.0],
                          [20., 0.0, -10., 5.0, 20.0, -20.0, -15.0, 10.0, 20.0, -15.0, 0.0])
    # α_cmd(t) = 10*sin(t)

    Am, Bm = get_linearized_model([α₀, q₀], δ₀, mach₀, h₀) 
    Q_lqr = [20. 0.0 0.0; 
             0.0 0.2 0.0; 
             0.0 0.0 10.0]
    R_lqr = 1.0
    C = [1.0; 0.0]
    D = 0.0;
    A = [0.0 C';zeros(2) Am]
    B = [D; Bm]
    Bᵣ = [-1.0;0.0;0.0]
    K_lqr = lqr(A, B, Q_lqr, R_lqr)
    Aᵣ = A - B*K_lqr

    Qᵣ = 100.0*I(3)
    Γᵤ = 0.001
    Γ_Θ = [0.00001 0.0;0.0 0.0001]
    Pᵣ = lyapc(Aᵣ',Qᵣ)

    ode_fn(ds, s, p, t) =begin
        x = s[1:8] # including ∫ (α-α_cmd) dt
        xₚ = s[2:8]
        xᵣ = s[9:11]
        K̂ₓ = s[12]
        Θ̂ₓ = s[13:14]
        e = x[1:3] - xᵣ
        Φ = x[2:3]
        
        α, q, mach, γ, h, δ, δdot = xₚ
        uᵦ = -(K_lqr*x[1:3])[1]
        u = (1.0 - K̂ₓ)*uᵦ - Θ̂ₓ'*Φ
        δ_cmd = sat(u, deg2rad(δ_lim))
        de = α - α_cmd(t)
        dx = dyn_full(α, q, mach, γ, h, δ_cmd, δ, δdot)
        dxᵣ = Aᵣ*xᵣ + Bᵣ*α_cmd(t)
        K̂ₓ_dot = Γᵤ*uᵦ*e'*Pᵣ*B 
        Θ̂ₓ_dot = Γ_Θ*Φ*e'*Pᵣ*B   
        dJ = e'*Q_lqr*e + δ'*R_lqr*δ
        ds .= [de; dx; dxᵣ; K̂ₓ_dot; Θ̂ₓ_dot; dJ]
    end
    x0 = [α₀, q₀, mach₀, γ₀, h₀, δ₀, δdot₀]
    s0 = [0.0; x0; 0.0; α₀; q₀; 0.0; 0.0; 0.0; 0.0]
    tspan = (0.0, 12.0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    u_hist = []
    for sol = sol.u
        x = sol[1:3]
        K̂ₓ = sol[12]
        Θ̂ₓ = sol[13:14]
        Φ = x[2:3]
        uᵦ = -(K_lqr*x)[1]
        u = rad2deg(sat((1.0 - K̂ₓ)*uᵦ - Θ̂ₓ'*Φ, deg2rad(δ_lim)))
        push!(u_hist, u)
    end
    println("Cost :", sol.u[end][end])

    y_cmd_hist = α_cmd.(sol.t)
    p11 = plot(sol.t,y_cmd_hist, layout=(2,3), subplot=1, label="cmd")
    plot!(p11, sol, idxs=(0,8), label="ref")
    plot!(p11, sol, idxs=(0,2), label="act")
    p12 = plot(p11, sol, idxs=(0,10), subplot=2)
    p21 = plot(p12, sol, idxs=(0,11:12),subplot=3)
    p22 = plot(p21, sol, idxs=(0, 4), subplot=4)
    p31 = plot(p22, sol.t, u_hist,subplot=5)
    display(p31)
    return sol, rad2deg.(u_hist)
end

function test_train_mrac_full(ϕ, W; h₀ = 10000.0, mach₀ = 3.5, aero_bias=[1.0,1.0,1.0], phy_bias=[1.0,1.0], plot_on=true)
    α₀ = 0.0; q₀ = 0.0; γ₀ = 0.0;
    δ₀ = 0.0; δdot₀ = 0.0;

    α_cmd(t) = stepCmd(t, [1.0, 3,0,  4.0, 5.0,  6.0,   9.0,   10.5,  13.0,  15.0,  16.0],
                          [-10.0, -10., 15.0, 15.0, -15.0, -15.0, 0.0, 0.0, -15.0, 0.0])
    # α_cmd(t) = 10*sin(t)

    Am, Bm = W[:, 1:end-1], W[:, end]
    nₓ = size(Am)[1]
    Q_lqr = zeros(nₓ+1, nₓ+1)
    Q_lqr[1,1] = 20.0; Q_lqr[2,2] = 0.2; Q_lqr[3,3] =10.0
    R_lqr = 20.0
    C = zeros(nₓ)
    C[1] = 1.0
    D = 0.0;
    A = [0.0 C';zeros(nₓ) Am]
    B = [D; Bm]
    Bᵣ = zeros(nₓ+1)
    Bᵣ[1] = -1.0
    K_lqr = lqr(A, B, Q_lqr, R_lqr)
    Aᵣ = A - B*K_lqr

    Qᵣ = 100.0*I(nₓ+1) # note
    Γᵤ = 0.05
    Γ_Θ =[0.00001 0.0;0.0 0.0001]
    Pᵣ = lyapc(Aᵣ',Qᵣ)

    ode_fn(ds, s, p, t) =begin
        x = s[1:8] # including ∫ (α-α_cmd) dt
        xₚ = s[2:8]
        xᵣ = s[9:13]
        K̂ₓ = s[14]
        Θ̂ₓ = s[15:16]
        e = [x[1:3];ϕ(x[2:3])] - xᵣ
        Φ = x[2:3]
        
        α, q, mach, γ, h, δ, δdot = xₚ
        uᵦ = -(K_lqr*[x[1:3];ϕ(x[2:3])])[1]
        u = (1.0 - K̂ₓ)*uᵦ - Θ̂ₓ'*Φ
        δ_cmd = sat(u, deg2rad(δ_lim))
        de = α - α_cmd(t)
        dx = dyn_full(α, q, mach, γ, h, δ_cmd, δ, δdot, aero_bias=aero_bias, phy_bias=phy_bias)
        dxᵣ = Aᵣ*xᵣ + Bᵣ*α_cmd(t)
        K̂ₓ_dot = Γᵤ*uᵦ*e'*Pᵣ*B 
        Θ̂ₓ_dot = Γ_Θ*Φ*e'*Pᵣ*B   
        dJ = e'*Q_lqr*e + δ'*R_lqr*δ
        ds .= [de; dx; dxᵣ; K̂ₓ_dot; Θ̂ₓ_dot; dJ]
    end
    x0 = [α₀, q₀, mach₀, γ₀, h₀, δ₀, δdot₀]
    s0 = [0.0; x0; 0.0; α₀; q₀; ϕ([α₀; q₀]); 0.0; 0.0; 0.0; 0.0]
    tspan = (0.0, 12.0)
    prob = ODEProblem(ode_fn, s0, tspan)
    sol = solve(prob, Tsit5())
    u_hist = []
    for sol = sol.u
        x = sol[1:3]
        K̂ₓ = sol[14]
        Θ̂ₓ = sol[15:16]
        Φ = x[2:3]
        uᵦ = -(K_lqr*[x;ϕ(x[2:3])])[1]
        u = rad2deg(sat((1.0 - K̂ₓ)*uᵦ - Θ̂ₓ'*Φ, deg2rad(δ_lim)))
        push!(u_hist, u)
    end
    println("Cost :", sol.u[end][end])

    y_cmd_hist = α_cmd.(sol.t)
    if plot_on == true
        p11 = plot(sol.t,y_cmd_hist, layout=(2,3), subplot=1, label="cmd", ylabel="α (deg)")
        plot!(p11, sol, idxs=(0,10), label="ref")
        plot!(p11, sol, idxs=(0,2), label="act",linestyle=:dash)
        p12 = plot(p11, sol, idxs=(0,14), subplot=2, ylabel="K_x")
        p21 = plot(p12, sol, idxs=(0,15:16),subplot=3, ylabel="Θ_x")
        p22 = plot(p21, sol, idxs=(0, 4), subplot=4, ylabel="Mach", label=nothing)
        p31 = plot(p22, sol.t, u_hist,subplot=5, xlabel="t", ylabel="Fin angle cmd (deg)", label="cmd")
        p31 = plot(p31, sol.t, rad2deg.(sol[7,:]),subplot=5,label="fin")
        p31 = plot(p31, sol.t, 30.0*ones(length(sol)),linestyle=:dash, color=:black, subplot=5, label=nothing)
        p31 = plot(p31, sol.t, -30.0*ones(length(sol)),linestyle=:dash, color=:black, subplot=5, label=nothing)
        p32 = plot(p31, sol, idxs=(0, 3), subplot=6, ylabel="q")
        display(p32)
    end
    
    return sol, u_hist, y_cmd_hist
end

function sim_mrac(ϕ, W, ;n=10, mach_range=[3.0;4.0],alt_range=[6000.0; 11000.0], bias_range=[0.7,1.3], phy_bias=[0.9,1.1], type=Float64)
    machs = uniform(mach_range[1],mach_range[2], n, type=type)
    alts = uniform(alt_range[1], alt_range[2], n, type=type)    
    bias_Ca = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_Cn = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_Cm = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_ma = uniform(phy_bias[1], phy_bias[2], n, type=type)
    bias_in = uniform(phy_bias[1], phy_bias[2], n, type=type)
    for i = 1:n
        (sol, u_hist, y_cmd_hist) = test_train_mrac_full(ϕ, W, h₀=alts[i], mach₀=machs[i], aero_bias=[bias_Ca[i], bias_Cn[i], bias_Cm[i]],
                                    phy_bias=[bias_ma[i], bias_in[i]], plot_on=false)
        df = DataFrame(t=sol.t, cmd=y_cmd_hist, alpha=sol[2,:], u=u_hist, mach=sol[4,:], h=sol[6,:])
        writedlm("data/mrac$i.csv", Iterators.flatten(([names(df)], eachrow(df))),',')
    end
end

function sim_lqr(;n=10, mach_range=[3.0;4.0],alt_range=[6000.0; 11000.0], bias_range=[0.7,1.3], phy_bias=[0.9,1.1], type=Float64)
    machs = uniform(mach_range[1],mach_range[2], n, type=type)
    alts = uniform(alt_range[1], alt_range[2], n, type=type)    
    bias_Ca = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_Cn = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_Cm = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_ma = uniform(phy_bias[1], phy_bias[2], n, type=type)
    bias_in = uniform(phy_bias[1], phy_bias[2], n, type=type)
    for i = 1:n
        (sol, u_hist, y_cmd_hist) = test_lqr_full(h₀=alts[i], mach₀=machs[i], aero_bias=[bias_Ca[i], bias_Cn[i], bias_Cm[i]]
                                 , phy_bias=[bias_ma[i], bias_in[i]] ,plot_on=false)
        df = DataFrame(t=sol.t, cmd=y_cmd_hist, alpha=sol[1,:], u=u_hist, mach=sol[3,:], h=sol[5,:], delta=sol[6,:])
        writedlm("data/lqr$i.csv", Iterators.flatten(([names(df)], eachrow(df))),',')
    end
end

function sim_koopman(ϕ,W;n=10, mach_range=[3.0;4.0],alt_range=[6000.0; 11000.0], bias_range=[0.7,1.3], phy_bias=[0.9,1.1], type=Float64)
    machs = uniform(mach_range[1],mach_range[2], n, type=type)
    alts = uniform(alt_range[1], alt_range[2], n, type=type)
    bias_Ca = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_Cn = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_Cm = uniform(bias_range[1], bias_range[2], n, type=type)
    bias_ma = uniform(phy_bias[1], phy_bias[2], n, type=type)
    bias_in = uniform(phy_bias[1], phy_bias[2], n, type=type)
    for i = 1:n
        (sol, u_hist, y_cmd_hist) = test_koopman_full(ϕ, W, h₀=alts[i], mach₀=machs[i], aero_bias=[bias_Ca[i], bias_Cn[i], bias_Cm[i]]
                                    ,phy_bias=[bias_ma[i], bias_in[i]] ,plot_on=false)
        df = DataFrame(t=sol.t, cmd=y_cmd_hist, alpha=sol[2,:], u=u_hist, mach=sol[4,:], h=sol[6,:], delta=sol[7,:])
        writedlm("data/koopman$i.csv", Iterators.flatten(([names(df)], eachrow(df))),',')
    end
end

function savecsv(sol,u_hist,y_cmd_hist;filename="default")
    df = DataFrame(t=sol.t, cmd=y_cmd_hist, alpha=sol[2,:], q=sol[3,:], delta=sol[7,:], u=u_hist, mach=sol[4,:], h=sol[6,:])
    writedlm("data/"*filename*".csv", Iterators.flatten(([names(df)], eachrow(df))),',')
end
end
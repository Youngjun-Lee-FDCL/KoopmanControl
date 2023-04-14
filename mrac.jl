
module mrac
using ControlSystems, LinearAlgebra
using MatrixEquations, DifferentialEquations
using Plots
# aircraft short period dynamics and control
function get_open_loop_plant()
    Ma = -9.1486f0
    Mq = -4.59f0
    Md = -4.59f0
    Aₚ = [-0.8060f0 1.0f0; Ma Mq]
    Bₚ = [-0.04f0; Md]
    Cₚ = [1.0f0; 0.0f0]

    # augmented 
    A = [[0.0f0 1.0f0 0.0f0]; zeros(Float32, 2) Aₚ]
    B = [0.0f0; Bₚ]
    Bᵣ = [-1.0f0; 0.0f0; 0.0f0]
    C = [0.0f0; Cₚ]
    return A, B, Bᵣ, C, Ma, Mq
end

function uncert_fn(α, q, Ma, Mq)
    ka = 1.5f0*Ma
    kq = 0.5*Mq
    return ka*α+kq*q
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


function test()
    # command
    y_cmd(t) = stepCmd(t, [2.,10.,22.,32.,44.,56.,64.,76.,90.,98.,110.,120.], deg2rad.([0.5,0.0,-0.5,0.0,1.,0.0,-1.0,0.0,0.5,0.0,-0.5,0.0]))
    Q_lqr = zeros(Float32, 3,3)
    Q_lqr[1,1] = 10.f0
    R_lqr = 1.f0

    Qᵣ = 100.f0*I(3)
    Γᵤ = 4000.f0
    Γ_Θ = 4000.f0
    (A, B, Bᵣ, C, Ma, Mq) = get_open_loop_plant()
    K_lqr = lqr(A, B, Q_lqr, R_lqr) # 1 by 3 matrix
    Aᵣ = A-B*K_lqr
    Pᵣ = lyapc(Aᵣ',Qᵣ)
    f(xₚ) = uncert_fn(xₚ[1], xₚ[2], Ma, Mq)
    ode_fn(ds, s, p, t) = begin
        x = s[1:3]
        xₚ = s[2:3]
        xᵣ = s[4:6]
        K̂ₓ = s[7]
        Θ̂ₓ = s[8:9]
        e = x - xᵣ
        Φ = xₚ
        uᵦ = -(K_lqr*x)[1] # baseline controller
        u = (1.0f0 - K̂ₓ)*uᵦ - Θ̂ₓ'*Φ
        x_dot = A*x + B*(u + f(xₚ)) +Bᵣ*y_cmd(t)
        xᵣ_dot = Aᵣ*xᵣ + Bᵣ*y_cmd(t)
        K̂ₓ_dot = Γᵤ*uᵦ*e'*Pᵣ*B 
        Θ̂ₓ_dot = Γ_Θ*Φ*e'*Pᵣ*B   
        J_dot = e'*Qᵣ*e+u'*R_lqr*u
        ds .= [x_dot; xᵣ_dot; K̂ₓ_dot; Θ̂ₓ_dot; J_dot]
    end
    α₀ = 0.0f0; q₀ = 0.0f0;
    x0 =[0.0f0; α₀; q₀; 0.0f0; α₀; q₀; 0.f0; zeros(Float32, 2); 0.0f0]
    tspan = (0.0f0, 140.0f0)
    prob = ODEProblem(ode_fn, x0, tspan)
    sol = solve(prob, Tsit5())
    println("Cost :",sol.u[end][end])
    y_cmd_hist = y_cmd.(sol.t)
    p11 = plot(sol.t,y_cmd_hist, layout=4, subplot=1)
    plot!(p11, sol, idxs=(0,5))
    plot!(p11, sol, idxs=(0,2))
    p12 = plot(p11, sol, idxs=(0,7), subplot=2)
    p21 = plot(p12, sol, idxs=(0,8:9),subplot=3)
    display(p21)
    
    return sol
end

function test2()
    # command
    y_cmd(t) = stepCmd(t, [2.,10.,22.,32.,44.,56.,64.,76.,90.,98.,110.,120.], deg2rad.([0.5,0.0,-0.5,0.0,1.,0.0,-1.0,0.0,0.5,0.0,-0.5,0.0]))
    Q_lqr = zeros(Float32, 3,3)
    Q_lqr[1,1] = 10.f0
    R_lqr = 1.f0

    Qᵣ = 100.f0*I(3)
    Γᵤ = 1000.f0
    Γ_Θ = 1000.f0
    ν = 5.1f0
    Qₙ = Qᵣ + (ν+1.0f0)/ν*I(3)
    Rₙ = ν/(ν+1.0f0)*I(3)
    
    (A, B, Bᵣ, C, Ma, Mq) = get_open_loop_plant()
    K_lqr = lqr(A, B, Q_lqr, R_lqr) # 1 by 3 matrix
    Aᵣ = A-B*K_lqr

    Pₙ = arec(Aᵣ', 1.0*inv(Rₙ), Qₙ)[1]
    Lₙ = Pₙ*inv(Rₙ)
    inv_Pₙ = inv(Pₙ)
    

    f(xₚ) = uncert_fn(xₚ[1], xₚ[2], Ma, Mq)
    ode_fn(ds, s, p, t) = begin
        x = s[1:3]
        xₚ = s[2:3]
        xᵣ = s[4:6]
        K̂ₓ = s[7]
        Θ̂ₓ = s[8:9]
        e = x - xᵣ
        Φ = xₚ
        uᵦ = -(K_lqr*x)[1] # baseline controller
        u = (1.0f0 - K̂ₓ)*uᵦ - Θ̂ₓ'*Φ
        x_dot = A*x + B*(u + f(xₚ)) + Bᵣ*y_cmd(t)
        xᵣ_dot = Aᵣ*xᵣ + Lₙ*(x - xᵣ) + Bᵣ*y_cmd(t)
        K̂ₓ_dot = Γᵤ*uᵦ*e'*inv_Pₙ*B 
        Θ̂ₓ_dot = Γ_Θ*Φ*e'*inv_Pₙ*B   
        J_dot = e'*Qᵣ*e+u'*R_lqr*u
        ds .= [x_dot; xᵣ_dot; K̂ₓ_dot; Θ̂ₓ_dot; J_dot]
    end
    α₀ = 0.0f0; q₀ = 0.0f0;
    x0 =[0.0f0; α₀; q₀; 0.0f0; α₀; q₀; 0.f0; zeros(Float32, 2); 0.0f0]
    tspan = (0.0f0, 140.0f0)
    prob = ODEProblem(ode_fn, x0, tspan)
    sol = solve(prob, Tsit5())
    println("Cost :",sol.u[end][end])
    y_cmd_hist = y_cmd.(sol.t)
    p11 = plot(sol.t,y_cmd_hist, layout=4, subplot=1, label="cmd")
    plot!(p11, sol, idxs=(0,5),label="ref")
    plot!(p11, sol, idxs=(0,2),label="act")
    p12 = plot(p11, sol, idxs=(0,7), subplot=2)
    p21 = plot(p12, sol, idxs=(0,8:9),subplot=3)
    display(p21)
    
    return sol
end
end
module accCont
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

function getAcc(x;bias_cz=1.0f0, bias_ma=1.0f0)
    α, q, mach, γ, h, δ, δ_dot = x
    Vₛ = sonicSpeed(h)
    V = mach*Vₛ
    Q = 1.0/2.0*density(h)*V^2
    return Q*S_ref*d_ref*Cz(α, mach, δ;bias=bias_cz)/mass(bias=bias_ma)
end

function generate_train_data()

end

function accRate(x0, δ_cmd, tspan;aero_bias=[1.0,1.0,1.0], phy_bias=[1.0,1.0])
    ode_fn(ds, s, p, t) = begin
        α, q, mach, γ, h, δ, δ_dot = s
        ds .= dyn_full(α, q, mach, γ, h, δ_cmd, δ, δ_dot, aero_bias=aero_bias, phy_bias=phy_bias)
    end
    prob = ODEProblem(ode_fn, x0, tpsan)
    sol = solve(prob, Tsit5())
    acc_hist = []
    for x in sol.u
        acc = getAcc(x;bias_cz=aero_bias[2], bias_ma=phy_bias[1])
        acc_hist = [acc_hist;acc]
    end
    dt = (tspan[end] - tspan[1])/2.0
    return (acc_hist[end] - acc_hist[1])/(2.0*dt)
end

end
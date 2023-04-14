
using DifferentialEquations, LinearAlgebra, ControlSystems, Plots
using Statistics
using Flux, Zygote, ForwardDiff
using Distributions, Random
struct Observables{S<:AbstractArray}
    s::S
end
(m::Observables)(x) = m.s*x[2]^2


function jvp(func, primal, tangent)
    g(t) = func(primal + t * tangent)
    jvp_result = ForwardDiff.derivative(g, 0.0)
    return jvp_result
end
# dynamics
n = 10
N = 1
batchsize=1
type=Float32
    μ = -1.0f0
    λ = 1.0f0
    A = [μ 0.0f0;0.0f0 λ]
    B = [0.0f0; 1.0f0]
    f(x, u) = A*x + [0.0f0; -λ*x[2]^2] + B*u
    nₓ = 2
    m = 1
    # generate data
    X, Y = generate_train_data(n, nₓ, type=type)
    train_set = Flux.DataLoader((X, Y), batchsize=batchsize)

    Flux.@functor Observables # makes trainable
    
    phi = Observables([0.1f0])
    
    W = rand(Float32, nₓ+N, nₓ+N+m)
    ps = Flux.params(phi, W)
    
    function loss(xandu, y, phi, W)
        x = xandu[1:2]
        u = xandu[3]        
        dx = f(x, u)        
        dϕdt = jvp(phi, x, dx)
        er = [y; dϕdt] .- -W*[x; phi(x); u]
        return sum(abs, er)
    end

    input = [input[:, i] for i = 1:batchsize]
    label = [label[:, i] for i = 1:batchsize]

    vals, grads = Flux.withgradient(phi, W) do phi, W
                #input, label = first(train_set)                
                sum(loss.(input, label, [phi], [W]))
    end


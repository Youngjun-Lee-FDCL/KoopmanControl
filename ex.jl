using Plots
using Distributions
using Flux
using Flux: mse

#
##### GENERATE DATA #########
#
num_samples = 50
x_noise_std = 0.01
y_noise_std = 0.1

function generate_data()
    x = reshape(range(-π/2, stop=π/2, length=num_samples), num_samples, 1)
    y_noise = rand(Normal(0,y_noise_std), num_samples)
    y = sin.(x).^2 .- 0.25 .+ y_noise
    
    return x', y'
end

X, Y = generate_data() # Training data of shape (1,50)

#
##### CUSTOM LAYER #########
#
struct Nonneg{F,S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    σ::F
end

Nonneg(W, b) = Nonneg(W, b, identity)

# Default activation function softplus keeps output non-negative without depressing fits to peaks
function Nonneg(in::Integer, out::Integer, σ=softplus) 
    return Nonneg(randn(out, in), randn(out), σ)
end

Flux.@functor Nonneg  # makes trainable

function (a::Nonneg)(x::AbstractArray)
    a.σ.(a.W * x .+ a.b)
end

# @treelike Nonneg # some say to use @treelike, but it's not used in the Flux definition of Dense

#
##### CALLBACK & PLOTS #########
#
LossLog = []
LossLog_T = []
function evalcb()
    loss_value = loss(X, Y)
    push!(LossLog,loss_value)
    push!(LossLog_T,length(LossLog))
    if mod(length(LossLog),500)==1
        update_loss_plot()
    end
end
    
function update_loss_plot()
    p_loss = plot(LossLog_T, LossLog, ylabel="Loss", xlabel="Index", yscale=:log10, legend=false)
    display(p_loss)
    return p_loss
end

function plot_with_fit(x, y, yfit, label)
    return plot([x x], [y yfit]; color=[:black :red],lw=[0 2], marker=[:circle :none], label=["Data" "Fit"], legend=:top, ylabel="Data & Fit")
end

#
##### MODEL / TRAINING ###############
#
use_nonneg = true # use custom (non-negativity) layer or Dense?

n = 10 # neurons in hidden layers
layer = use_nonneg ? Nonneg(n, 1) : Dense(n, 1)

m = Chain(Dense(1,n,tanh),Dense(10,n,tanh),layer) #Chain(layer)

opt = ADAM()
dataset = [([a], [b]) for (a,b) in zip(X, Y)]
loss(x, y) = mse(m(x), y)

for idx = 1 : 100
    Flux.train!(loss, Flux.params(m), dataset, opt; cb=evalcb)
end

p_loss = update_loss_plot() #final update
p_fit = plot_with_fit(X', Y', m(X)', "Data & Fit")
plot(p_loss, p_fit,layout=(2,1))
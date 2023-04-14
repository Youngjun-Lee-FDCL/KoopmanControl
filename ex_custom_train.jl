using DifferentialEquations, LinearAlgebra, ControlSystems, Plots
using Statistics
using Flux
# Auxiliary functions for generting our data
function generate_real_data(n)
    x1 = rand(1, n) .- 0.5
    x2 = (x1 .* x1)*3 .+ randn(1,n)*0.1
    return vcat(x1, x2)
end

function generate_fake_data(n)
    θ = 2*π*rand(1, n)
    r = rand(1, n)/3
    x1 = @. r*cos(θ)
    x2 = @. r*sin(θ)+0.5
    return vcat(x1, x2)
end

function NeuralNetwork()
    return Chain(
        Dense(2, 25, relu),
        Dense(25, 1, σ)
    )
end


function main2()

# Create our data
train_size = 5000
real = generate_real_data(train_size)
fake = generate_fake_data(train_size)

# Organizing the data in batches
X = hcat(real, fake)
Y = vcat(ones(train_size), zeros(train_size))
data = Flux.Data.DataLoader((X, Y'), batchsize=100, shuffle=true)


# Defining our model, optimization algorithm and loss function
m = NeuralNetwork()
opt = Descent(0.05)

loss(x,y) = sum(Flux.Losses.binarycrossentropy(m(x), y))

ps = Flux.params(m)
epochs = 20
for i in 1:epochs
    Flux.train!(loss, ps, data, opt)
end
@show mean(m(real))
@show mean(m(fake))

# Training method2
m = NeuralNetwork()
function trainModel!(m, data;epochs=20)
    for epoch = 1:epochs
        for d in data
            gs = gradient(Flux.params(m)) do 
                l = loss(d...)
            end
            Flux.update!(opt, Flux.params(m), gs)
        end
    end
    @show mean(m(real))
    @show mean(m(fake))
end
trainModel!(m,data;epochs=20)

scatter(real[1,1:100], real[2,1:100], zcolor=m(real)')
scatter!(fake[1,1:100], fake[2,1:100], zcolor=m(fake)', legend=false)

end
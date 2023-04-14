module test_node
using DifferentialEquations, Flux, DiffEqFlux, OrdinaryDiffEq, Optim
using Plots
using ParameterizedFunctions # required for the `@ode_def` macro
using Sundials
using Random
using Flux.Losses: mse
using BSON: @save, @load

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
end

function main1()
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]

u0_f(p, t0) = [p[2], p[4]]
tspan_f(p) = (0.0, 10*p[4])
prob = ODEProblem(lotka_volterra, u0_f, tspan_f, p)
sol = solve(prob)
plot(sol)

end
function main2()
    p = [1.5,1.0,3.0,1.0]
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(lotka_volterra,u0,tspan,p)
    sol = solve(prob,Tsit5(),saveat=0.1)
    
    p = [2.2, 1.0, 2.0, 0.4]
    params = Flux.params(p)

    function predict_rd()
        solve(prob, Tsit5(), p=p, saveat=0.1)[1,:]
    end
    loss_rd() = sum(abs2, x-1 for x in predict_rd())
    data = Iterators.repeated((), 100)
    opt = ADAM(0.1)
    cb = function ()
        display(plot(solve(remake(prob, p=p), Tsit5(), saveat=0.1), ylim=(0,6)))
    end

    # Display the ODE with the initial parameter values.
    cb()

    Flux.train!(loss_rd, params, data, opt, cb=cb)
end

rober = @ode_def Rober begin
    dy₁ = -k₁*y₁+k₃*y₂*y₃
    dy₂ =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    dy₃ =  k₂*y₂^2
end k₁ k₂ k₃

function main3()    
      prob = ODEProblem(rober,[1.0;0.0;0.0],(0.0,1e11),(0.04,3e7,1e4))
      sol = solve(prob,KenCarp4(),abstol=1e-11, reltol=1e-11) #not working with CVODE_Adams()
      plot(sol,xscale=:log10,tspan=(0.1,1e11))
end

function delay_lotka_volterra(du,u,h,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = (α - β*y)*h(p,t-0.1)[1]
    du[2] = dy = (δ*x - γ)*y
end

function plot_result(t, true_data, pred_data, n=2)
    plts = Array{Plots.Plot{Plots.GRBackend}}(undef, n, 1)
    for i=1:n
        p = scatter(t, true_data[i, :], label="data")
        scatter!(p, t, pred_data[i, :], label="prediction")
        plts[i] = p
    end
    return plts
end
function plot_phase_portrait(true_data, pred_data)
    p = scatter(true_data[1, :], true_data[2, :], label="data")
    scatter!(p, pred_data[1, :], pred_data[2, :], label="prediction")
    return p
end
function try_test(u0,prob,n_ode,t)
    prob=remake(prob,u0=u0)
    true_data = solve(prob, Tsit5(), saveat=t)
    pred_data=n_ode(u0)
    plts=plot_result(t, true_data, pred_data)
    display(plot(plts[1], plts[2], layout=(2,1)))
end
function node()
    u0_1 = Float32[2.;0.]
    u0_2 = Float32[-2.0;0.]
    u0_3 = Float32[0.0;0.]
    datasize = 300
    tspan = (0.0f0, 5.f0)
    function trueODEfunc(du, u, p, t)
        true_A = [-0.1 2.0; -2.0 -0.1]
        du .= ((u.^3)'true_A)'
    end

    t = range(tspan[1], tspan[2], length=datasize)
    prob = ODEProblem(trueODEfunc, u0_1, tspan)
    ode_data1 = Array(solve(prob, Tsit5(), saveat=t))
    prob2 = remake(prob, u0=u0_2)
    ode_data2 = Array(solve(prob2, Tsit5(), saveat=t))
    prob3 = remake(prob, u0=u0_3)
    ode_data3 = Array(solve(prob3, Tsit5(), saveat=t))
    ode_data = (ode_data1, ode_data2, ode_data3)
    # Now let's a neural ODE against this data. To do so, we will define a single layer neural network which just has the
    # same neural ODE  as before (but lower the tolerances to help it converge closer makes for a better animation!)
    dudt = Chain(Dense(2,100,gelu),
                 Dense(100,2))
    n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
    ps = Flux.params(n_ode)
    # now train the neural network. To do so, define a prediction function like before, and then define a loss between our prediction and data:
    function predict_n_ode()
        (n_ode(u0_1), n_ode(u0_2), n_ode(u0_3))
    end
    function loss_n_ode() 
        (pred_data1, pred_data2, pred_data3)= predict_n_ode()
        sum(abs2, ode_data1 .- pred_data1) + sum(abs2, ode_data2 .- pred_data2) #+ sum(abs2, ode_data3 .- pred_data3)
    end
    data = Iterators.repeated((),1000)
    opt = ADAM(0.1)
    cb = function ()
        display(loss_n_ode())
        (cur_pred1, cur_pred2, cur_pred3) = predict_n_ode()
        p1 = plot_phase_portrait(ode_data1, cur_pred1)
        p2 = plot_phase_portrait(ode_data2, cur_pred2)
        display(plot(p1, p2, layout=(1,2)))
    end

    cb()
    Flux.train!(loss_n_ode, ps, data, opt, cb=cb)
    return (prob, n_ode, t)
end

function parametric_ode_system!(du, u, p, t)
    x, y =u 
    a1, b1, c1, d1, a2, b2, c2, d2 = p
    du[1] = dx = a1*x + b1*y + c1*exp(-d1*t)
    du[2] = dy = a2*x + b2*y + c2*exp(-d2*t)
end

function main4()
    u0 = [2.0f0, 0.0f0]
    A = [[-0.1f0, 1.0f0] [-1.0f0, -0.1f0]]

    datasize = 2000
    batchsize = 200
    Ti = 0.0f0
    Tf = 15.0f0
    tspan = (Ti, Tf)
    t = range(tspan[1], tspan[2], length=datasize)
    function dudt(du, u, p, t) 
        du .=A*u
    end
    prob_true = ODEProblem(dudt, u0, tspan)
    xtrain = Array(solve(prob_true, Tsit5(), saveat=t, reltol=1e-7))
    train_set = Flux.DataLoader(xtrain, batchsize=batchsize)
    
    # Neural ode model
    hid_layer_size = 20
    dudt = Chain(Dense(2,hid_layer_size,tanh),
                 Dense(hid_layer_size,2))
    n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
    return n_ode
    # Loss functions
    loss(y_hat, y) = sum(abs, y_hat .- y)

    
    # Train loop
    opt_state = Flux.setup(ADAM(0.02), n_ode)
    for data in train_set
        
        grads = Flux.gradient(n_ode) do m
            pre_traj = m(u0)
            loss(pre_traj, data)
        end

        # Update the parameters so as to reduce the loss,
        # according the chosen optmisation rule:
        Flux.update!(opt_state, model, grads[1])
    end

    
    

    ps = Flux.params(n_ode)

    # Dataset & batches
    batchsize = 200
    MAX_BATCHES = round(Int, datasize/batchsize)
    data = ((rand(1:size(xtrain)[2] -batchsize), batchsize) for i in 1:MAX_BATCHES)

    # Optimizer
    opt = ADAM(0.002, (0.9, 0.999))

    function predict_n_ode(p)
        Array(n_ode(u0, p))
    end

    function loss_n_ode(p, start, k)
        pred = predict_n_ode(p)
        loss = sum(abs, xtrain[:, start:start+k] .- pred[1, start:start+k])
        loss, pred
    end

    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, xtrain .- pred)
        return loss, pred
    end


    list_plots = []
    iter = 0
    cb = function (p,loss,pred;doplot=false) #callback function to observe training
        global list_plots, iter
        if iter == 0
            list_plots = []
        end
        iter += 1

        display(loss)
        # plot current prediction against data
        
        pl = scatter(t,xtrain[1,:],label="data")
        scatter!(pl,t,pred[1,:],label="prediction")
        push!(list_plots, pl)
        if doplot
          display(plot(pl))
        end
        return false
    end

    result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, ps,
                                          opt, cb = cb, maxiters = 300)
    


    """
    plot(trange, dataset_outs[1],
    linewidth=2, ls=:dash,
    title="Neural ODEs to fit params",
    xaxis="t",
    label="dataset x(t)",
    legend=true)
    """

end

function main5()
    u0 = Float32[2.0; 0.0]
    datasize = 30
    tspan = (0.0f0, 1.5f0)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    function trueODEfunc(du, u, p, t)
        true_A = [-0.1 2.0; -2.0 -0.1]
        du .= ((u.^3)'true_A)'
    end

    prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
    ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

    # define n ode
    dudt2 = FastChain((x, p) -> x.^3,
                    FastDense(2, 50, tanh),
                    FastDense(50, 2))
    prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

    function predict_neuralode(p)
        Array(prob_neuralode(u0, p))
    end
      
    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, ode_data .- pred)
        return loss, pred
    end

    function loss_n_ode(p, start, k)
        pred = predict_neuralode(p)
        loss = sum(abs2,ode_data[:,start:start+k] .- pred[:,start:start+k])
        loss,pred
    end

    MAX_BATCHES = 1000
    k = 15 #batch size
    data = ((rand(1:size(ode_data)[2] -k), k) for i in 1:MAX_BATCHES)

    # Callback function to observe training
    list_plots = []
    iter = 0
    callback = function (p, l, pred; doplot = false)
        if iter == 0
            list_plots = []
        end
        iter += 1

        display(l)

        # plot current prediction against data
        plt = scatter(tsteps, ode_data[1,:], label = "data")
        scatter!(plt, tsteps, pred[1,:], label = "prediction")
        push!(list_plots, plt)
        if doplot
            display(plot(plt))
        end

        return false
    end
    
    result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 300)
    
    result_neuralode2 = DiffEqFlux.sciml_train(loss_n_ode,
    result_neuralode.minimizer, LBFGS(), data, cb = callback)
    return result_neuralode2
    
    end

    function train()
        # Generate data
        u0 = [2.0f0, 0.0f0]
        A = [[-0.1f0, 1.0f0] [-1.0f0, -0.1f0]]

        # Initialize the optimizer for this model:
        opt = ADAM(0.02)
        opt_state = Flux.setup(opt, model)
        
        # data & batches
        function dudt(du, u, p, t)
            du .= A*u
        end
        datasize = 30
        tspan = (0.0f0, 1.5f0)
        tsteps = range(tspan[1], tspan[2], length = datasize)
        t = 
        prob = ODEProblem(dudt, u0, tspan)
        train_data = Array(solve(prob, Tsit5(), saveat=t))
        
    end
end
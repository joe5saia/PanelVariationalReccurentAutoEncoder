###################################################################################################
#=
Helper functions for vrnn.jl
    
Adapted from https://arxiv.org/abs/1506.02216
=#
###################################################################################################
using DataFrames, CSV, Flux, Revise, Distributions, LinearAlgebra, Random, LoopVectorization, Dates, FixedEffectModels, Zygote
using Parameters: @with_kw
using Flux: gate
using BSON: @save, @load

@with_kw mutable struct Args
    η = 1e-3                # learning rate
    epochs = 5000           # number of epochs
    seed = 1234             # random seed
    cuda = false            # use GPU
    x_dim = 10              # Dimensionality of data
    y_dim = 10              # Dimensionality of the subset of data we want to predict
    h_dim = 4               # Dimensionality of hidden state
    z_dim = 3               # Dimensionality of latent variable
end

function makedata(;backward=false)
    df = CSV.read("data_40.csv")
    categorical!(df, [:datadateq, :sector])
    df2 = partial_out(df, @formula(sizeq + divpayq + alevq + nlevq + carq + salerq + revtrq + xoprrq + aqcyrq + nopirq + intanrq ~ fe(datadateq)&fe(sector)))[1]
    dataelements = (;
            bsdata = Matrix{Float32}(df[!, [:sizeq, :divpayq, :alevq, :nlevq, :carq, :salerq, :revtrq, :xoprrq, :aqcyrq, :nopirq, :intanrq]])' ,
            sectors = Flux.onehotbatch(df.sector, levels(df.sector)),
            dates = Flux.onehotbatch(df.datadateq, levels(df.datadateq))
    )
    x = permutedims(Matrix{Float32}(df2))
    hz = size(df2,1) ÷ 40
    backward ? z = [x[:, (1:hz) .+ hz*(n-1)] for n in reverse(1:40)] : z = [x[:, (1:hz) .+ hz*(n-1)] for n in 1:40]
    return (z,z)
end


#Phi_x(args::Args) = Chain(Dense(args.x_dim, 2*args.phi_x_dim, swish), Dense(2*args.phi_x_dim, args.phi_x_dim))
#Phi_z(args::Args) = Chain(Dense(args.z_dim, 2*args.phi_z_dim, swish), Dense(2*args.phi_z_dim, args.phi_z_dim))
Encoder(args::Args) =  Chain(Dense(args.x_dim + args.h_dim, 4*args.z_dim, sigmoid), Dense(4*args.z_dim, 2*args.z_dim))
Decoder(args::Args)  = Chain(Dense(args.z_dim + args.h_dim, 4*args.y_dim, sigmoid), Dense(4*args.y_dim, 2*args.y_dim))
Prior(args::Args)  = Chain(Dense(args.h_dim, 4*args.z_dim, sigmoid), Dense(4*args.z_dim, 2*args.z_dim))
Hidden(args::Args)  = Chain(Dense(args.x_dim + args.z_dim, 2*args.x_dim, sigmoid), LSTM(2*args.x_dim, args.h_dim)) 
h(m) = m[end].state[2]

function μlogσ(output)
    n = size(output,1) ÷ 2
    return @views (gate(output, n, 1), gate(output, n, 2))
end

function forward!(xi, encoder, decoder, prior, hidden)
    # update inference parameters for (x', h) -> z'
    e = encoder(vcat(xi, h(hidden)))
    # update prior parameters for h -> z'
    p = prior(h(hidden))
    # Reprarameterize z' by sampling z' | x' h
    n = size(e,1) ÷ 2
    μ =  gate(e, n, 1) 
    logσ = gate(e, n, 2)
    z = μ .+ exp.(logσ) .* randn(eltype(logσ), size(logσ)...)
    # Decoder parameters for (z', h) -> x'
    d = decoder(vcat(z, h(hidden)))
    # recurence (x', z', h) -> h'
    hidden(vcat(xi, z))
    return e, p, d
end

function kld_normal(e, p)
    # (-) Kullback–Leibler divergence for 2 mulitivariate normal distributions
    # Σ_0 and Σ_1 are diagonal, so everything can be done element-wise
    n = size(e,1) ÷ 2
    μ0 =  gate(e, n, 1) 
    logσ0 = gate(e, n, 2)
    μ1 =  gate(p, n, 1) 
    logσ1 = gate(p, n, 2)
    T = eltype(μ0)
    T1 = convert(T, 1)
    T2 = convert(T, 2)
    sum(@. (exp(T2 * logσ0) + (μ0-μ1)^2) / (exp(T2 * logσ1)) + T2 * logσ1 - T2 * logσ0 - T1)/(2 * size(μ0,2))
end

function mycrossentropy(d, y)
    μ, logσ = μlogσ(d)
    sum(@. logpdf(Normal(μ, exp(logσ)), y))/size(μ,2)
end

function step_loss(xi, yi, encoder, decoder, prior, hidden)
    e, p, d = forward!(xi, encoder, decoder, prior, hidden)
    return kld_normal(e, p) - mycrossentropy(d, yi)
end

function total_loss!(x, y, encoder, decoder, prior, hidden)
    Flux.reset!(hidden)
    totaloss = oftype(x[1][1], 0)
    @inbounds for i in 1:length(x)
        totaloss += step_loss(x[i], x[i], encoder, decoder, prior, hidden)
    end
    return totaloss/length(x)
end


function trainmod(args, m = nothing, opt = nothing; backward = false)
    # Get Data
    train_data = makedata(;backward=backward)
    x = train_data[1]
    y = train_data[2]
    # Constructing Model
    if m === nothing
        encoder = Encoder(args::Args)
        decoder = Decoder(args::Args)
        prior = Prior(args::Args)  
        hidden = Hidden(args::Args) 
        m = (;encoder, decoder, prior, hidden)
        # Run it once to make the hidden state variables have the correct size
        xprobe = train_data[1][1]
        hidden(vcat(xprobe, rand(args.z_dim, size(xprobe,2)) ) )
        hidden[end].init = (zeros(eltype(hidden[end].state[1]), size(hidden[end].state[1])), zeros(eltype(hidden[end].state[1]), size(hidden[end].state[1])) )
    else
        encoder, decoder, prior, hidden = m
    end
    loss(x, y) = total_loss!(x, y, encoder, decoder, prior, hidden)
    ## Training
    (opt === nothing) && (opt = ADAM(args.η))
    ps = Flux.params(encoder, decoder, prior, hidden)
    Random.seed!(args.seed)
    for i in 1:args.epochs
        Flux.reset!(hidden)
        train_loss, back = Zygote.pullback(() -> loss(x,x), ps)
        gs = back(one(train_loss))
        Flux.update!(opt, ps, gs)
        @show i, loss(x,x)
        m = (;encoder, decoder, prior, hidden)
        @save "vaemods/model_back=$(backward).bson" m opt args
    end
    return m
end


###################################################################################################
#=
Prediction Functions
=#
###################################################################################################

function predict!(x, encoder, decoder, prior, hidden)
    # roll the model forward
    # If no part of x' is not missing, do the full step to update the hidden state
    # Else do a 2 step EM update and then roll model forward to update the hidden state
    # 1.A) update the prior on z' according to h
    # 1.B) Draw a new z' from prior and calculate ϕ_z
    # 1.C) calculate expected value of x' given z' and h
    # 1.D) replace missing x' with expected value
    # 2.A) update inference parameters for (x', h) -> z'
    # 2.B) Reprarameterize z' by sampling z' | x' h
    # 2.C) calculate expected value of x' given z' and h and replace missing values
    # 3) Roll hidden state forward
    xi0 = copy(x[:,1]) # Preallocate storage vector
    Flux.reset!(hidden)
    for (i,xi) in enumerate(eachcol(x))
        misses = ismissing.(xi)
        if all(.!misses)
            # No missing, roll hidden state forward
            forward!(xi, encoder, decoder, prior, hidden)
        else
            # update prior parameters for h -> z'
            p = prior(h(hidden))
            # Reprarameterize z' by E[z' | h]
            z = μlogσ(p)[1] 
            # Decoder parameters for (z', h) -> x'
            d = decoder(vcat(z, h(hidden)))
            # Find E[x' |z' h]
            μx = μlogσ(d)[1]
            # Fill in missings
            xi[misses] .= μx[misses]
            counter = 0
            xi0 .= 10
            while (norm(xi - xi0, Inf) < 0.01) && (counter > 1000)
                # Redo cycle, this time using full x' to incorporate nonmissing values of x
                # store results from previous iteration
                xi0[:] .= xi
                # update inference parameters for (x', h) -> z'
                e = encoder(vcat(xi, h(hidden)))
                # Reprarameterize z' by E[z' |x', h]
                z = μlogσ(e)[1] 
                # Decoder parameters for (z', h) -> x'
                d = decoder(vcat(z, h(hidden)))
                # Find E[x' |z' h]
                μx = μlogσ(d)[1]
                # Fill in missings
                xi[misses] = μx[misses]
            end
            # recurence (x', z', h) -> h'
            hidden(vcat(xi, z))
            # Actually save results to X
            x[misses, i] .= xi[misses] 
        end
    end
end

function load_mod(file)
    # Load the model and reset the hidden state variables to vectors of zeros
    #file = "vaemods/model.bson"
    @load file m opt args
    encoder, decoder, prior, hidden = m
    hidden = Hidden(args)
    Flux.loadparams!(hidden, Flux.params(m.hidden))
    return ((;encoder, decoder, prior, hidden), args, opt)
end

function make_test_data(x0, p=0.05)
    Random.seed!(1234)
    missels = rand(Bernoulli(p), size(x0))
    x1 = copy(x0)
    x1[missels] .= missing
    return x1
end

function avg_predictions(xf, xb)
    x = copy(xf)
    T = size(xf,2)
    for i in 1:T
        x[:,i] .= (xf[:,i] .* i/T) .+ (xb[:,i] .* (T-i)/T)
    end
    return x
end

function test_train(datas, modf, modb, p=0.05)
    Nf = size(datas[1],2)
    X1 = Array{Array{Float32,2}}(undef, Nf)
    X0 = Array{Array{Float32,2}}(undef, Nf)
    Missels = Array{Array{Float32,2}}(undef, Nf)
    Distance = Array{Array{Union{Missing,Float32},1}}(undef, Nf)
    for i in 1:Nf
        x0 = Matrix{Union{Missing, Float32}}(hcat([z[:, Nf] for z in datas]...))
        x1f = make_test_data(x0, p)
        x1b = copy(x1f)[:, end:-1:1]
        missels = ismissing.(x1f)
        predict!(x1f, modf...)
        predict!(x1b, modb...)
        x1 = avg_predictions(x1f, x1b[:, end:-1:1])
        X0[i] = x0
        X1[i] = x1
        Missels[i] = missels
        Distance[i] = abserror(x1, x0, missels)
    end
    return X1, X0, Distance, Missels
end

function abserror(x1, x0, missels)
    T = eltype(x1)
    N = size(x1,1)
    result = Vector{T}(missing, N)
    for i in 1:N
        j = missels[i,:]
        any(j) && (result[i] = mean( abs.(x1[i, j] .- x0[i, j]) ))
    end
    return result
end

function naive_interoplate(datas, p=0.25)
    Nf = size(datas[1],2)
    X1 = Array{Array{Float32,2}}(undef, Nf)
    X0 = Array{Array{Float32,2}}(undef, Nf)
    Missels = Array{Array{Float32,2}}(undef, Nf)
    Distance = Array{Array{Union{Missing,Float32},1}}(undef, Nf)
    for i in 1:Nf
        x0 = Matrix{Union{Missing, Float32}}(hcat([z[:, Nf] for z in datas]...))
        x1 = make_test_data(x0, p)
        missels = ismissing.(x1)
        for j in 1:size(x1,1)
            linear_impute!(view(x1, j , :))
        end
        X0[i] = x0
        X1[i] = x1
        Missels[i] = missels
        Distance[i] = abserror(x1, x0, missels)
    end
    return X1, X0, Distance, Missels
end



function linear_impute!(x::AbstractVector)
    # Check to see if there is even any data to use, if not then set everything 0
    all(ismissing.(x)) && (x[:] .= 0)
    # Handle first/last elements of vector by carrying backward/forward next valid observation
    ismissing(x[1]) && ( x[1] = x[findfirst(.!ismissing.(x))] )
    ismissing(x[end]) && ( x[end] = x[findlast(.!ismissing.(x))] )
    # Do linear interpolation
    for i in 2:length(x)-1
        if ismissing(x[i])
            j0 = findlast(.!ismissing.(x[1:i]))
            j1 = findfirst(.!ismissing.(x[i:end])) + i - 1
            x[i] = ((j1-j0-1) * x[j0] + x[j1]) / (j1-j0)
        end
    end
    return x
end


function run_predictions(datas, modf, modb)
    X1, X0, Distance, Missels = test_train(datas, modf, modb, 0.01)
    σ0 = map(x -> std(x, dims=2), X0)
    Reldistance = hcat([Distance[i]./ σ0[i] for i in 1:length(Distance)]...)
    reldistance = mapslices(mean ∘ skipmissing, Reldistance, dims=2)
    df1 = DataFrame([0.0001 reldistance'])
    for p in 0.05:0.05:0.90
        X1, X0, Distance, Missels = test_train(datas, modf, modb, p)
        σ0 = map(x -> std(x, dims=2), X0)
        Reldistance = hcat([Distance[i]./ σ0[i] for i in 1:length(Distance)]...)
        reldistance = mapslices(mean ∘ skipmissing, Reldistance, dims=2)
        append!(df1, DataFrame([p reldistance']))
    end
    
    
    X1, X0, Distance, Missels = naive_interoplate(datas, 0.01)
    df2 = DataFrame([0.0001 reldistance'])
    σ0 = map(x -> std(x, dims=2), X0)
    Reldistance = hcat([Distance[i]./ σ0[i] for i in 1:length(Distance)]...)
    reldistance = mapslices(mean ∘ skipmissing, Reldistance, dims=2)
    df2 = DataFrame([0.0001 reldistance'])
    for p in 0.05:0.05:0.90
        X1, X0, Distance, Missels = naive_interoplate(datas, p)
        σ0 = map(x -> std(x, dims=2), X0)
        Reldistance = hcat([Distance[i]./ σ0[i] for i in 1:length(Distance)]...)
        reldistance = mapslices(mean ∘ skipmissing, Reldistance, dims=2)
        append!(df2, DataFrame([p reldistance']))
    end
    return df1, df2
end
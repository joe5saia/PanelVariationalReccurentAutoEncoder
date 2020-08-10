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
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 10            # number of epochs
    seed = 1234             # random seed
    cuda = false            # use GPU
    x_dim = 10              # Dimensionality of data
    y_dim = 10              # Dimensionality of the subset of data we want to predict
    h_dim = 3               # Dimensionality of hidden state
    z_dim = 2               # Dimensionality of latent variable
    phi_x_dim = 10          # Output dimension of ϕ_x
    phi_z_dim = 2           # Output dimension of ϕ_z
end

function makedata()
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
    z = [x[:, (1:hz) .+ hz*(n-1)] for n in 1:40]
    return (z,z)
end


Phi_x(args::Args) = Chain(Dense(args.x_dim, 2*args.phi_x_dim, swish), Dense(2*args.phi_x_dim, args.phi_x_dim, swish))
Phi_z(args::Args) = Chain(Dense(args.z_dim, 2*args.phi_z_dim, swish), Dense(2*args.phi_z_dim, args.phi_z_dim, swish))
Encoder(args::Args) =  Chain(Dense(args.phi_x_dim + args.h_dim, 2*args.z_dim, swish), Dense(2*args.z_dim, 2*args.z_dim))
Decoder(args::Args)  = Chain(Dense(args.phi_z_dim + args.h_dim, 2*args.y_dim, swish), Dense(2*args.y_dim, 2*args.y_dim))
Prior(args::Args)  = Chain(Dense(args.h_dim, 2*args.z_dim, swish), Dense(2*args.z_dim,2*args.z_dim))
Hidden(args::Args)  = Chain(Dense(args.phi_x_dim + args.phi_z_dim, 2*args.h_dim), LSTM(2*args.h_dim, args.h_dim)) 


function μlogσ(output)
    n = size(output,1) ÷ 2
    return (gate(output, n, 1), gate(output, n, 2))
end

h(m) = m[2].state[2]

function forward(xi, phi_x, phi_z, encoder, decoder, prior, hidden)
    _phi_x = phi_x(xi)
    # update inference parameters for (x', h) -> z'
    e = encoder(vcat(phi_x(xi), h(hidden)))
    # update prior parameters for h -> z'
    p = prior(h(hidden))
    # Reprarameterize z' by sampling z' | x' h
    z = μlogσ(e)[1] .+ exp.(μlogσ(e)[2]) .* randn(Float32, size(μlogσ(e)[2])... )
    _phi_z = phi_z(z)
    # Decoder parameters for (z', h) -> x'
    d = decoder(vcat(_phi_z, h(hidden)))
    # recurence (x', z', h) -> h'
    hidden(vcat(_phi_x, _phi_z))
    return e, p, d
end

function kld_normal(e, p)
    # Kullback–Leibler divergence for 2 mulitivariate normal distributions
    # Σ_0 and Σ_1 are diagonal, so everything can be done element-wise
    μ0, logσ0 = μlogσ(e)
    μ1, logσ1 = μlogσ(p)
    l =  oftype(μ0[1], 0)
    @inbounds for i in eachindex(μ0)
        l += (exp(2 * logσ0[i]) + (μ0[i]-μ1[i])^2) / (exp(2 * logσ1[i])) + 2 * logσ1[i] - 2 * logσ0[i] - 1
    end
    return l/2
end

function mycrossentropy(d, y)
    μ, logσ = μlogσ(d)
    normals = @. Normal(μ, exp(logσ))
    #l = sum(@. logpdf(Normal(μ, exp(logσ)), y))
    l = oftype(μ[1], 0)
    @inbounds for j in 1:size(y,2)
        @inbounds for i in 1:size(y,1)
            l += logpdf.(normals[i], y[i, j])
        end
    end
    return l
end

function step_loss(xi, yi, phi_x, phi_z, encoder, decoder, prior, hidden)
    e, p, d = forward(xi, phi_x, phi_z, encoder, decoder, prior, hidden)
    return kld_normal(e, p) - mycrossentropy(d, yi)
end

function total_loss!(x, y, phi_x, phi_z, encoder, decoder, prior, hidden)
    Flux.reset!(hidden)
    totaloss = oftype(x[1][1], 0)
    @inbounds for i in 1:length(x)
        totaloss += step_loss(x[i], x[i], phi_x, phi_z, encoder, decoder, prior, hidden)
    end
    return totaloss
end

function predict!(x, phi_x, phi_z, encoder, decoder, prior, hidden)
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
    Flux.reset!(hidden)
    for (i,xi) in enumerate(eachcol(x))
        misses = .!ismissing.(xi)
        if all(misses)
            # If all data exists then do a forward pass to update the hidden state
            _phi_x = phi_x(xi)
            # update inference parameters for (x', h) -> z'
            e = encoder(vcat(phi_x(xi), h(hidden)))
            # update prior parameters for h -> z'
            p = prior(h(hidden))
            # Reprarameterize z' by sampling z' | x' h
            z = μlogσ(e)[1] .+ μlogσ(e)[2] .* randn(Float32, size(μlogσ(e)[2])... )
            _phi_z = phi_z(z)
            # Decoder parameters for (z', h) -> x'
            d = decoder(vcat(_phi_z, h(hidden)))
            # recurence (x', z', h) -> h'
            hidden(vcat(_phi_x, _phi_z))
        else
            p = prior(h(hidden))
            z = μlogσ(p)[1] .+ μlogσ(p)[2] .* randn(Float32, size(μlogσ(p)[2])... )
            _phi_z = phi_z(z)
            d = decoder(vcat(_phi_z, h(hidden)))
            μx = μlogσ(d)[1]
            @show misses = .!misses[1:size(μx,1)]
            x[ismissing.(xi), i] = μx[misses]
        end
    end
end

function trainmod(m = nothing)
    # Get Data
    train_data = makedata()
    # Constructing Model
    args = Args(x_dim = size(train_data[1][1],1), y_dim = size(train_data[1][1],1) )
    if m === nothing
        phi_x = Phi_x(args::Args)
        phi_z = Phi_z(args::Args)
        encoder = Encoder(args::Args)
        decoder = Decoder(args::Args)
        prior = Prior(args::Args)  
        hidden = Hidden(args::Args) 
        m = (;phi_x, phi_z, encoder, decoder, prior, hidden)
        # Run it once to make the hidden state variables have the correct size
        xprobe = train_data[1][1]
        hidden(vcat(phi_x(xprobe), phi_z(rand(args.z_dim, size(xprobe,2)) )))
        hidden[2].init = (zeros(eltype(hidden[2].state[1]), size(hidden[2].state[1])), zeros(eltype(hidden[2].state[1]), size(hidden[2].state[1])) )
    else
        phi_x, phi_x, encoder, decoder, prior, hidden = m
    end
    loss(x, y) = total_loss!(x, y, phi_x, phi_z, encoder, decoder, prior, hidden)
    ## Training
    opt = ADAM()
    ps = Flux.params(phi_x, phi_z, encoder, decoder, prior, hidden)
    Random.seed!(args.seed)
    for i in 1:args.epochs
        Flux.reset!(hidden)
        train_loss, back = Zygote.pullback(() -> loss(x,x), ps)
        gs = back(one(train_loss))
        Flux.update!(opt, ps, gs)
        @show i, loss(x,x)
        m = (;phi_x, phi_z, encoder, decoder, prior, hidden)
        @save "vaemods/model.bson" m opt
    end
    return (m)
end




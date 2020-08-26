using DataFrames, CSV, Flux, Revise, Distributions, LinearAlgebra, Random, LoopVectorization, Dates, FixedEffectModels, Zygote
using Parameters: @with_kw
using Flux: gate
using BSON: @save, @load

@with_kw mutable struct Args
    η = 1e-3                # learning rate
    epochs = 2000           # number of epochs
    seed = 1234             # random seed
    cuda = false            # use GPU
    x_dim = 10              # Dimensionality of data
    h_dim = 4               # Dimensionality of hidden state
end



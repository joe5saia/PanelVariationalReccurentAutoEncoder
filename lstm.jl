###################################################################################################
#=
Simple long short-term memory autoencoder
=#
###################################################################################################
using DataFrames, CSV, Flux, Revise, Distributions, LinearAlgebra, Random, LoopVectorization, FixedEffectModels
using BSON: @save

includet("vrnn_helper.jl")
includet("lstm_helper.jl")

function make_model(x_dim, h_dim)
    Chain(Dense(args.x_dim, 4*args.h_dim), LSTM(4*args.h_dim, 4 * args.x_dim)) 
end
###################################################################################################
#=
Trains a variational reccurent autoeconding neural net to fill in missing values in compustat data
Adapted from https://arxiv.org/abs/1506.02216
=#
###################################################################################################
using DataFrames, CSV, Flux, Revise, Distributions, LinearAlgebra, Random, LoopVectorization, FixedEffectModels
using BSON: @save, @load

include("vrnn_helper.jl")

dataset = makedata()
args = Args(x_dim = size(first(dataset)[1],1), y_dim = size(first(dataset)[1],1) )

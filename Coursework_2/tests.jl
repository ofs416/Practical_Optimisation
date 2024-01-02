using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")

m2 = Matrix{Float64}(undef, (10, 3, 3))
m2


using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")

m1 = rand(1:5, 5, 3)
m2 = rand(1:2, 5, 3)
println(m1)
println(m2)
m1[1, :] = m2[1, :]
println(m1)
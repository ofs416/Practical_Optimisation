using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools
using LinearAlgebra

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")
include("HybridGAPSO.jl")


scores = []
for iter in 1:200
    push!(scores, PSO(8, 200, 0.8, 2.5, 1.25, false))
end
println(mean(scores))
println(std(scores))


#PGAPSO(8, 100, 0.8, 1.5, 1.0, 0.7, 0.001, var_locus_crossover, false)
#@benchmark PSO(2, 120, 0.8, 2.5, 1.25, false)


#rand(LinRange(-0.5, 1., 1000), 5, 2)[1, :]' .+ rand(LinRange(-0.0, 0.0, 1000), 5, 2)
#rand(eachcol([1 2 3 4 5; 2 3 4 5 6]))' .+ rand(LinRange(-0.0, 0.0, 1000), 5, 2)


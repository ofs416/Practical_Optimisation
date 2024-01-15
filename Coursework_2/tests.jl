using Printf
using Statistics,StatsBase, Distributions
using Plots
using BenchmarkTools
using LinearAlgebra
using Random

include("KBFunc.jl")
#include("GeneticAlgo.jl")
include("Misc.jl")
include("HybridGAPSO.jl")
#include("ParticleSwarm.jl")

println("test")




#GA(2, 100, 0.7, 0.1, loci_crossover, true)

#@benchmark GA(2, 100, 0.7, 0.05, loci_crossover, false)
#PGAPSO(2, 100, 0.8, 2.5, 1.25, 0.7, 0.1, true)

scores = []
for iter in 1:10
    push!(scores, PGAPSO(8, 100, 0.8, 2.5, 1.25, 0.7, 0.1, false))
    #push!(scores, PSO(8, 100, 0.8, 2.5, 1.25, false))
    #push!(scores, GA(8, 100, 0.8, 0.10, false))
end
println(mean(scores))
println(std(scores))





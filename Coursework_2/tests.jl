using Printf
using Statistics,StatsBase, Distributions
using Plots
using BenchmarkTools
using LinearAlgebra
using Random

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")
include("HybridGAPSO.jl")
include("ParticleSwarm.jl")

println("test")

@benchmark GA(2, 200, 0.7, 0.01, var_locus_crossover, false)
PGAPSO(8, 120, 0.8, 2.5, 1.25, 0.7, 0.01, var_locus_crossover, false)

scores = []
for iter in 1:50
    push!(scores, PGAPSO(8, 120, 0.8, 2.5, 1.25, 0.7, 0.001, 
                         var_locus_crossover, false))
end
println(mean(scores))
println(std(scores))


#@benchmark GA(2, 200, 0.7, 0.01, var_locus_crossover, false)

#PGAPSO(8, 200, 0.8, 2.5, 1.25, 0.7, 0.5, var_locus_crossover, false)
#@benchmark PSO(2, 120, 0.8, 2.5, 1.25, false)


#rand(LinRange(-0.5, 1., 1000), 5, 2)[1, :]' .+ rand(LinRange(-0.0, 0.0, 1000), 5, 2)
#rand(eachcol([1 2 3 4 5; 2 3 4 5 6]))' .+ rand(LinRange(-0.0, 0.0, 1000), 5, 2)



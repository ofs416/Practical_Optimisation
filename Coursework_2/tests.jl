using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")
include("HybridGAPSO.jl")


#scores = Float64[]
#for iter in 1:25 
#    push!(scores,PS(8, 200, 0.9, 2.5, 1.25, score_top5, false))
#end
#println(mean(scores))
#println(std(scores))


#GAPSO(2, 200, 0.9, 2.5, 1.25, 0.7, 0.001, var_locus_crossover, score_top5, true)
PS(2, 200, 0.9, 2.5, 1.25, score_top5, true)


#rand(LinRange(-0.5, 1., 1000), 5, 2)[1, :]' .+ rand(LinRange(-0.0, 0.0, 1000), 5, 2)
#rand(eachcol([1 2 3 4 5; 2 3 4 5 6]))' .+ rand(LinRange(-0.0, 0.0, 1000), 5, 2)

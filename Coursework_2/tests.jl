using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")

fib(n::Int) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)




avg = Matrix{Float64}(undef, (9, 4))

for (i, pop) in enumerate(20:10:100)
    for (j, mut) in enumerate([0.0001, 0.001, 0.01, 0.1])
        println(i, j)
    end
end

avg .+= 1
avg ./= 0.5
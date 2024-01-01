using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")

fib(n::Int) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)

range = LinRange(0, 10, 11)


popu = pop_initial(range, 10, 3)
f = KBF(popu)

histogram(randn(10000), bins=100)

mean(randn(10000))
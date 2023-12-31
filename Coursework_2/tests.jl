using Printf
using Statistics, Distributions, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")

fib(n::Int) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)


sample(1:2, Weights([0.8, 0.1]), (2,50))
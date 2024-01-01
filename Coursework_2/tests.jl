using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")

fib(n::Int) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)

using Printf
using Statistics, Distributions
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")

fib(n::Int) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)



list = [1,2,3,4,5]
val = [4,5,6,7,8]

val[list[1:2]]
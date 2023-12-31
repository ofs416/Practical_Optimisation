using Printf
using Statistics, StatsBase
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")

fib(n::Int) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)


function parent_select(e_i)
    if all(x-> x<1, e_i)
        parent_f = sample(1:4, Weights(e_i), 1)[1]
    else 
        parent_f = sample(1:4, 1)[1]
        while e_i[parent_f] < 1
            parent_f = sample(1:4, 1)[1]
        end
        e_i[parent_f] -= 1
    end
    return parent_f
end

parents = Vector{Vector{Int}}()
p_si = [0.1234567, 0.4, 0.3, 1 - 0.4 - 0.3 - 0.1234567]
e_i = p_si .* 10
println(e_i)
while (length(parents) < 5)
    parent11 = parent_select(e_i)
    parent22 = parent_select(e_i)
    pair = [parent11, parent22]
    push!(parents, pair)
end
println(e_i)
println(parents)
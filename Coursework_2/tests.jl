using Printf
using Statistics, Distributions
using Plots

include("KBFunc.jl")
include("GeneticAlgo.jl")

child1 = ["", ""]
child2 = ["", ""]
parent1, parent2 = [1., 2.], [3., 4.]
child1_final, child2_final = Float64[], Float64[]
for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
    for (bit1, bit2) in zip(bitstring(parent1_x)[2:end], bitstring(parent2_x)[2:end])
        bit1, bit2 = crossover(bit1, bit2)
        bit1, bit2 = mutate(bit1, bit2, 0.1)
        child1[index] = child1[index] * bit1
        child2[index] = child2[index] * bit2
    end
    push!(child1_final,reinterpret(Float64, parse(Int64, child1[index], base=2)))
    push!(child2_final,reinterpret(Float64, parse(Int64, child2[index], base=2)))
end



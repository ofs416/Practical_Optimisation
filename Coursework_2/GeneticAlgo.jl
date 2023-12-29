using Printf
using Statistics

include("KBFunc.jl")


function pop_initial(range)
    pops = Vector{Vector{Float64}}()
    while length(pops) < 150
        sample = vec(rand(range, (2,)))
        if constraint(sample)
            push!(pops, sample)
        end
    end
    return pops
end


function standard_P_s(samples::Vector{Vector{Float64}})
    f = KBF(samples)
    f_Σ = sum(f)
    return f ./ f_Σ
end


function breed(parent1::Vector{Float64}, parent2::Vector{Float64})
    child1 = ["", ""]
    child2 = ["", ""]
    child1_final, child2_final = Float64[], Float64[]
    for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
        for (bit1, bit2) in zip(bitstring(parent1_x), bitstring(parent2_x))
            if bit1 == bit2
                child1[index] = child1[index]  * bit1
                child2[index]  = child2[index]  * bit2
            else
                temp = rand(0:1)
                child1[index]  = child1[index]  * string(temp)
                child2[index]  = child2[index]  * string(1 - temp)
            end
        end
        push!(child1_final,reinterpret(Float64, parse(Int, child1[index], base=2)))
        push!(child2_final,reinterpret(Float64, parse(Int, child2[index], base=2)))
    end
    return child1_final, child2_final
end


function breed_pop(population::Vector{Vector{Float64}})
    probs = standard_P_s(population)
    dist = Categorical(probs)
    new_pop = Vector{Vector{Float64}}()
    while length(new_pop) < 150
        offspring1, offspring2 = breed(population[rand(dist)], population[rand(dist)])
        if constraint(offspring1) & (length(new_pop) < 150)
                push!(new_pop, offspring1)
        end
        if constraint(offspring2) & (length(new_pop) < 150)
            push!(new_pop, offspring2)
        end
    end
    return new_pop
end

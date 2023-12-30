using Statistics, Distributions

include("KBFunc.jl")


function pop_initial(range, pop_size::Int)
    pops = Vector{Vector{Float64}}()
    while length(pops) < pop_size
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
    return f ./ f_Σ, f_Σ
end


function mutate(bit1::String, bit2::String, mut_prob::Float64)
    dist = Categorical([1.0-mut_prob, mut_prob])
    rand_num = rand(dist, 2)
    if rand_num[1] == 2
        bit1 = string(1 - parse(Int, bit1))
    end
    if rand_num[2] == 2
        bit2 = string(1 - parse(Int, bit2))
    end
    return bit1, bit2
end


function var_crossover(bit1::Char, bit2::Char)
    if bit1 != bit2
        temp = rand(0:1, 2)
        bit1 = string(temp[1])
        bit2 = string(temp[2])
    end
    return bit1, bit2
end


function breed_mut(parent1::Vector{Float64}, parent2::Vector{Float64}, crossover, mut_prob::Float64)
    child1 = ["", ""]
    child2 = ["", ""]
    child1_final, child2_final = Float64[], Float64[]
    for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
        for (bit1, bit2) in zip(bitstring(parent1_x)[2:end], bitstring(parent2_x)[2:end])
            bit1, bit2 = crossover(bit1, bit2)
            bit1, bit2 = mutate(bit1, bit2, mut_prob)
            child1[index] = child1[index] * bit1
            child2[index] = child2[index] * bit2
        end
        push!(child1_final,reinterpret(Float64, parse(Int, child1[index], base=2)))
        push!(child2_final,reinterpret(Float64, parse(Int, child2[index], base=2)))
    end
    return child1_final, child2_final
end



function single_iteration(popu::Vector{Vector{Float64}}, crossover, mut_prob::Float64)
    pop_size = length(popu)
    probs, f_Σ = standard_P_s(popu)
    dist = Categorical(probs)
    new_pop = Vector{Vector{Float64}}()
    while length(new_pop) < pop_size
        offspring1, offspring2 = breed_mut(popu[rand(dist)], popu[rand(dist)],crossover, mut_prob)
        if constraint(offspring1) & (length(new_pop) < pop_size)
                push!(new_pop, offspring1)
        end
        if constraint(offspring2) & (length(new_pop) < pop_size)
            push!(new_pop, offspring2)
        end
    end
    return new_pop, f_Σ
end

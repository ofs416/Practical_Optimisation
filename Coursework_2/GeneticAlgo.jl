using Statistics, StatsBase

include("KBFunc.jl")


function pop_initial(range, pop_size::Int)::Vector{Vector{Float64}}
    pops = Vector{Vector{Float64}}()
    while length(pops) < pop_size
        sample = vec(rand(range, (2,)))
        #if constraint(sample)
        push!(pops, sample)
        #end
    end
    return pops
end


function score_sum(f::Vector{Float64})::Float64
    return sum(f) / length(f)
end


function score_top5(f::Vector{Float64})::Float64
    return sum(partialsort(f , 1:5, rev=true)) / 5
end


function prop_selction(f::Vector{Float64})::Vector{Float64}
    f_Σ = sum(f)
    prob = f ./ f_Σ
    if sum(prob) != 1.00
        prob[1] += (1-sum(prob))
    end
    return prob
end


function roulette_parents(f::Vector{Float64}, prob_method)::Vector{Vector{Int64}}
    probs = prob_method(f)
    parents = sample(1:length(f), Weights(probs), (2,50))
    parents = [[a,b] for (a,b) in eachcol(parents)]
    #if 1 in [argmax(f) in pair for pair in parents] == false
    #    pair[1] = [argmax(f), rand(dist, (2, 1))]
    #end 
    return parents
end


function locus_crossover(bit1::Char, bit2::Char, pos::Int, locus::Int)::Tuple{Char, Char}
    if pos <= locus
        bit1, bit2 = bit2, bit1 
    end
    return bit1, bit2
end


function var_locus_crossover(bit1::Char, bit2::Char, pos::Int, locus::Int)::Tuple{Char, Char}
    if bit1 != bit2
        bit1, bit2 = locus_crossover(bit1, bit2, pos, locus)
    end
    return bit1, bit2
end


function var_rand_crossover(bit1::Char, bit2::Char, args...)::Tuple{Char, Char}
    if bit1 != bit2
        temp = rand(0:1)
        bit1 = only(string(temp))
        bit2 = only(string(1 - temp))
    end
    return bit1, bit2
end


function mutate(bit1::Char, bit2::Char, mut_prob::Float64)::Tuple{Char, Char}
    rand_num = sample(1:2, Weights([1.0-mut_prob, mut_prob]), 2)
    if rand_num[1] == 2
        bit1 = only(string(1 - parse(Int, bit1)))
    end
    if rand_num[2] == 2
        bit2 = only(string(1 - parse(Int, bit2)))
    end
    return bit1, bit2
end


function breed_mut(parent1::Vector{Float64}, parent2::Vector{Float64}, crossover, mut_prob::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    child1, child2 = String["", ""], String["", ""]
    child1_final, child2_final = Vector{Float64}(undef, 2), Vector{Float64}(undef, 2)
    for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
        locus = rand(1:64)
        for (pos, (bit1, bit2)) in enumerate(zip(bitstring(parent1_x)[2:end], bitstring(parent2_x)[2:end]))
            bit1, bit2 = crossover(bit1, bit2, pos, locus)
            bit1, bit2 = mutate(bit1, bit2, mut_prob)
            child1[index] = child1[index] * bit1
            child2[index] = child2[index] * bit2
        end
        child1_final[index] = reinterpret(Float64, parse(Int, child1[index], base=2))
        child2_final[index] = reinterpret(Float64, parse(Int, child2[index], base=2))
    end
    return child1_final, child2_final
end


function single_iteration(popu::Vector{Vector{Float64}}, f::Vector{Float64}, crossover, mut_prob::Float64)::Tuple{Vector{Vector{Float64}}, Vector{Float64}}
    pop_size = length(popu)
    selected_parents_indices = roulette_parents(f, prop_selction)
    new_pop = Vector{Vector{Float64}}()
    pair = 1
    while length(new_pop) < pop_size
        if pair > length(selected_parents_indices)
            pair = 1
        end
        parent1, parent2 = popu[selected_parents_indices[pair]]
        offspring1, offspring2 = breed_mut(parent1, parent2, crossover, mut_prob)
        if (offspring1[1] >= 0) && (offspring1[1] <= 10) && (offspring1[2] >= 0) && (offspring1[2] <= 10) && (length(new_pop) < pop_size)
            push!(new_pop, offspring1)
        end
        if (offspring2[1] >= 0) && (offspring2[1] <= 10) && (offspring2[2] >= 0) && (offspring2[2] <= 10) && (length(new_pop) < pop_size)
            push!(new_pop, offspring2)
        end
        pair += 1
    end
    return new_pop, KBF(new_pop)
end

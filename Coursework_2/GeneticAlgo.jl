using Statistics, Distributions

include("KBFunc.jl")


function pop_initial(range, pop_size::Int)::Vector{Vector{Float64}}
    pops = Vector{Vector{Float64}}()
    while length(pops) < pop_size
        sample = vec(rand(range, (2,)))
        if constraint(sample)
            push!(pops, sample)
        end
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
    dist = Categorical(probs)
    parents = rand(dist, (2, round(Int, length(f)/2)))
    parents = [[a,b] for (a,b) in eachcol(parents)]
    return parents
end



function mutate(bit1::Char, bit2::Char, mut_prob::Float64)::Tuple{Char, Char}
    dist = Categorical([1.0-mut_prob, mut_prob])
    rand_num = rand(dist, 2)
    if rand_num[1] == 2
        bit1 = only(string(1 - parse(Int, bit1)))
    end
    if rand_num[2] == 2
        bit2 = only(string(1 - parse(Int, bit2)))
    end
    return bit1, bit2
end


function var_crossover(bit1::Char, bit2::Char)::Tuple{Char, Char}
    if bit1 != bit2
        temp = rand(0:1, 2)
        bit1 = only(string(temp[1]))
        bit2 = only(string(temp[2]))
    end
    return bit1, bit2
end


function breed_mut(parent1::Vector{Float64}, parent2::Vector{Float64}, crossover, mut_prob::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    child1, child2 = String["", ""], String["", ""]
    child1_final, child2_final = Vector{Float64}(undef, 2), Vector{Float64}(undef, 2)
    for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
        for (bit1, bit2) in zip(bitstring(parent1_x)[2:end], bitstring(parent2_x)[2:end])
            bit1, bit2 = crossover(bit1, bit2)
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
    selected_parents = roulette_parents(f, prop_selction)
    new_pop = Vector{Vector{Float64}}()
    pair = 1
    while length(new_pop) < pop_size
        if pair > length(selected_parents)
            pair = 1
        end
        parent1, parent2 = popu[selected_parents[pair]]
        offspring1, offspring2 = breed_mut(parent1, parent2, crossover, mut_prob)
        if constraint(offspring1) && (length(new_pop) < pop_size)
                push!(new_pop, offspring1)
        end
        if constraint(offspring2) && (length(new_pop) < pop_size)
            push!(new_pop, offspring2)
        end
        pair += 1
    end
    return new_pop, KBF(new_pop)
end

using Statistics, StatsBase

include("KBFunc.jl")
include("Misc.jl")


function prop_Psi(f::Vector{Float64})::Vector{Float64}
    f_Σ = sum(f)
    prob = f ./ f_Σ
    return prob
end


function rank_Psi(f::Vector{Float64})::Vector{Float64}
    s = 2
    n = length(f)
    r_i = ranking(f) 
    num = @. s * (n + 1. - 2. * r_i) + 2. * (r_i - 1.)
    denom = n * (n - 1.)
    prob = num ./ denom
    return prob
end


function srswr(e_i, pop_size)
    if all(x-> x<1, e_i)
        parent_f = sample(1:pop_size, Weights(e_i), 1)[1]
    else 
        parent_f = sample(1:pop_size, 1)[1]
        while e_i[parent_f] < 1
            parent_f = sample(1:pop_size, 1)[1]
        end
        e_i[parent_f] -= 1
    end
    return parent_f
end


function roulette_parents(f::Vector{Float64}, prob_method)::Vector{Vector{Int64}}
    p_si = prob_method(f)
    parents = sample(1:length(f), Weights(p_si), (2,50))
    parents = [[a,b] for (a,b) in eachcol(parents)]
    #if 1 in [argmax(f) in pair for pair in parents] == false
    #    pair[1] = [argmax(f), rand(dist, (2, 1))]
    #end 
    return parents
end


function tournament_parents(f::Vector{Float64}, sub_size)::Vector{Vector{Int64}}
    pop_size = length(f)
    parents = Vector{Vector{Int64}}()
    while (length(parents) < floor(pop_size/2))
        sub_pop_f = sample(f, sub_size, replace=false)
        parent1_f, parent2_f = partialsort(sub_pop_f , 1:2, rev=true)
        parent1 = findfirst(parent -> parent == parent1_f, f)
        parent2 = findfirst(parent -> parent == parent2_f, f)
        push!(parents, [parent1, parent2])
    end
    return parents
end


function srsw_parents(f::Vector{Float64})::Vector{Vector{Int64}}
    pop_size = length(f)
    parents = Vector{Vector{Int}}()
    e_i = prop_Psi(f) .* pop_size
    while (length(parents) < floor(pop_size/2))
        parent11 = srswr(e_i, pop_size)
        parent22 = srswr(e_i, pop_size)
        pair = [parent11, parent22]
        push!(parents, pair)
    end
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


function mutate(bit1::Char, bit2::Char, p_m::Float64)::Tuple{Char, Char}
    rand_num = sample(1:2, Weights([1.0-p_m, p_m]), 2)
    if rand_num[1] == 2
        bit1 = only(string(1 - parse(Int, bit1)))
    end
    if rand_num[2] == 2
        bit2 = only(string(1 - parse(Int, bit2)))
    end
    return bit1, bit2
end


function breed_mut(dim::Int, parent1::Vector{Float64}, parent2::Vector{Float64}, crossover, p_c::Float64,p_m::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    child1, child2 = String["" for dim in 1:dim], String["" for dim in 1:dim]
    child1_final, child2_final = Vector{Float64}(undef, dim), Vector{Float64}(undef, dim)
    crossover_flag = sample([0, 1], Weights([1-p_c, p_c]))
    for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
        locus = rand(2:64)
        for (pos, (bit1, bit2)) in enumerate(zip(bitstring(parent1_x)[2:end], bitstring(parent2_x)[2:end]))
            if crossover_flag == 1
                bit1, bit2 = crossover(bit1, bit2, pos, locus)
            end
            bit1, bit2 = mutate(bit1, bit2, p_m)
            child1[index] = child1[index] * bit1
            child2[index] = child2[index] * bit2
        end
        child1_final[index] = reinterpret(Float64, parse(Int, child1[index], base=2))
        child2_final[index] = reinterpret(Float64, parse(Int, child2[index], base=2))
    end
    return child1_final, child2_final
end


function single_iteration(dim::Int, popu::Vector{Vector{Float64}}, f::Vector{Float64}, crossover, p_c::Float64, p_m::Float64)::Tuple{Vector{Vector{Float64}}, Vector{Float64}}
    pop_size = length(popu)
    selected_parents_indices = srsw_parents(f) #tournament_parents(f, 15) #roulette_parents(f, prop_Psi)
    new_pop = Vector{Vector{Float64}}()
    pair = 1
    while length(new_pop) < pop_size
        if pair > length(selected_parents_indices)
            pair = 1
        end
        parent1, parent2 = popu[selected_parents_indices[pair]]
        offspring1, offspring2 = breed_mut(dim::Int, parent1, parent2, crossover, p_c, p_m)
        if constraint(offspring1)  && (length(new_pop) < pop_size)
            push!(new_pop, offspring1)
        end
        if constraint(offspring2) && (length(new_pop) < pop_size)
            push!(new_pop, offspring2)
        end
        pair += 1
    end
    return new_pop, KBF(new_pop)
end

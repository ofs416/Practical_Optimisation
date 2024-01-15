using Statistics, StatsBase, Distributions, LinearAlgebra

include("KBFunc.jl")
include("Misc.jl")


mutable struct GA_Popul
    pop_size::Int
    pop_dim::Int
    positions::Vector{Vector{Float64}}
    scores::Vector{Float64}

    function GA_Popul(pop_size, pop_dim)
        positions = pop_initial(LinRange(0, 10, 1000), pop_size, pop_dim)
        scores = KBF(positions)
        return new(pop_size, pop_dim, positions, scores)
    end 

    function GA_Popul(pop_size, pop_dim, positions)
        scores = KBF(positions)
        return new(pop_size, pop_dim, positions, scores)
    end 

    function GA_Popul(pop_size, pop_dim, positions, scores)
        return new(pop_size, pop_dim, positions, scores)
    end 
end


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

function loci_crossover(bit1::Char, bit2::Char, pos::Int, loci1::Int, loci2::Int)::Tuple{Char, Char}
    if (loci1 < pos <= loci2)
        bit1, bit2 = bit2, bit1 
    end
    return bit1, bit2
end

function rand_crossover(bit1::Char, bit2::Char)::Tuple{Char, Char}
    if bit1 != bit2
        temp = rand(0:1)
        bit1 = only(string(temp))
        bit2 = only(string(1 - temp))
    end
    return bit1, bit2
end


function bin_breed(parent1::Vector{Float64}, parent2::Vector{Float64}, p_c::Float64
                    )::Tuple{Vector{Float64}, Vector{Float64}}
    if sample([0, 1], Weights([1-p_c, p_c])) == 1
        dim = length(parent1)
        child1, child2 = String["" for i in 1:dim], String["" for j in 1:dim]
        child1_final, child2_final = Vector{Float64}(undef, dim), Vector{Float64}(undef, dim)
        for (index, (parent1_x, parent2_x)) in enumerate(zip(parent1, parent2))
            locus1 = rand(2:50)
            locus2 = rand((locus1+1):64)
            for (pos, (bit1, bit2)) in enumerate(zip(bitstring(parent1_x)[2:end], bitstring(parent2_x)[2:end]))
                bit1, bit2 = loci_crossover(bit1, bit2, pos, locus1, locus2)
                child1[index] = child1[index] * bit1
                child2[index] = child2[index] * bit2
            end
            child1_final[index] = reinterpret(Float64, parse(Int, child1[index], base=2))
            child2_final[index] = reinterpret(Float64, parse(Int, child2[index], base=2))
        end
        return child1_final, child2_final
    else
        return parent1, parent2
    end
end


function cont_mutate(offspring::Vector{Float64}, p_m::Float64)::Vector{Float64}
    if sample(1:2, Weights([1.0-p_m, p_m])) == 2
        offspring = rand(0.:0.0001:10., length(offspring)) #length(offspring)*randn(length(offspring))/2 #
    end
    return offspring
end


function single_iteration(popu::GA_Popul, p_c::Float64, p_m::Float64)::GA_Popul
    selected_parents_indices = srsw_parents(popu.scores) #tournament_parents(f, 15) #roulette_parents(f, prop_Psi)
    new_pop = Vector{Vector{Float64}}()
    pair = 1
    while length(new_pop) < popu.pop_size
        if pair > length(selected_parents_indices)
            pair = 1
        end
        parent1, parent2 = popu.positions[selected_parents_indices[pair]]
        offspring1, offspring2 = bin_breed(parent1, parent2, p_c)
        ofs1, ofs2 = cont_mutate(offspring1, p_m), cont_mutate(offspring2, p_m)
        if constraint(offspring1)  && (length(new_pop) < popu.pop_size)
            push!(new_pop, offspring1)
        end
        if constraint(offspring2) && (length(new_pop) < popu.pop_size)
            push!(new_pop, offspring2)
        end
        pair += 1
    end
    popu.positions = new_pop
    popu.scores = KBF(new_pop)
    return popu
end


function GA(dim::Int, pop_size::Int, p_c::Float64, p_m::Float64, plots::Bool)
    iterations = floor(10000/pop_size)
    range = LinRange(0, 10, 1000)
    if plots
        objfunc = KBF(vec([[i,j] for i in range, j in range]))
    else  
        objfunc = Vector()
    end  
    popu = GA_Popul(pop_size, dim)
    pos_archive, val_archive = rand(15.:15., 10, dim), rand(0.:0., 10,1)
    archive_scorings = Float64[score_top1(popu.scores)]
    ga1_scorings = Float64[score_top1(popu.scores)]
    ga10_scorings = Float64[score_top10(popu.scores)]
    contscatplot(popu.positions, range, objfunc, string(0), plots)
    for iter in 1:iterations
        popu = single_iteration(popu, p_c, p_m)
        for (val, pos) in zip(popu.scores, popu.positions)
            pos_archive, val_archive = update_archives(pos_archive, val_archive, val, pos)
        end
        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(popu.positions, range, objfunc, string(iter), plots)
        end
        push!(archive_scorings, score_top1(vec(val_archive)))
        push!(ga1_scorings, score_top1(popu.scores))
        push!(ga10_scorings, score_top10(popu.scores))
    end
    contscatplot(eachrow(pos_archive), range, objfunc, "archive", plots)
    plot(0:iterations, [archive_scorings, ga1_scorings, ga10_scorings])
    savefig("Figures\\f_sum.png") 
    return [archive_scorings[end], ga1_scorings[end], ga10_scorings[end]]
end
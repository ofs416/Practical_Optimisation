using Printf
using Statistics


function score_all(f::Vector{Float64})::Float64
    return -sum(f) / length(f)
end


function score_top5(f::Vector{Float64})::Float64
    return -sum(partialsort(f , 1:5, rev=true)) / 5
end


function score_top1(f::Vector{Float64})::Float64
    return -sum(partialsort(f , 1:1, rev=true))
end


function pop_initial(range, pop_size::Int, dim::Int)::Vector{Vector{Float64}}
    pops = Vector{Vector{Float64}}()
    while length(pops) < pop_size
        sample = vec(rand(range, (dim,)))
        push!(pops, sample)
    end
    return pops
end


function ranking(f::Vector{Float64})::Vector{Float64}
    ordered_f = sort(f , rev=true)
    f_ranks = Int[]
    for i in f
        rank = findnext(x -> x==i, ordered_f, 1)
        while rank in f_ranks
            rank = findnext(x -> x==i, ordered_f, rank + 1)
        end
        push!(f_ranks, rank)
    end
    return f_ranks
end



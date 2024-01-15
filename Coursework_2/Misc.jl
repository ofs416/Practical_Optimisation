using Printf
using Statistics, StatsBase, Distributions
using LinearAlgebra


function score_all(f::Vector{Float64})::Float64
    return -sum(f) / length(f)
end


function score_top5(f::Vector{Float64})::Float64
    return -sum(partialsort(f , 1:5, rev=true)) / 5
end


function score_top10(f::Vector{Float64})::Float64
    return -sum(partialsort(f , 1:10, rev=true)) / 10
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


function swarm_initial(range, pop_size::Int, dim::Int)::Matrix{Float64}
    return rand(range, (pop_size,dim))
end


function ranking(f::Vector{Float64})::Array{Int}
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


function update_archives(pos_archive, val_archive, val, pos)

    norms = sum(norm.(pos_archive .- pos'), dims=2)
    D_min = 2
    D_sim = 0.2

    if any(x-> x<val, val_archive) && all(x-> x>D_min ,norms)
        val_archive[argmin(val_archive)] = val
        pos_archive[argmin(val_archive)[1], :] = pos
    elseif all(x-> x<val, val_archive) 
        val_archive[argmin(norms)] = val
        pos_archive[argmin(norms)[1], :] = pos
    else
        loc = findfirst(x->x==1, (norms .< D_sim) .& (val_archive .< val))
        if loc !== nothing
            val_archive[loc[1]] = val
            pos_archive[loc[1], :] = pos
        end
    end
    return pos_archive, val_archive
end

function contscatplot(pos, range, objfunc, label::String, plots::Bool)
    if plots
        contourf(range, range, objfunc, plot_title="Contour Plot", camera=(180, 30), color=:turbo)
        scatter!(Tuple.(pos), label="Population")
        savefig("Figures\\iter_" * label * ".png") 
    end
end
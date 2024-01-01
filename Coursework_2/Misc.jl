using Printf
using Statistics


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



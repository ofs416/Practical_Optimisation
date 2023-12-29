using Printf
using Statistics

include("KBFunc.jl")


function parent_initial(range)
    parents = []
    while length(parents) < 100
        sample = vec(rand(range, (2,)))
        if constraint(sample)
            push!(parents, sample)
        end
    end
    parents_final = [Float64[a, b] for (a, b) in parents]
    return parents_final
end



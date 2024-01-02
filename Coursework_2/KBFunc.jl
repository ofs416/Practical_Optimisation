using Printf
using Statistics

# Long but it meeans not all conditions are checked if the prior is already false (due to && instead of &)
function constraint(x::Vector{Float64})::Bool
    dim = length(x)
    return all(x-> 0<=x<=10, x) && (sum(x) < 7.5*dim) && (prod(x) > 0.75)
end

# Kean's Bump Funtion
function KBF(points::Union{Matrix{Float64}, Vector{Vector{Float64}}})::Vector{Float64}
    cost = Float64[]
    if typeof(points) == Vector{Vector{Float64}}
        points = mapreduce(permutedims, vcat, points)
    end
    for x in eachrow(points) 
        x = vec(copy(x))
        if constraint(x) 
            cosargs =  cos.(x)
            num =  sum(cosargs .^4) - 2prod(cosargs .^2)
            denom = sqrt(sum([index*num^2 for (index, num) in enumerate(x)]))
            push!(cost, num/denom)
        else 
            push!(cost, 0)
        end
    end
    return cost
end

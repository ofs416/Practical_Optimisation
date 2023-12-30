using Printf
using Statistics


function constraint(x::Vector{Float64})::Bool
    return (minimum(x) >= 0) && (maximum(x) <= 10) && (sum(x) < 15) && (prod(x) > 0.75)
end


function KBF(points::Vector{Vector{Float64}})::Vector{Float64}
    cost = Float64[]
    for x in points
        if constraint(x) 
            cosargs =  broadcast(cos, x)
            num =  sum(cosargs .^4) - 2prod(cosargs .^2)
            denom = sqrt(sum([index*num^2 for (index, num) in enumerate(x)]))
            push!(cost, num/denom)
        else 
            push!(cost, -Inf)
        end
    end
    return cost
end


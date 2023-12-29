using Printf
using Statistics

function KBF(points::Vector{Vector{Float16}})
    cost = Float64[]
    for x in points
        if (sum(x) >= 15*length(x)/2) | (prod(x) <=0.75)
            push!(cost, Inf64)
        else
            cosargs =  broadcast(cos, x)
            num =  sum(cosargs .^4) - 2prod(cosargs .^2)
            denom = sqrt(sum([index*num^2 for (index, num) in enumerate(x)]))
            push!(cost, num/denom)
        end
    end
    return cost
end


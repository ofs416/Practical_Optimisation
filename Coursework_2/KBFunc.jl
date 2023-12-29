using Printf
using Statistics

function constraint(x::Vector{Float64})
    return (sum(x) < 15*length(x)/2) & (prod(x) > 0.75)
end

function KBF(points::Vector{Vector{Float64}})
    cost = Float64[]
    for x in points
        if constraint(x) 
            cosargs =  broadcast(cos, x)
            num =  sum(cosargs .^4) - 2prod(cosargs .^2)
            denom = sqrt(sum([index*num^2 for (index, num) in enumerate(x)]))
            push!(cost, num/denom)
        else 
            push!(cost, Inf64)
        end
    end
    return cost
end


function standard_P_s(samples::Vector{Vector{Float64}})
    f = KBF(samples)
    f_Σ = sum(f)
    return f ./ f_Σ
end

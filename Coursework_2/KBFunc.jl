using Printf
using Statistics

# Long but it meeans not all conditions are checked if the prior is already false (due to && instead of &)
function constraint(x::Vector{Float64})::Bool
    return (x[1] >= 0) && (x[1] <= 10) && (x[2] >= 0) && (x[2] <= 10) && (sum(x) < 15) && (prod(x) > 0.75)
end

# Kean's Bump Funtion
function KBF(points::Vector{Vector{Float64}})::Vector{Float64}
    cost = Float64[]
    for x in points 
        cosargs =  cos.(x)
        num =  sum(cosargs .^4) - 2prod(cosargs .^2)
        denom = sqrt(sum([index*num^2 for (index, num) in enumerate(x)]))
        if constraint(x) 
            push!(cost, num/denom)
        else 
            push!(cost, 0)
        end
    end
    return cost
end


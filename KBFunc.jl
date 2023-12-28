using Printf
using Statistics
using Plots

function KBF(args...)
    if (sum(args) >= 15*length(args)/2) | (prod(args) <=0.75)
        return Inf
    else
        cosargs =  broadcast(cos, args)
        num =  sum(cosargs .^4) - 2prod(cosargs .^2)
        denom = sqrt(sum([index*num^2 for (index, num) in enumerate(args)]))
        return num/denom
    end
end


x = y = LinRange(0, 10, 1000)
contourf(x, y, KBF, plot_title="Contour Plot", camera=(180, 30), color=:turbo)
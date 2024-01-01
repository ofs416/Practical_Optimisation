using Printf
using Statistics, Distributions
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")



function GA(dim::Int, pop_size::Int, p_c::Float64, p_m::Float64, crossover, scoring, plots::Bool)
    iterations = floor(10000/pop_size)
    range = LinRange(0, 10, 1000)

    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)

    popu = pop_initial(range, pop_size, dim)
    f = KBF(popu)
    scores = Float64[scoring(f)]
    contscatplot(popu, range, objfunc, string(0), plots)

    for iter in 1:iterations
        popu, f = single_iteration(dim, popu, f, crossover, p_c, p_m)
        push!(scores, scoring(f))
        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(popu, range, objfunc, string(iter), plots)
        end
    end
        contscatplot(popu, range, objfunc, "final", plots)
        plot(0:iterations, scores)
        savefig("Coursework_2/Figures/f_sum.png") 
    return scores[end]
end


#avg = Matrix{Float64}(undef, (10, 3))
#stdev = Matrix{Float64}(undef, (10, 3))
#for (i, pop) in enumerate(100:100:1000)
#    for (j, mut) in enumerate([0.0001, 0.001, 0.01])
#        println(pop, mut)
#        scores = Float64[]
#        for iter in 1:75
#            push!(scores, GA(8, 200, 0.005, var_locus_crossover, score_top5, false))
#        end
#        avg[i, j] = mean(scores)
#        stdev[i, j] = std(scores)
#    end
#end

#println(avg)
#println(stdev)



#scores = Float64[]
#for iter in 1:100 
#    push!(scores, GA(8, 200, 0.005, var_locus_crossover, score_top5, false))
#end

GA(2, 500, 0.7, 0.001, var_locus_crossover, score_top1, true)

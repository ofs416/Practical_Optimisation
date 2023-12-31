using Printf
using Statistics, Distributions
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")


function contscatplot(popu, range, objfunc, label::String, plots::Bool)
    if plots
        contourf(range, range, objfunc, plot_title="Contour Plot", camera=(180, 30), color=:turbo)
        scatter!(Tuple.(popu), label="Population")
        savefig("Coursework_2/Figures/iter_" * label * ".png") 
    end
end

function GA(pop_size::Int, mut_prob::Float64, crossover, scoring, plots::Bool)::Float64
    iterations = floor(10000/pop_size)
    range = LinRange(0, 10, 1000)
    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)

    popu = pop_initial(range, pop_size)
    f = KBF(popu)
    scores = Float64[scoring(f)]
    contscatplot(popu, range, objfunc, string(0), plots)

    for iter in 1:iterations
        popu, f = single_iteration(popu, f, crossover, mut_prob)
        push!(scores, scoring(f))
        if (iter % 25 == 0) | (iter in 1:10)
            contscatplot(popu, range, objfunc, string(iter), plots)
        end
    end
        contscatplot(popu, range, objfunc, "final", plots)
        plot(0:iterations, scores)
        savefig("Coursework_2/Figures/f_sum.png") 
    return scores[end]
end


#avg = Matrix{Float64}(undef, (5, 3))
#for (i, pop) in enumerate(20:20:100)
#    for (j, mut) in enumerate([0.0001, 0.001, 0.01])
#        println(pop, mut)
#        for iter in 1:75
#            global avg[i, j] += GA(pop, mut, var_locus_crossover, score_top5, false)
#        end
#    end
#end
#avg ./= 75
#avg

avg = 0
for iter in 1:50 
    global avg += GA(80, 0.01, var_locus_crossover, score_top5, false)
end
println(avg/50)
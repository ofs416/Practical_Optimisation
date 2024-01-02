using Printf
using Statistics, Distributions
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")
include("ParticleSwarm.jl")



function GA(dim::Int, pop_size::Int, p_c::Float64, p_m::Float64, crossover, scoring, plots::Bool)
    iterations = floor(10000/pop_size)
    range = LinRange(0, 10, 1000)

    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)

    popu = GA_Popul(pop_size, dim)
    scorings = Float64[scoring(popu.scores)]
    contscatplot(popu.positions, range, objfunc, string(0), plots)

    for iter in 1:iterations
        popu = single_iteration(popu, crossover, p_c, p_m)
        push!(scorings, scoring(popu.scores))
        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(popu.positions, range, objfunc, string(iter), plots)
        end
    end
    contscatplot(popu.positions, range, objfunc, "final", plots)
    plot(0:iterations, scorings)
    savefig("Coursework_2/Figures/f_sum.png") 
    return scorings[end]
end


function PS(dim::Int, pop_size::Int, innertia::Float64, phi_p::Float64, phi_g::Float64, scoring, plots::Bool)
    iterations = floor(10000/pop_size)
    range = LinRange(0, 10, 1000)
    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)

    test_swarm = Swarm_Popul(pop_size,dim)
    scorings = Float64[scoring(test_swarm.val)]
    contscatplot(eachrow(test_swarm.pos), range, objfunc, string(0), plots)
    for iter in 1:iterations
        test_swarm = update_velocity(test_swarm, innertia, phi_p, phi_g)
        test_swarm = update_positions(test_swarm)
        push!(scorings, scoring(test_swarm.val))
        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(eachrow(test_swarm.pos), range, objfunc, string(iter), plots)
        end
    end 
    plot(0:iterations, scorings)
    savefig("Coursework_2/Figures/f_sum.png") 
    return scorings[end]
end

# PS(2, 500, 0.3, 0.48, 0.5, score_top1, true)

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



scores = Float64[]
for iter in 1:100 
    push!(scores, PS(2, 500, 0.3, 0.48, 0.5, score_top1, false))
end
println(mean(scores))
#GA(2, 250, 0.7, 0.001, var_locus_crossover, score_top1, true)

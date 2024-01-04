using Printf
using Statistics, Distributions
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")
include("ParticleSwarm.jl")


function GAPSO(
                dim::Int, pop_size::Int, innertia::Float64, phi_p::Float64,
                phi_g::Float64, p_c::Float64, p_m::Float64, crossover, scoring,
                plots::Bool
                )
    iterations = floor(Int, 10000/pop_size)
    range = LinRange(-2, 12, 1000)
    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)

    swarm = Swarm_Popul(pop_size, dim, innertia, phi_p, phi_g)
    gapopu = GA_Popul(pop_size, dim)
    swarm.pos = mapreduce(permutedims, vcat, gapopu.positions)

    scorings = Float64[scoring(swarm.val)]
    contscatplot(eachrow(swarm.pos), range, objfunc, string(0), plots)
    contscatplot(gapopu.positions, range, objfunc, string(0.0), plots)

    for iter in 1:iterations
        if 0 < iter % 4 <= 2
            swarm = update_velocity(swarm)
            swarm = update_positions(swarm)
            swarm = update_pos_archives(swarm)
            swarm = update_hparams(swarm, iterations)
            vec_pos = [row[:] for row in eachrow(swarm.pos)]
            gapopu = GA_Popul(pop_size, dim, vec_pos, swarm.val)
            push!(scorings, scoring(swarm.val))
        else
            swarm = update_velocity(swarm)
            gapopu = single_iteration(gapopu, crossover, p_c, p_m)
            swarm.pos = mapreduce(permutedims, vcat, gapopu.positions) 
            swarm.val_temp = gapopu.scores
            swarm = update_pos_archives(swarm)
            swarm = update_hparams(swarm, iterations)
            push!(scorings, scoring(gapopu.scores))
        end
        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(eachrow(swarm.pos), range, objfunc, string(iter), plots)
        end
    end 

    plot(0:iterations, scorings)
    savefig("Figures\\f.png") 

    return scorings[end]
end


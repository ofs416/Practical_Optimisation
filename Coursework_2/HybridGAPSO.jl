using Printf
using Statistics, Distributions
using Plots
using BenchmarkTools

include("KBFunc.jl")
include("GeneticAlgo.jl")
include("Misc.jl")
include("ParticleSwarm.jl")


function SGAPSO(
                dim::Int, pop_size::Int, innertia::Float64, phi_p::Float64,
                phi_g::Float64, p_c::Float64, p_m::Float64, crossover,
                plots::Bool
                )
    iterations = floor(Int, 10000/pop_size)
    range = LinRange(-2, 12, 1000)
    if plots
        objfunc = KBF(vec([[i,j] for i in range, j in range]))
    else  
        objfunc = Vector()
    end  

    swarm = Swarm_Popul(pop_size, dim, innertia, phi_p, phi_g)
    gapopu = GA_Popul(pop_size, dim)
    swarm.pos = mapreduce(permutedims, vcat, gapopu.positions)

    archive_scorings = Float64[score_top1(vec(swarm.val_archive))]
    swarm1_scorings = Float64[score_top1(swarm.val)]
    swarm10_scorings = Float64[score_top10(swarm.val)]
    contscatplot(eachrow(swarm.pos), range, objfunc, string(0), plots)

    for iter in 1:iterations
        if 0 < iter % 10 <= 5
            swarm = update_velocity(swarm)
            swarm = update_positions(swarm)
            swarm = update_pos_archives(swarm)
            swarm = update_hparams(swarm, iterations)
            vec_pos = [row[:] for row in eachrow(swarm.pos)]
            gapopu = GA_Popul(pop_size, dim, vec_pos, swarm.val)
        else
            swarm = update_velocity(swarm)
            gapopu = single_iteration(gapopu, crossover, p_c, p_m)
            swarm.pos = mapreduce(permutedims, vcat, gapopu.positions) 
            swarm.val = gapopu.scores
            swarm = update_pos_archives(swarm)
            swarm = update_hparams(swarm, iterations)
        end

        push!(archive_scorings, score_top1(vec(swarm.val_archive)))
        push!(swarm1_scorings, score_top1(swarm.val))
        push!(swarm10_scorings, score_top10(swarm.val))

        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(eachrow(swarm.pos), range, objfunc, string(iter), plots)
        end
    end 

    contscatplot(eachrow(swarm.pos_archive), range, objfunc, "archive", plots)
    plot(0:iterations, [archive_scorings, swarm1_scorings, swarm10_scorings])
    savefig("Figures\\f_sum.png") 

    return [archive_scorings[end], swarm1_scorings[end], swarm10_scorings[end]]
end


function PGAPSO(
    dim::Int, pop_size::Int, innertia::Float64, phi_p::Float64,
    phi_g::Float64, p_c::Float64, p_m::Float64, crossover,
    plots::Bool
    )
    iterations = floor(Int, 10000/pop_size)
    range = LinRange(-2, 12, 1000)
    if plots
        objfunc = KBF(vec([[i,j] for i in range, j in range]))
    else  
        objfunc = Vector()
    end  

    swarm = Swarm_Popul(pop_size, dim, innertia, phi_p, phi_g)
    archive_scorings = Float64[score_top1(vec(swarm.val_archive))]
    swarm1_scorings = Float64[score_top1(swarm.val)]
    swarm10_scorings = Float64[score_top10(swarm.val)]
    contscatplot(eachrow(swarm.pos), range, objfunc, string(0), plots)

    for iter in 1:iterations
        sampleindex = sample(1:pop_size, (2, floor(Int,pop_size/2)), replace=false)
        swarm = update_velocity(swarm)

        vec_pos = [row[:] for row in eachrow(swarm.pos[sampleindex[1,:], :])]
        gapopu = GA_Popul(floor(Int,pop_size/2), dim, vec_pos, swarm.val[sampleindex[1,:]])
        gapopu = single_iteration(gapopu, crossover, p_c, p_m)
        swarm.pos[sampleindex[1,:], :] = mapreduce(permutedims, vcat, gapopu.positions) 

        swarm.pos[sampleindex[2,:], :] = max.(min.(swarm.pos[sampleindex[2,:], :] + 
                                                   swarm.vels[sampleindex[2,:], :], 10), 0)  
        
        swarm.val = KBF(swarm.pos)
        swarm = update_pos_archives(swarm)
        swarm = update_hparams(swarm, iterations)

    
        push!(archive_scorings, score_top1(vec(swarm.val_archive)))
        push!(swarm1_scorings, score_top1(swarm.val))
        push!(swarm10_scorings, score_top10(swarm.val))

        if (iter % 10 == 0) | (iter in 1:10)
        contscatplot(eachrow(swarm.pos), range, objfunc, string(iter), plots)
        end
    end 

    contscatplot(eachrow(swarm.pos_archive), range, objfunc, "archive", plots)
    plot(0:iterations, [archive_scorings, swarm1_scorings, swarm10_scorings])
    savefig("Figures\\f_sum.png") 

    return [archive_scorings[end], swarm1_scorings[end], swarm10_scorings[end]]
end
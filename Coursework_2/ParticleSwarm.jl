using Statistics, StatsBase
using LinearAlgebra

include("KBFunc.jl")
include("Misc.jl")


mutable struct Swarm_Popul
    pop_size::Int
    pop_dim::Int
    innertia::Float64
    phi_p::Float64
    phi_g::Float64
    innertia_ini::Float64
    phi_p_ini::Float64
    phi_g_ini::Float64
    innertia_temp::Matrix{Float64}
    cog_temp::Matrix{Float64}
    social_temp::Matrix{Float64}
    pos::Matrix{Float64}
    vels::Matrix{Float64}
    part_best_pos::Vector{Union{Matrix{Float64}, Vector{Float64}}}
    sw_best_pos::Tuple{Vector{Float64}, Float64}
    val::Vector{Float64}
    pos_archive::Matrix{Float64}
    val_archive::Matrix{Float64}

    function Swarm_Popul(pop_size, pop_dim, innertia, phi_p, phi_g)
        innertia_ini, phi_p_ini, phi_g_ini = innertia, phi_p, phi_g
        pos = rand(LinRange(0, 10, 1000), pop_size, pop_dim)
        val = KBF(pos)
        part_best_pos = [pos, val]
        sw_best_pos = (pos[argmax(val), :], maximum(val))
        vels = rand(LinRange(-10, 10, 1000), pop_size, pop_dim)
        innertia_temp, cog_temp = Matrix{Float64}(undef, 1,1), Matrix{Float64}(undef, 1,1) 
        social_temp = Matrix{Float64}(undef, 1,1)
        pos_archive = rand(15.:15., 10, pop_dim)
        val_archive = rand(0.:0., 10,1)
        return new(
                    pop_size, pop_dim, innertia, phi_p, phi_g, innertia_ini, phi_p_ini, 
                    phi_g_ini, innertia_temp, cog_temp, social_temp, pos, vels, 
                    part_best_pos, sw_best_pos, val, pos_archive, val_archive)
    end 
end


function update_velocity(swarm::Swarm_Popul)::Swarm_Popul
    swarm.innertia_temp = swarm.innertia .* swarm.vels
    swarm.cog_temp = swarm.phi_p .* rand(0:0.00001:1, swarm.pop_size, swarm.pop_dim) .*
                     (swarm.part_best_pos[1] - swarm.pos)
    swarm.social_temp = swarm.phi_g .* rand(0:0.00001:1, swarm.pop_size, swarm.pop_dim) .* 
                        (reshape(swarm.sw_best_pos[1], (1, swarm.pop_dim)) .- swarm.pos)
    swarm.vels = max.(min.(swarm.innertia_temp  + swarm.cog_temp + swarm.social_temp, 4), -4)
    return swarm
end


function update_positions(swarm::Swarm_Popul)::Swarm_Popul
    swarm.pos = max.(min.(swarm.pos + swarm.vels, 10), 0)  
    swarm.val = KBF(swarm.pos)
    return swarm
end

function update_pos_archives(swarm::Swarm_Popul)::Swarm_Popul
    for (index, (val1, val2)) in enumerate(zip(swarm.val, swarm.part_best_pos[2]))
        if val1 > val2 
            swarm.part_best_pos[1][index, :] = swarm.pos[index, :]
        end
        if val1 > swarm.sw_best_pos[2]
            swarm.sw_best_pos = (swarm.pos[index, :], val1)
        end
        swarm.pos_archive, swarm.val_archive =
         update_archives(swarm.pos_archive, swarm.val_archive, val1, swarm.pos[index, :])
    end
    return swarm
end


function update_hparams(swarm::Swarm_Popul, iters::Int)::Swarm_Popul
    swarm.innertia = swarm.innertia - (0.5 * swarm.innertia_ini / iters)
    swarm.phi_p = swarm.phi_p - (0.5 * swarm.phi_p_ini / iters)
    swarm.phi_g = swarm.phi_g + (swarm.phi_g_ini / iters)
    return swarm
end


function PSO(dim::Int, pop_size::Int, innertia::Float64, phi_p::Float64,
            phi_g::Float64, plots::Bool)
    iterations = floor(Int, 10000/pop_size)
    range = LinRange(-2, 12, 1000)
    if plots
        objfunc = KBF(vec([[i,j] for i in range, j in range]))
    else  
        objfunc = Vector()
    end  

    swarm = Swarm_Popul(pop_size, dim, innertia, phi_p, phi_g)
    archive_scorings = Float64[score_top1(swarm.val)]
    swarm1_scorings = Float64[score_top1(swarm.val)]
    swarm10_scorings = Float64[score_top10(swarm.val)]
    contscatplot(eachrow(swarm.pos), range, objfunc, string(0), plots)
    for iter in 1:iterations
        swarm = update_velocity(swarm)
        swarm = update_positions(swarm)
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


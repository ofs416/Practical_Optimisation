using Statistics, StatsBase

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
    part_best_pos::Matrix{Float64}
    sw_best_pos::Tuple{Vector{Float64}, Float64}
    val::Vector{Float64}
    val_temp::Vector{Float64}

    function Swarm_Popul(pop_size, pop_dim, innertia, phi_p, phi_g)
        innertia_ini, phi_p_ini, phi_g_ini = innertia, phi_p, phi_g
        pos = rand(LinRange(0, 10, 1000), pop_size, pop_dim)
        part_best_pos = pos
        val = KBF(pos)
        sw_best_pos = (pos[argmax(val), :], maximum(val))
        vels = rand(LinRange(-10, 10, 1000), pop_size, pop_dim)
        innertia_temp, cog_temp = Matrix{Float64}(undef, 1,1), Matrix{Float64}(undef, 1,1) 
        social_temp, val_temp = Matrix{Float64}(undef, 1,1), Vector{Float64}(undef, 1) 
        return new(
                    pop_size, pop_dim, innertia, phi_p, phi_g, innertia_ini, phi_p_ini, 
                    phi_g_ini, innertia_temp, cog_temp, social_temp, pos, vels, 
                    part_best_pos, sw_best_pos, val, val_temp
                   )
    end 
end


function update_velocity(swarm::Swarm_Popul)::Swarm_Popul
    swarm.innertia_temp = swarm.innertia .* swarm.vels
    swarm.cog_temp = swarm.phi_p .* rand(0:0.00001:1, swarm.pop_size, swarm.pop_dim) .* (swarm.part_best_pos - swarm.pos)
    swarm.social_temp = swarm.phi_g .* rand(0:0.00001:1, swarm.pop_size, swarm.pop_dim) .* (reshape(swarm.sw_best_pos[1], (1, swarm.pop_dim)) .- swarm.pos)
    swarm.vels = max.(min.(swarm.innertia_temp  + swarm.cog_temp + swarm.social_temp, 3), -3)
    return swarm
end


function update_positions(swarm::Swarm_Popul)::Swarm_Popul
    swarm.pos = max.(min.(swarm.pos + swarm.vels, 10), 0)  
    swarm.val_temp = KBF(swarm.pos)
    return swarm
end

function update_pos_archives(swarm::Swarm_Popul)::Swarm_Popul
    for (index, (val1, val2)) in enumerate(zip(swarm.val_temp, swarm.val))
        if val1 > val2 
            swarm.part_best_pos[index, :] = swarm.pos[index, :]
        end
        if val1 > swarm.sw_best_pos[2]
            swarm.sw_best_pos = (swarm.pos[index, :], val1)
        end
    end
    swarm.val = swarm.val_temp
    return swarm
end


function update_hparams(swarm::Swarm_Popul, iters::Int)::Swarm_Popul
    swarm.innertia = swarm.innertia - (0.5 * swarm.innertia_ini / iters)
    swarm.phi_p = swarm.phi_p - (0.5 * swarm.phi_p_ini / iters)
    swarm.phi_g = swarm.phi_g + (swarm.phi_g_ini / iters)
    return swarm
end


function PS(dim::Int, pop_size::Int, innertia::Float64, phi_p::Float64, phi_g::Float64, scoring, plots::Bool)
    iterations = floor(Int, 10000/pop_size)
    range = LinRange(-2, 12, 1000)
    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)

    test_swarm = Swarm_Popul(pop_size, dim, innertia, phi_p, phi_g)
    scorings = Float64[scoring(test_swarm.val)]
    contscatplot(eachrow(test_swarm.pos), range, objfunc, string(0), plots)
    for iter in 1:iterations
        test_swarm = update_velocity(test_swarm)
        test_swarm = update_positions(test_swarm)
        test_swarm = update_pos_archives(test_swarm)
        test_swarm = update_hparams(test_swarm, iterations)
        push!(scorings, scoring(test_swarm.val))
        if (iter % 10 == 0) | (iter in 1:10)
            contscatplot(eachrow(test_swarm.pos), range, objfunc, string(iter), plots)
        end
    end 

    plot(0:iterations, scorings)
    savefig("Figures\\f_sum.png") 

    return scorings[end]
end


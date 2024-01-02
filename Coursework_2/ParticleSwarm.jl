using Statistics, StatsBase

include("KBFunc.jl")
include("Misc.jl")


mutable struct Swarm_Popul
    pop_size::Int
    pop_dim::Int
    pos::Matrix{Float64}
    vels::Matrix{Float64}
    part_best_pos::Matrix{Float64}
    sw_best_pos::Tuple{Vector{Float64}, Float64}
    val::Vector{Float64}

    function Swarm_Popul(pop_size, pop_dim)
        pos = rand(LinRange(0, 10, 1000), pop_size, pop_dim)
        part_best_pos = pos
        val = KBFM(pos)
        sw_best_pos = (pos[argmax(val), :], maximum(val))
        vels = rand(LinRange(-10, 10, 1000), pop_size, pop_dim)
        return new(pop_size, pop_dim, pos, vels, part_best_pos, sw_best_pos, val)
    end 
end


function update_velocity(swarm::Swarm_Popul, innertia::Float64, phi_p::Float64, phi_g::Float64)::Swarm_Popul
    r_p = rand(0:0.00001:1, swarm.pop_size, swarm.pop_dim)
    r_g = rand(0:0.00001:1, swarm.pop_size, swarm.pop_dim)
    first = innertia .* swarm.vels
    second = phi_p .* r_p .* (swarm.part_best_pos - swarm.pos)
    third = phi_g .* r_g .* (reshape(swarm.sw_best_pos[1], (1, swarm.pop_dim)) .- swarm.pos)
    swarm.vels = first + second + third
    return swarm
end


function update_positions(swarm::Swarm_Popul)::Swarm_Popul
    swarm.pos += swarm.vels
    vals = KBFM(swarm.pos)
    for (index, (val1, val2)) in enumerate(zip(vals, swarm.val))
        if val1 > val2 
            swarm.part_best_pos[index, :] = swarm.pos[index, :]
        end
        if val1 > swarm.sw_best_pos[2]
            swarm.sw_best_pos = (swarm.pos[index, :], val1)
        end
    end
    swarm.val = vals
    return swarm
end



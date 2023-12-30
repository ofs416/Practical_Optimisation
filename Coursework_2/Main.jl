using Printf
using Statistics, Distributions
using Plots

include("KBFunc.jl")
include("GeneticAlgo.jl")


function contscatplot(popu, range, objfunc, label::String)
    contourf(range, range, objfunc, plot_title="Contour Plot", camera=(180, 30), color=:turbo)
    scatter!(Tuple.(popu), label="Population")
    savefig("Coursework_2/Figures/iter_" * label * ".png") 
end

function GA(pop_size::Int, crossover, mut_prob::Float64)
    range = LinRange(0, 10, 1000)
    z = vec([[i,j] for i in range, j in range])
    objfunc = KBF(z)
    popu = pop_initial(range, pop_size)
    contscatplot(popu, range, objfunc, string(0))

    f_Σ = Float64[standard_P_s(popu)[2]]
    for iter in 1:(10000/pop_size)
        popu, temp = single_iteration(popu,crossover, mut_prob)
        push!(f_Σ, temp)
        if iter % 20 == 0
            contscatplot(popu, range, objfunc, string(iter))
        end
    end
    contscatplot(popu, range, objfunc, "final")

    plot(0:(10000/pop_size), f_Σ)
    savefig("Coursework_2/Figures/f_sum.png") 
    return f_Σ[end]/pop_size
end

GA(100, var_crossover, 0.001)
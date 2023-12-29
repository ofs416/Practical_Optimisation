using Printf
using Statistics, Distributions
using Plots

include("KBFunc.jl")
include("GeneticAlgo.jl")


range = LinRange(0, 10, 1000)
z = vec([[i,j] for i in range, j in range])
println(maximum(KBF(z)))

popu = pop_initial(range)

contourf(range, range, KBF(z), plot_title="Contour Plot", camera=(180, 30), color=:turbo)
scatter!(Tuple.(popu), label="Population")
savefig("Coursework_2/Figures/iter_0.png") 

for iter in 1:500
    global popu = breed_pop(popu)
    if iter % 100 == 0
        contourf(range, range, KBF(z), plot_title="Contour Plot", camera=(180, 30), color=:turbo)
    scatter!(Tuple.(popu), label="Population")
    savefig("Coursework_2/Figures/iter_" * string(iter) * ".png") 
    end
end

contourf(range, range, KBF(z), plot_title="Contour Plot", camera=(180, 30), color=:turbo)
scatter!(Tuple.(popu), label="Final Population")


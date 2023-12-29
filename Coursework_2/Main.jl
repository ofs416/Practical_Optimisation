using Printf
using Statistics, StaticArrays
using Plots

include("KBFunc.jl")
include("GeneticAlgo.jl")


range = LinRange(0, 10, 1000)
z = vec([[i,j] for i in range, j in range])


parents = parent_initial(range)


contourf(range, range, KBF(z), plot_title="Contour Plot", camera=(180, 30), color=:turbo)
scatter!(Tuple.(parents), label="Parents")

range = LinRange(0, 10, 1000)
parents = parent_initial(range)
sum(standard_P_s(parents))
using Printf
using Statistics
using Plots

include("KBFunc.jl")


x_1 = x_2 = LinRange(0, 10, 1000)
pop_grid = [0 for i in x_1, j in x_2]
z = [KBF(i, j) for i in x_1, j in x_2]

contourf(x, y, z, plot_title="Contour Plot", camera=(180, 30), color=:turbo)


struct GA
    obj_func_iter::Int16
    
end
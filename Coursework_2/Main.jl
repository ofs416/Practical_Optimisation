using Printf
using Statistics
using Plots

include("KBFunc.jl")


x_1 = x_2 = LinRange(0, 10, 1000)
pop_grid = [0 for i in x_1, j in x_2]

contourf(x_1, x_2, KBF, plot_title="Contour Plot", camera=(180, 30), color=:turbo)



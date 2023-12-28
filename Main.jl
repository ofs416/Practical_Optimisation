using Printf
using Statistics
using Plots

include("KBFunc.jl")


x = y = LinRange(0, 10, 1000)
contourf(x, y, KBF, plot_title="Contour Plot", camera=(180, 30), color=:turbo)
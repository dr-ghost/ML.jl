module ML

using Statistics, LinearAlgebra, StaticArrays, Plots

export greet, Linear_Regression

include("Tensor.jl")
include("nn.jl")
include("loss/Losses.jl")
include("models/Model.jl")
include("autograd.jl")
include("data/Data.jl")
include("optimizers/optimizers.jl")
include("Train.jl")

include("models/Linear_Regression.jl")

greet() = "Welcome .. Welcome to Hogwarts lil' slytherin!!! "

end

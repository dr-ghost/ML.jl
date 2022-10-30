module ML

export greet, Linear_Regression

include("Tensor.jl")
include("nn.jl")
include("models/Model.jl")
include("autograd.jl")
include("loss/Losses.jl")
include("data/Data.jl")
include("optimizers/optimizers.jl")
include("Train.jl")

include("models/Linear_Regression.jl")

greet() = "Hello AI Warriors"

end

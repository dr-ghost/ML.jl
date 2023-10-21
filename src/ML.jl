module ML

export greet, CNode, Node, INode, ONode, addNode, CGraph, computeNode, gradNode, makeGraph!, computeGraph, gradGraph, Tensor

include("Tensor.jl")
include("nn.jl")
include("loss/Losses.jl")
include("models/Model.jl")
include("compgr/cgraph.jl")
include("compgr/cgraphfn.jl")
include("data/Data.jl")
include("optimizers/optimizers.jl")
include("Train.jl")

include("models/Linear_Regression.jl")

greet() = "Welcome .. Welcome to Hogwarts lil' slytherin!!! "

using ._Tensor_

node1 = addNode()
node2 = addNode(convert(Array{CNode, 1}, [node1]))

graph = CGraph(convert(Array{CNode, 1},[node1, node2]))

idict = Dict((node1, 1) => Tensor([2.0 ; 4.0]), (node1, 2) => Tensor([2.0 ; 4.0]), (node2, 2) => Tensor([3.0 ; 4.0]))
computeGraph(graph, convert(Dict{Tuple{CNode, Int}, SArray}, idict))

println(graph.output_nodes[1].output)

end


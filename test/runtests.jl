using Test, ML, DataStructures, StaticArrays, ML._Tensor_

node1 = addNode()
node2 = addNode(convert(Array{CNode, 1}, [node1]))

graph = CGraph(convert(Array{CNode, 1},[node1, node2]))

idict = Dict((node1, 1) => Tensor([2.0 ; 4.0]), (node1, 2) => Tensor([2.0 ; 4.0]), (node2, 2) => Tensor([3.0 ; 4.0]))
computeGraph(graph, convert(Dict{Tuple{CNode, Int}, SArray}, idict))

println(graph.output_nodes[1].output)

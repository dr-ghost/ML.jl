using DataStructures

# Node functions template
function computeNode(node::CNode)
    raise("Not implemented")
end
function computeGrad(node::CNode)
    raise("Not implemented")
end
function ∇(node::CNode)
    raise("Not implemented")
end

#Inode
function computeNode(node::INode, x::SArray)
    node.output = x
end

#Onode
function computeNode(node::ONode)
    node.output = node.input_nodes[1].output
end

#add Node
function computeNode(node::addNode)
    t = SArray{Tuple{size(node.input_nodes[1].output)...}}(zeros(size(node.input_nodes[1].output)))
    for inode in node.input_nodes
        t = t + inode.output
    end
    node.output = t
end

function computeGrad(node::addNode, out_idx::Int64, graph::CGraph)
    
end

function ∇(node::addNode)::SVector
        
end

# Graph functions
function makeGraph!(graph::CGraph)
    graphNodes::Array{CNode, 1} = convert(Array{CNode, 1}, [])

    hashmp = Dict{CNode, Int}()

    for node in graph.nodes
        hashmp[node] = length(node.input_nodes)
    end

    qu = Deque{CNode}()

    for node in graph.nodes
        if (hashmp[node] == 0)
            push!(qu, node)
        end
    end

    while !isempty(qu)
        node = popfirst!(qu)
        push!(graphNodes, node)

        for output_node in node.output_nodes
            hashmp[output_node] -= 1
            if (hashmp[output_node] == 0)
                push!(qu, output_node)
            end
        end
    end

    n_onode = 0
    for node in graphNodes
        if (length(node.input_nodes) != node.reqInputs)
            for i in length(node.input_nodes) + 1:node.reqInputs
                newInput = INode()
                newInput.output_nodes = [node]
                push!(node.input_nodes, newInput)
                push!(graph.input_nodes, newInput)
                newInput.Sno = i
            end
        end

        if (length(node.output_nodes) == 0)
            n_onode += 1
            newOutput = ONode()
            newOutput.input_nodes = [node]
            newOutput.out_idx = n_onode
            push!(node.output_nodes, newOutput)
            push!(graph.output_nodes, newOutput)
        end
    end

    for node in graphNodes
        for i = 1:n_onode
            push!(node.gradients, SArray{Tuple{size(graph.output_nodes[i].output)...}}(zeros(size(graph.output_nodes[i].output)...)))
        end
    end

    graph.nodes = graphNodes
end

function computeGraph(graph::CGraph, X::Dict{Tuple{CNode, Int}, SArray})
    for inode in graph.input_nodes
        computeNode(inode, X[(inode.output_nodes[1], inode.Sno)])
    end

    for node in graph.nodes
        computeNode(node)
    end

    for onode in graph.output_nodes
        computeNode(onode)
    end
end

function gradGraph(graph::CGraph)
    for inode in graph.output_nodes
    end
end

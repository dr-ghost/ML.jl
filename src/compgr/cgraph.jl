using StaticArrays

abstract type CNode end

mutable struct Node <: CNode 
    input_nodes::Array{CNode, 1}
    output_nodes::Array{CNode, 1}
    output::SArray
    gradients::Array{SArray, 1}
    isgrad::Bool

    function CNode(input_nodes::Array{CNode, 1} = [], grad::Bool = true)
        obj = new(input_nodes, [])

        for node in obj.input_nodes
            push!(node.output_nodes, obj)
        end

        obj.output = Nothing
        obj.gradients = Nothing
        obj.isgrad = grad

        obj
    end
end

mutable struct INode <: CNode
    output_nodes::Array{CNode, 1}
    output::SArray
    isgrad::Bool
    Sno::Int

    function INode(output::Array{CNode, 1})
        obj = new(output)
        obj.output = @SArray [0.0]
        obj.isgrad = false
        obj.Sno = -1

        obj
    end

    function INode()
        obj = new()
        obj.output_nodes = convert(Array{CNode, 1}, [])
        obj.output = @SArray [0.0]
        obj.isgrad = false
        obj.Sno = -1
        obj
    end
end

mutable struct ONode <: CNode
    input_nodes::Array{CNode, 1}
    output::SArray
    isgrad::Bool
    out_idx

    function ONode(input_nodes::Array{CNode, 1})
        obj = new(input_nodes)
        obj.output = @SArray [0.0]
        obj.isgrad = false

        obj
    end

    function ONode()
        obj = new([])
        obj.output = @SArray [0.0]
        obj.isgrad = false
        obj.out_idx = -1

        obj
    end
end

mutable struct addNode <: CNode
    input_nodes::Array{CNode, 1}
    output_nodes::Array{CNode, 1}
    output::SArray
    gradients::Array{SArray, 1}
    isgrad::Bool
    reqInputs::Int

    function addNode(input_nodes::Array{CNode, 1} = convert(Array{CNode, 1}, []), grad::Bool = true, inputs::Int = 2)
        obj = new(input_nodes)

        obj.output_nodes = convert(Array{CNode, 1}, [])

        for node in obj.input_nodes
            push!(node.output_nodes, obj)
        end

        obj.output = @SArray [0.0]

        obj.gradients = convert(Array{SArray, 1}, [])

        obj.isgrad = grad
        obj.reqInputs = inputs

        obj
    end
end

mutable struct CGraph
    nodes::Array{CNode, 1}
    input_nodes::Array{INode, 1}
    output_nodes::Array{ONode, 1}

    fr_input_nodes::Array{CNode, 1}
    fr_output_nodes::Array{CNode, 1}

    function CGraph(nodes::Array{CNode, 1})
        obj = new(nodes)
        obj.input_nodes = convert(Array{INode, 1}, [])
        obj.output_nodes = convert(Array{ONode, 1}, [])

        obj.fr_input_nodes = convert(Array{CNode, 1}, [])
        obj.fr_output_nodes = convert(Array{CNode, 1}, [])
        makeGraph!(obj)

        obj
    end
end


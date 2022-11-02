__precompile__()

module HyperParameters

    export  HyperParameter, save_hyperparameters

    abstract type HyperParameter end

    function save_hyperparameters(param::HyperParameter, ignore::Array = [])
        error("Not implemented")
    end
end

module Progress
    using ..HyperParameters
    
    export ProgressBoard

    struct ProgressBoard <: HyperParameter
        xLabel::String
        yLabel::String
        xLim::Tuple
        yLim::Tuple
        xscale::String
        yscale::String
        fig
        figsize::Tuple
        axes
        display::Bool
        function ProgressBoard(xLabel="", yLabel = "", xLim = 0.0, yLim = 0.0, xscale = "", yscale = "", fig = nothing, figsize = (3, 2), axes = nothing, display = false)
            tmp = new(xLabel, yLabel, xLim, yLim, xscale, yscale, fig, figsize, axes, display)
            save_hyperparameters(tmp)
            return tmp
        end
    end
end

module Models

    using ..Tensors
    using ..HyperParameters

    export Model, loss, forward, plot, training_step 
    abstract type Model <:HyperParameter end

    function loss(model::Model, ŷ::Tensor, y::Tensor)::Float64
        error("Not implemented")
    end

    function forward(model::Model, x::Tensor)::Tensor
        @assert hasproperty(model, :net) "Neural network not defined"
        return nn(model, x)
    end

    function plot(model::Model, key::Tensor, value::Tensor, b_train)
        @assert hasproperty(model, :trainer) "Model not trainable"
        
        p = plot()

        plot!(p, key, value, label = b_train ? "Train" : "Val", xlabel = model.progress_board.xLabel, ylabel = model.progress_board.yLabel, xlim = model.progress_board.xLim, ylim = model.progress_board.yLim, xscale = model.progress_board.xscale, yscale = model.progress_board.yscale, title = "Loss", legend = :topleft)

    end

    function training_step(model::Model, batch::Tensor)
        l = loss(model, forward(model, batch[:length(batch)-1]), batch[-1])    
        plot(model, :loss, l, true)
        return l
    end

    function validation_step(model::Model, batch::Tensor)
        l = loss(model, forward(model, batch[:length(batch)-1]), batch[-1])    
        plot(model, :loss, l, true)
    end
end

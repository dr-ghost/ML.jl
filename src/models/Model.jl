__precompile__()

module _HyperParameter_

    export  HyperParameter, save_hyperparameters

    abstract type HyperParameter end

    function save_hyperparameters(model::HyperParameter, path::String)
    end
end

module _Progress_
    using .._HyperParameter_
end

module _Model_

    using .._HyperParameter_, .._Loss_, .._nn_, StaticArrays

    export Model, loss, forward, plot, training_step 
    abstract type Model <:HyperParameter end

    function loss(model::Model, ŷ::SArray, y::SArray)::Float64
        if (hasproperty(model, :squared_loss))
            squared_loss(ŷ, y)
        elseif (hasproperty(model, :cross_entropy_loss))
            cross_entropy_loss(ŷ, y)
        else
            error("Not implemented")
        end
    end

    function forward(model::Model, x::SArray)::SArray
        @assert hasproperty(model, :net) "Neural network not defined"
    end

    """
    function plot(model::Model, key::SArray, value::SArray, b_train)
        @assert hasproperty(model, :trainer) "Model not trainable"
        p = plot()

        plot!(p, key, value, label = b_train ? "Train" : "Val", xlabel = model.progress_board.xLabel, ylabel = model.progress_board.yLabel, xlim = model.progress_board.xLim, ylim = model.progress_board.yLim, xscale = model.progress_board.xscale, yscale = model.progress_board.yscale, title = "Loss", legend = :topleft)

    end
    """

    function training_step(model::Model, batch::SArray)
        l = loss(model, forward(model, batch[:length(batch)-1]), batch[-1])    
        #plot(model, :loss, l, true)
        return l
    end

    function validation_step(model::Model, batch::SArray)
        l = loss(model, forward(model, batch[:length(batch)-1]), batch[-1])    
        #plot(model, :loss, l, true)
    end
end

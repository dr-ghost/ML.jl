__precompile__()

module _Optimizer_
    export Optimizer, SDG, Adam

    using .._HyperParameter_

    abstract type Optimizer <: HyperParameter end

    struct SDG <: Optimizer
        lr::Float64
        momentum::Float64
        dampening::Float64
        weight_decay::Float64
        nesterov::Bool
        function SDG(lr::Float64 = 0.01, momentum::Float64 = 0.0, dampening::Float64 = 0.0, weight_decay::Float64 = 0.0, nesterov::Bool = false)
            t = new(lr, momentum, dampening, weight_decay, nesterov)
            save_hyperparameters(t)
            return t
        end
    end

    struct Adam <: Optimizer
        lr::Float64
        betas::Tuple
        eps::Float64
        weight_decay::Float64
        amsgrad::Bool
        function Adam(lr::Float64 = 0.001, betas::Tuple = (0.9, 0.999), eps::Float64 = 1e-8, weight_decay::Float64 = 0.0, amsgrad::Bool = false)
            t = new(lr, betas, eps, weight_decay, amsgrad)
            save_hyperparameters(t)
            return t
        end
    end
end
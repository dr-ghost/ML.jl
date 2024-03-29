using ._Model_, ._HyperParameter_, ._Data_, ._Train_, ._nn_, StaticArrays

struct Linear_Regression <: Model
    num_inputs::Integer
    α::Float64
    σ::Float64
    w::SArray
    b::SArray
    trainer::Trainer
    function Linear_Regression(num_inputs::Integer, α::Float64, σ::Float64 = 0.01, trainer::Trainer = nothing)
        t = new(num_inputs, α, σ, trainer)
        save_hyperparameters(t)
        t.w = normal(w)
        t.b = zeros(b)
        return t
    end
end

function forward(model::Linear_Regression, x::SArray)
    return model.w * x + model.b
end

function loss(model::Linear_Regression, ŷ::SArray, y::SArray)::Float64
    return (ŷ - y)^2 / 2
end

function train(model::Linear_Regression, data::DataModule)
    fit(model, model.trainer, data)
end
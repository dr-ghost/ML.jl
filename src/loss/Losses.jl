__precompile__()

module Losses
    using ..Tensors

    export squared_loss, cross_entropy_loss

    function squared_loss(ŷ::Tensor, y::Tensor)::Float64
        return sum((ŷ - y).^2) / 2
    end

    function cross_entropy_loss(ŷ::Tensor, y::Tensor)::Float64
        return -sum(y .* log(y\hat))
    end

end

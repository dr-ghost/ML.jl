__precompile__()

module _Loss_
    using StaticArrays

    export squared_loss, cross_entropy_loss

    function squared_loss(ŷ::SArray, y::SArray)::Float64
        return sum((ŷ - y).^2) / 2
    end

    function cross_entropy_loss(ŷ::SArray, y::SArray)::Float64
        return -sum(y .* log(y\hat))
    end

end

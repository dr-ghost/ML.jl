__precompile__()

module _Tensor_
    using StaticArrays

    export Tensor
    function Tensor(x::Array)
        return SArray{Tuple{size(x)...}}(x)
    end
end

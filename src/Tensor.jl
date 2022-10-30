__precompile__()

module Tensors

    import Base.@pure

    import Statistics

    using LinearAlgebra
    using StaticArrays



    #exports
    export ⋅, ×, dot, diagm, tr, det, norm, eigvals, eigvecs, eigen, mean
    export AbstractTensor, Tensor
    export normal

    abstract type AbstractTensor{order, dim, T <: Number} <: AbstractArray{T, order} end

    struct Tensor{order, dim, T <: Number, M} <: AbstractTensor{order, dim, T}
        data::NTuple{M, T}
        grad::Bool
        Tensor{order, dim, T, M}(data::NTuple, grad::Bool) where {order, dim, T, M} = new{order, dim, T, M}(data, grad)
    end

    #manipulation
    

    # Utility functions

    get_data(t::AbstractTensor) = t.data

    @pure n_components(::Type{Tensor{order, dim}}) where {order, dim} = dim^order
    @pure get_type(::Type{Type{x}}) where {x} = x
    @pure get_base(::Type{<:Tensor{order, dim}}) where {order, dim} = Tensor{order, dim}

    @pure Base.eltype(::Type{Tensor{order, dim, T, M}}) where {order, dim, T, M} = T
    @pure Base.eltype(::Type{Tensor{order, dim, T}})    where {order, dim, T}    = T
    @pure Base.eltype(::Type{Tensor{order, dim}})       where {order, dim}       = Any

    ##
    Base.IndexStyle(::Type{<:Tensor}) = IndexLinear()

    ##size
    @pure Base.size(::Type{Tensor{order, dim}}) where {order, dim} = (dim, )^order

    # Distribution

    @pure Base.rand(::Type{Tensor{order, dim, T}}) where {order, dim, T} = Tensor{order, dim, T, dim^order}(rand(T, dim^order), false)

    @pure Base.zeros(::Type{Tensor{order, dim, T}}) where {order, dim, T} = Tensor{order, dim, T, dim^order}(zeros(T, dim^order), false)

    @pure normal(::Type{Tensor{order, dim, T}}) where {order, dim, T} = Tensor{order, dim, T, dim^order}(randn(T, dim^order), false)

end
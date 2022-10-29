__precompile__()

module Tensor

import Base.@pure

import Statistics

using LinearAlgebra
using StaticArrays

include("autograd.jl")

#exports
export ⋅, ×, dot, diagm, tr, det, norm, eigvals, eigvecs, eigen, mean
export AbstractTensor, tensor


abstract type AbstractTensor{order, dim, T <: Number} <: AbstractArray{T, order} end

struct tensor{order, dim, T <: Number, M} <: AbstractTensor{order, dim, T}
    data::NTuple{M, T}
    tensor{order, dim, T, M}(data::NTuple) where {order, dim, T, M} = new{order, dim, T, M}(data)
end

# Utility functions

get_data(t::AbstractTensor) = t.data

@pure n_components(::Type{tensor{order, dim}}) where {order, dim} = dim^order
@pure get_type(::Type{Type{x}}) where {x} = x
@pure get_base(::Type{<:tensor{order, dim}}) where {order, dim} = Tensor{order, dim}

@pure Base.eltype(::Type{tensor{order, dim, T, M}}) where {order, dim, T, M} = T
@pure Base.eltype(::Type{tensor{order, dim, T}})    where {order, dim, T}    = T
@pure Base.eltype(::Type{tensor{order, dim}})       where {order, dim}       = Any

##
Base.IndexStyle(::Type{<:tensor}) = IndexLinear()

##size
@pure Base.size(::Type{tensor{order, dim}}) where {order, dim} = (dim, )^order

end
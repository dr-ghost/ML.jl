__precompile__()

module Tensor

import Base.@pure

import Statistics
using Statistics : mean
using LinearAlgebra
using StaticArrays

include("autograd.jl")

#exports
export ⋅, ×, dot, diagm, tr, det, norm, eigvals, eigvecs, eigen, mean
export AbstractTensor


abstract type AbstractTensor{order, dim, T <: Number} <: AbstractArray{T, order} end

struct Tensor{order, dim, T <: Number, M} <: AbstractTensor{order, dim, T}
    data::NTuple{M, T}
    Tensor{order, dim, T, M}(data::NTuple) where {order, dim, T, M} = new{order, dim, T, M}(data)
end

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
Base.size(t::Tensor{order, dim}) where {order, dim} = ntuple(i -> dim, order)
Base.length(::Type{Tensor{order, dim, T, M}}) where {order, dim, T, M} = 

@inline          Tensor{order, dim, T}(data::Union{AbstractArray, Tuple, Function}) where {order, dim, T} = convert(Tensor{order, dim, T}, Tensor{order, dim}(data))
@inline          Tensor{order, dim, T, M}(data::Union{AbstractArray, Tuple, Function})  where {order, dim, T, M} = Tensor{order, dim, T}(data)
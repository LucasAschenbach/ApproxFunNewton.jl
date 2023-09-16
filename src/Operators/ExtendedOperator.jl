export ExtendedOperator

import Base: convert, getindex
using LinearAlgebra
using BandedMatrices
using LowRankMatrices
import LowRankMatrices: rank
using Infinities
using ApproxFun
import ApproxFunBase: Operator, bandwidths, isbanded, israggedbelow, isbandedbelow, colstop,
                      Space, ArraySpace, ConstantSpace, domainspace, rangespace, components, domain, dimension,
                      CachedOperator, AlmostBandedMatrix, cache, resizedata!

# Appends compactly supported columns to the left of the operator
# ┌────────┬────────── ⋯
# │  cols  │    op    
# │        │        
struct ExtendedOperator{T,DS,RS,BI} <: Operator{T}
    cols::Array{Vector{T}}
    op::Operator{T}
    domainspace::DS
    rangespace::RS
    bandwidths::BI
    colsbandwidth::Int
    israggedbelow::Bool

    function ExtendedOperator(cols::Array{Vector{T}},op::Operator{T},domainspace::DS,rangespace::RS) where {T,DS,RS}
        cbw = max([length(col)-i for (i,col) in enumerate(cols)]..., 0)
        if isbanded(op)
            l,u = bandwidths(op)
            bw = (max(cbw, l-length(cols)), u+length(cols))
        else
            l,u = dimension(domainspace)-1, dimension(rangespace)-1
            bw = (l,u+length(cols))
        end

        new{T,DS,RS,typeof(bw)}(cols,op,domainspace,rangespace,bw,cbw,israggedbelow(op))
    end
end

function ExtendedOperator(cols::Array{Vector{T}},op::Operator{T}) where {T}
    ds = ArraySpace(Base.typed_vcat(Space,fill(ConstantSpace(T),length(cols)), collect(components(domainspace(op)))))
    ExtendedOperator(cols,op,ds,rangespace(op))
end

ExtendedOperator(col::Vector{T},op::Operator{T}) where {T} = ExtendedOperator([col],op)

function convert(::Type{Operator{T}}, S::ExtendedOperator) where {T}
    if T == eltype(S)
        S
    else
        ExtendedOperator(map(x -> convert(Vector{T},c),S.cols), convert(Operator{T},S.op))
    end
end

domainspace(A::ExtendedOperator) = A.domainspace
rangespace(A::ExtendedOperator) = A.rangespace
domain(A::ExtendedOperator) = domain(domainspace(A))
bandwidths(A::ExtendedOperator) = A.bandwidths
isbanded(A::ExtendedOperator) = isbanded(A.op)
Base.size(A::ExtendedOperator) = (dimension(domainspace(A)),dimension(rangespace(A)))

function getindex(A::ExtendedOperator{T},i::Integer,j::Integer) where {T}
    ncols = length(A.cols)
    if j ≤ ncols
        if i ≤ length(A.cols[j])
            return A.cols[j][i]
        else
            return zero(T)
        end
    else
        return getindex(A.op,i,j-ncols)
    end
end

function colstop(A::ExtendedOperator, j::Integer)
    if isbandedbelow(A)
        min(j+bandwidth(A,1)::Int,size(A,1))::Int
    else # assume is raggedbelow
        ncols = length(A.cols)
        if j ≤ ncols
            length(A.cols[j])::Int
        else
            ApproxFunBase.colstop(A.op,j-ncols)::Int
        end
    end
end

function CachedOperator(extop::ExtendedOperator{T};padding::Bool = false) where {T}
    ds = domainspace(extop)
    rs = rangespace(extop)
    ncols = length(extop.cols)
    cbw = extop.colsbandwidth

    C = cache(extop.op,padding=padding) # assume C.data is AlmostBandedMatrix

    m,n = C.datasize
    sz = (m,n+ncols)
    lop,uop = bandwidths(C)
    lextop = max(cbw,lop-ncols)

    data = C.data
    lband,uband = bandwidths(data)
    lbandextop,ubandextop = (max(cbw,lband-ncols),uband+ncols)
    r = LowRankMatrices.rank(data.fill)

    ret = AlmostBandedMatrix(Zeros{T}(sz),(lbandextop,ubandextop),r)

    # fill the fill matrix
    ret.fill.V[1:n+ncols,1:r] = Matrix(view(extop,1:r,1:n+ncols))'
    ret.fill.U[1:r,1:r] = Matrix{T}(I,r,r)

    for k = 1:m
        jran = rowrange(ret.bands, k)
        ret[k,jran] = extop[k,jran]
    end

    CachedOperator(extop,ret,sz,ds,rs,(lextop,∞))
end

# Grow cached extended operator

function resizedata!(co::CachedOperator{T,<:AlmostBandedMatrix{T},<:ExtendedOperator{T}},
    n::Integer,::Colon) where {T<:Number}
    if n ≤ co.datasize[1]
        return co
    end

    l,u = bandwidths(co.data.bands)
    m = max(co.datasize[2],n+u)
    sz = (n,m)
    r = rank(co.data.fill)
    co.data = pad(co.data,sz...)

    if m > co.datasize[2]
        co.data.fill.V[co.datasize[2]+1:m,1:r] = Matrix(view(co.op,1:r,co.datasize[2]+1:m))'
    end

    iran = co.datasize[1]+1:n
    jran = max(1,iran[1]-l):m
    axpy!(1.0,view(co.op,iran,jran),
              view(co.data.bands,iran,jran))

    co.datasize=sz
    co
end

resizedata!(co::CachedOperator{T,<:AlmostBandedMatrix{T},<:ExtendedOperator{T}},
    n::Integer,m::Integer) where {T<:Number} = resizedata!(co,max(n,m+bandwidth(co.data.bands,1)),:)

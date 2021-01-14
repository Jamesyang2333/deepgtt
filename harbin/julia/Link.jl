mutable struct Link
    gid::Int64
    I::Float32
    O::Float32
    S::Float32
    C::Float32
end

Link(gid) = Link(gid, zero(Float32), zero(Float32), zero(Float32), zero(Float32))
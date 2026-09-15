#my_geometry.jl
# SVector2
# SVector3
# unit
# rot2
# signed_angle
# wrap_angle
# displacement
# periodic_displacement
# centre_of_mass_periodic

using LinearAlgebra


struct SVector2
    x::Float64
    y::Float64
end

Base.getindex(v::SVector2, i::Int) = (
    i == 1 ? v.x :
    i == 2 ? v.y :
    throw(BoundsError(v, i))
)

Base.:-(v::SVector2) = SVector2(-v.x, -v.y)
Base.:+(a::SVector2, b::SVector2) = SVector2(a.x + b.x, a.y + b.y)
Base.:-(a::SVector2, b::SVector2) = SVector2(a.x - b.x, a.y - b.y)
Base.:*(α::Real, v::SVector2) = SVector2(α * v.x, α * v.y)
Base.:*(v::SVector2, α::Real) = α * v
Base.:/(v::SVector2, α::Real) = SVector2(v.x / α, v.y / α)

LinearAlgebra.dot(a::SVector2, b::SVector2) = a.x * b.x + a.y * b.y
LinearAlgebra.norm(v::SVector2) = sqrt(v.x * v.x + v.y * v.y)

unit(v::SVector2; ϵ = 1e-12) = (norm(v) < ϵ ? SVector2(0.0, 0.0) : v / norm(v))

struct SVector3
    x::Float64
    y::Float64
    z::Float64
end

Base.getindex(v::SVector3, i::Int) = (
    i == 1 ? v.x :
    i == 2 ? v.y :
    i == 3 ? v.z :
    throw(BoundsError(v, i))
)

Base.:-(v::SVector3) = SVector3(-v.x, -v.y, -v.z)
Base.:+(a::SVector3, b::SVector3) = SVector3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::SVector3, b::SVector3) = SVector3(a.x - b.x, a.y - b.y, a.z - b.z)
Base.:*(α::Real, v::SVector3) = SVector3(α * v.x, α * v.y, α * v.z)
Base.:*(v::SVector3, α::Real) = α * v
Base.:/(v::SVector3, α::Real) = SVector3(v.x / α, v.y / α, v.z / α)

LinearAlgebra.dot(a::SVector3, b::SVector3) = a.x * b.x + a.y * b.y + a.z * b.z
LinearAlgebra.norm(v::SVector3) = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

unit(v::SVector3; ϵ = 1e-12) = (norm(v) < ϵ ? SVector3(0.0, 0.0, 0.0) : v / norm(v))


@inline function displacement(p::V, q::V, L::Real) where {V<:Union{SVector2,SVector3}}
    d = q - p
    return V(ntuple(k -> d[k] - L * round(d[k] / L), fieldcount(V))...)
end


function cross(a::SVector3, b::SVector3)
    return SVector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )
end

@inline angle_of(v::SVector2) = atan(v.y, v.x)

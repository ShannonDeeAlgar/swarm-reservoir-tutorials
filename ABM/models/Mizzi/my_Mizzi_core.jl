# ============================================================
# Mizzi model: parameters, homes and Voronoi adjacency
# Mizzi et al., "Reservoir Computing with Territorial Agents", Sec. 2
# ============================================================

using Random
using LinearAlgebra
using StaticArrays

"""
    MizziParams(homes; kh=80, kd=60, kf=20, plot_padding=1)

Parameters for the two-dimensional territorial-agent force model. `homes`
contains one unique fixed home location per agent. Voronoi-cell adjacency is
computed once from the dual Delaunay triangulation and stored in `neighbours`.

The defaults `(kh, kd, kf) = (80, 60, 20)` and `dt=0.02` used by callers are
the values stated in Sec. 2 of the paper. Territory optimisation/MDL is not
part of this implementation.
"""
struct MizziParams
    N::Int
    kh::Float64
    kd::Float64
    kf::Float64
    homes::Vector{SVector2}
    neighbours::Vector{Vector{Int}}
    plot_padding::Float64
    plot_extent::Float64
end

function mizzi_voronoi_neighbours(homes::Vector{SVector2})
    N = length(homes)
    N >= 1 || error("At least one territorial home is required.")
    length(unique(homes)) == N || error("Territorial home locations must be unique.")

    neighbours = [Int[] for _ in 1:N]
    N == 1 && return neighbours
    if N == 2
        push!(neighbours[1], 2)
        push!(neighbours[2], 1)
        return neighbours
    end

    # Cells i and j share a boundary when some nonzero interval of their
    # perpendicular bisector is no farther from i/j than from every other
    # home. Intersect those one-dimensional half-plane constraints directly.
    # This avoids compiling a full triangulation package for a small fixed set.
    tol = 1e-10
    for i in 1:N-1, j in i+1:N
        hi, hj = homes[i], homes[j]
        midpoint = (hi + hj) / 2
        edge = hj - hi
        tangent = SVector2(-edge.y, edge.x)
        lower, upper = -Inf, Inf
        feasible = true
        for k in 1:N
            (k == i || k == j) && continue
            hk = homes[k]
            delta = hk - hi
            a = 2 * dot(tangent, delta)
            b = dot(hk, hk) - dot(hi, hi) - 2 * dot(midpoint, delta)
            if abs(a) <= tol
                if b < -tol
                    feasible = false
                    break
                end
            elseif a > 0
                upper = min(upper, b / a)
            else
                lower = max(lower, b / a)
            end
        end
        if feasible && lower < upper - tol
            push!(neighbours[i], j)
            push!(neighbours[j], i)
        end
    end
    foreach(sort!, neighbours)
    return neighbours
end

function MizziParams(homes::AbstractVector;
    kh::Real = 80.0,
    kd::Real = 60.0,
    kf::Real = 20.0,
    plot_padding::Real = 1.0,
)
    H = SVector2[SVector2(Float64(h[1]), Float64(h[2])) for h in homes]
    kh >= 0 || error("kh must be non-negative.")
    kd >= 0 || error("kd must be non-negative.")
    kf >= 0 || error("kf must be non-negative.")
    plot_padding >= 0 || error("plot_padding must be non-negative.")

    neighbours = mizzi_voronoi_neighbours(H)
    extent = maximum(max(abs(h.x), abs(h.y)) for h in H) + Float64(plot_padding)
    return MizziParams(length(H), Float64(kh), Float64(kd), Float64(kf),
        H, neighbours, Float64(plot_padding), extent)
end

function update_MizziParams(P::MizziParams;
    kh::Real=P.kh, kd::Real=P.kd, kf::Real=P.kf,
    homes::AbstractVector=P.homes, plot_padding::Real=P.plot_padding,
)
    return MizziParams(homes; kh=kh, kd=kd, kf=kf,
        plot_padding=max(Float64(plot_padding), 0.0))
end

"Sample distinct home locations from columns of a two-dimensional input."
function mizzi_homes_from_input(U::AbstractMatrix, N::Int;
    rng::AbstractRNG = Random.default_rng(),
)
    size(U, 1) == 2 || error("Territorial homes require a 2×T embedded input; got size $(size(U)).")
    N >= 1 || error("N must be positive.")
    N <= size(U, 2) || error("Cannot sample $N homes from only $(size(U,2)) input points.")
    all(isfinite, U) || error("Input used for home sampling contains NaN or Inf.")

    selected = SVector2[]
    for j in randperm(rng, size(U, 2))
        h = SVector2(Float64(U[1, j]), Float64(U[2, j]))
        h in selected || push!(selected, h)
        length(selected) == N && break
    end
    length(selected) == N || error("Input contains fewer than $N distinct embedded points.")
    return selected
end

"Sample home locations uniformly from a square; useful as a control."
function mizzi_homes_uniform(N::Int;
    halfwidth::Real = 2.5,
    rng::AbstractRNG = Random.default_rng(),
)
    N >= 1 || error("N must be positive.")
    halfwidth > 0 || error("halfwidth must be positive.")
    return [SVector2((2rand(rng) - 1) * halfwidth,
                     (2rand(rng) - 1) * halfwidth) for _ in 1:N]
end

function init_state_Mizzi_2d(P::MizziParams;
    rng::AbstractRNG = Random.default_rng(),
    position_noise::Real = 0.0,
    velocity_noise::Real = 0.0,
)
    position_noise >= 0 || error("position_noise must be non-negative.")
    velocity_noise >= 0 || error("velocity_noise must be non-negative.")
    pos = [h + SVector2(position_noise * randn(rng), position_noise * randn(rng))
           for h in P.homes]
    vel = [SVector2(velocity_noise * randn(rng), velocity_noise * randn(rng))
           for _ in 1:P.N]
    desired = fill(SVector2(0.0, 0.0), P.N)
    return SwarmState{SVector2}(pos, vel, desired)
end

"Return the index of the Voronoi cell containing point `p`."
function mizzi_territory_owner(P::MizziParams, p::SVector2)
    return argmin(eachindex(P.homes)) do i
        d = p - P.homes[i]
        d.x * d.x + d.y * d.y
    end
end

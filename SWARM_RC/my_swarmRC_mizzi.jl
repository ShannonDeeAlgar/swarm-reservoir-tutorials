# ============================================================
# Mizzi territorial-agent reservoir wrapper
# Basic force model from Mizzi et al., without consistency capacity or MDL
# ============================================================

using Random
using LinearAlgebra

"The paper's prey input: a 2D point gated by Voronoi-cell adjacency."
struct MizziPreyCoupling <: InputCoupling end

"""
    MizziReservoir <: AbstractReservoir

A two-dimensional Mizzi reservoir of territorial agents. Every agent has a unique fixed
home. The reservoir input is the prey location and the readout features are
the paper's raw `4N` state: home-relative x positions, home-relative y
positions, x velocities, then y velocities. No Gaussian observation layer is
needed or fitted.
"""
Base.@kwdef mutable struct MizziReservoir <: AbstractReservoir
    rng::AbstractRNG
    state::SwarmState{SVector2}
    P::MizziParams
    dt::Float64 = 0.02
    coupling::InputCoupling = MizziPreyCoupling()
    position_noise::Float64 = 0.0
    velocity_noise::Float64 = 0.0
end

function build_Mizzi_reservoir(P::MizziParams, dt::Float64 = 0.02;
    coupling::InputCoupling = MizziPreyCoupling(),
    position_noise::Float64 = 0.0,
    velocity_noise::Float64 = 0.0,
    rng::AbstractRNG = Random.default_rng(),
)
    coupling isa MizziPreyCoupling || coupling isa NoCoupling ||
        error("MizziReservoir supports MizziPreyCoupling or NoCoupling.")
    state = init_state_Mizzi_2d(P;
        rng=rng, position_noise=position_noise, velocity_noise=velocity_noise)
    return MizziReservoir(rng=rng, state=state, P=P, dt=dt,
        coupling=coupling, position_noise=position_noise,
        velocity_noise=velocity_noise)
end

"Build a Mizzi reservoir with homes sampled from an embedded 2×T input."
function build_Mizzi_reservoir(U::AbstractMatrix, N::Int, dt::Float64 = 0.02;
    rng::AbstractRNG = Random.default_rng(), params_kwargs...,
)
    homes = mizzi_homes_from_input(U, N; rng=rng)
    P = MizziParams(homes; params_kwargs...)
    return build_Mizzi_reservoir(P, dt; rng=rng)
end

function reset!(res::MizziReservoir;
    rng::AbstractRNG = Random.default_rng(),
)
    res.rng = rng
    res.state = init_state_Mizzi_2d(res.P; rng=rng,
        position_noise=res.position_noise, velocity_noise=res.velocity_noise)
    return res
end

function reservoir_step!(res::MizziReservoir, u::AbstractVector;
    rng::AbstractRNG = Random.default_rng(),
)
    length(u) == 2 || error("MizziReservoir expects prey input (u(t-τ), u(t)); got length $(length(u)).")
    all(isfinite, u) || error("Mizzi prey input contains NaN or Inf.")
    prey = res.coupling isa MizziPreyCoupling ?
        SVector2(Float64(u[1]), Float64(u[2])) : nothing
    Mizzi_step_2d!(res.state, res.P, res.dt; prey=prey, rng=rng)
    return res
end

function raw_state(res::MizziReservoir)
    N = res.P.N
    Pmat = Matrix{Float64}(undef, 2, N)
    Vmat = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        Pmat[1, i], Pmat[2, i] = res.state.pos[i].x, res.state.pos[i].y
        Vmat[1, i], Vmat[2, i] = res.state.vel[i].x, res.state.vel[i].y
    end
    return (P=Pmat, V=Vmat)
end

"Paper state S(t): home-relative positions followed by velocities (4N)."
function feature_map(res::MizziReservoir)
    N = res.P.N
    px = Vector{Float64}(undef, N)
    py = similar(px)
    vx = similar(px)
    vy = similar(px)
    @inbounds for i in 1:N
        px[i] = res.state.pos[i].x - res.P.homes[i].x
        py[i] = res.state.pos[i].y - res.P.homes[i].y
        vx[i] = res.state.vel[i].x
        vy[i] = res.state.vel[i].y
    end
    return vcat(px, py, vx, vy)
end

clone_reservoir(res::MizziReservoir) = MizziReservoir(
    rng=res.rng, state=copy_state(res.state), P=res.P, dt=res.dt,
    coupling=res.coupling, position_noise=res.position_noise,
    velocity_noise=res.velocity_noise)

state_vector(res::MizziReservoir, obs=nothing) = feature_map(res)

function perturb_state!(res::MizziReservoir, ξ::AbstractVector)
    N = res.P.N
    length(ξ) == 4N || error("perturb_state!: expected length $(4N), got $(length(ξ)).")
    @inbounds for i in 1:N
        p, v = res.state.pos[i], res.state.vel[i]
        res.state.pos[i] = SVector2(p.x + ξ[i], p.y + ξ[N+i])
        res.state.vel[i] = SVector2(v.x + ξ[2N+i], v.y + ξ[3N+i])
    end
    return res
end

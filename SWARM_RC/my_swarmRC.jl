# ============================================================
# my_swarmRC.jl
# Swarm-reservoir integration: wires the Couzin swarm to the generic
# reservoir-computing interface.
# ============================================================
#
# Load order (see SWARM_RC/load_SwarmRC.jl):
#   ABM/load_ABM.jl                    -> SwarmState, CouzinParams machinery
#   ABM/models/Couzin/load_Couzin.jl   -> Couzin rules + nondimensionalisation
#   RC/my_reservoir_core.jl            -> AbstractReservoir, train_and_evaluate_reservoir
#   SWARM_RC/my_swarmRC.jl             -> this file
#
# This file assumes those are already loaded and provide:
#   SwarmState, CouzinParams, desired_direction_2d, update_velocity_2d,
#   update_position_2d, init_state_Couzin_2d, displacement, unit, SVector2,
#   periodic_disp, AbstractReservoir, feature_map, reset!, reservoir_step!,
#   raw_state, build_observation_layer!
#
# Minimal end-to-end example is at the bottom of this file.

using Random, LinearAlgebra, Statistics
using ProgressMeter

# ============================================================
# 1) Input coupling — the swappable design point for T1-S2/H3/H4
# ============================================================
#
# DP Eq. 3: x_i(n+1) = g_i(x_i(n), N_i(n), u(n))
# The external input u(n) perturbs each agent's *desired direction* rather
# than entering through a fixed linear mapping. Every coupling mechanism is
# one implementation of the same interface:
#
#     couple_direction(coupling, i, pos, vel, P, u, base_dir) -> SVector2
#
# `base_dir` is agent i's desired direction from the unperturbed Couzin rule
# (`desired_direction_2d`); `u` is the current global input (a 2-vector). Return
# the direction agent i should actually turn towards this step.
#
# To compare a new coupling mechanism (Project T1-S2: repulsive/attractive
# driver, pursuit, spatial forcing, nonlinear injection, global scalar
# forcing, ...), define a new struct and one `couple_direction` method —
# nothing else in the pipeline changes.

abstract type InputCoupling end

"No input coupling: the swarm evolves under its own dynamics only."
struct NoCoupling <: InputCoupling end
couple_direction(::NoCoupling, i, pos, vel, P, u, base_dir) = base_dir

"""
    TemperatureSpeedCoupling(; base_speed, gain=0.30,
                               min_speed=0.1base_speed,
                               max_speed=5base_speed)

Global scalar input coupling adapted from Lund et al. (2026). The input is
interpreted as a temperature-like signal and mapped to a target speed

    s_target(n) = clamp(base_speed * (1 + gain * u(n)), min_speed, max_speed).

This type describes the input-to-actuator mapping, not a particular swarm.
Each compatible reservoir decides how its own dynamics relax toward the target
speed. At present LymburnReservoir supports it; MizziReservoir deliberately
does not because global speed control conflicts with territorial damping and
settling at fixed homes.
"""
struct TemperatureSpeedCoupling <: InputCoupling
    base_speed::Float64
    gain::Float64
    min_speed::Float64
    max_speed::Float64
end

function TemperatureSpeedCoupling(; base_speed::Real, gain::Real = 0.30,
    min_speed::Real = 0.1 * base_speed, max_speed::Real = 5.0 * base_speed)
    base_speed > 0 || throw(ArgumentError("base_speed must be positive."))
    min_speed > 0 || throw(ArgumentError("min_speed must be positive."))
    max_speed >= min_speed || throw(ArgumentError("max_speed must be at least min_speed."))
    return TemperatureSpeedCoupling(float(base_speed), float(gain),
        float(min_speed), float(max_speed))
end

target_speed(c::TemperatureSpeedCoupling, u::Real) =
    clamp(c.base_speed * (1 + c.gain * float(u)), c.min_speed, c.max_speed)

"""
    PredatorCoupling(; rp, alpha=1.0, flee=true)

Predator-driven coupling (Lymburn et al. 2021): the external input `u(n)` is
treated as a 2D predator position in the swarm's domain. Agents within `rp`
of the predator blend their base direction with a flee/chase direction,
weighted by `alpha` (0 = ignore predator, 1 = flee/chase only); agents
outside `rp` are unaffected.

`rp` is in the same units as `P.L`/`P.Zr` — if `P` is nondimensionalised
(see `nondimensionalise!` in ABM/models/Couzin/my_Couzin_scaling.jl), `rp`
should be expressed in units of `P.Zr` too, so the input's spatial scale is
comparable to the swarm's own interaction range (DP Phase 0A).
"""
Base.@kwdef struct PredatorCoupling <: InputCoupling
    rp::Float64
    alpha::Float64 = 1.0
    flee::Bool = true
end

function couple_direction(c::PredatorCoupling, i, pos, vel, P, u, base_dir)
    dP = displacement(pos[i], SVector2(u[1], u[2]), P.L)
    rP = norm(dP)
    (rP == 0.0 || rP > c.rp) && return base_dir
    u_pred = c.flee ? -unit(dP) : unit(dP)
    return unit((1 - c.alpha) * base_dir + c.alpha * u_pred)
end

# ============================================================
# 2) Coupled Couzin step
# ============================================================
# Identical to `Couzin_step_2d!` (ABM/models/Couzin/my_Couzin_Rules2d.jl) except
# each agent's desired direction is passed through `couple_direction` before
# the turning-rate/noise update. With `coupling = NoCoupling()` this produces
# exactly the same trajectory as `Couzin_step_2d!`.

function Couzin_step_with_input!(
    state::SwarmState{SVector2}, P::CouzinParams, dt::Float64,
    coupling::InputCoupling, u::AbstractVector;
    rng::AbstractRNG = Random.default_rng(),
)
    pos = state.pos
    vel = state.vel
    desired = state.desired
    N = length(pos)

    @inbounds for i in 1:N
        base = desired_direction_2d(i, pos, vel, P)
        desired[i] = couple_direction(coupling, i, pos, vel, P, u, base)
    end

    @inbounds for i in 1:N
        vel[i] = update_velocity_2d(vel[i], desired[i], P, dt, rng)
    end

    @inbounds for i in 1:N
        pos[i] = update_position_2d(pos[i], vel[i], P, dt)
    end

    return nothing
end

# ============================================================
# 3) CouzinReservoir <: AbstractReservoir
# ============================================================

"""
    KernelLayer(C, inv2w)

Gaussian kernel observation layer: `C` is 2×M kernel centres, `inv2w[m]` is
1/(2σ_m²) for kernel m. Built by `build_observation_layer!` /
`build_observation_layer_coverage_kmeans!`, consumed by `feature_map`.
"""
Base.@kwdef struct KernelLayer
    C::Matrix{Float64}
    inv2w::Vector{Float64}
end

"""
    CouzinReservoir <: AbstractReservoir

A Couzin swarm driven by an external input `u(n)` through `coupling`. Build
with `build_Couzin_reservoir`; conforms to the `AbstractReservoir` interface
in RC/my_reservoir_core.jl (`reset!`, `reservoir_step!`, `feature_map`,
`raw_state`, `build_observation_layer!`), so `train_and_evaluate_reservoir`
works on it directly — including the memory/separability/stability
diagnostics already implemented there (Project T1-S1).
"""
Base.@kwdef mutable struct CouzinReservoir <: AbstractReservoir
    rng::AbstractRNG
    state::SwarmState{SVector2}
    P::CouzinParams
    dt::Float64
    coupling::InputCoupling = NoCoupling()
    init_mode::Symbol = :clustered
end

"""
    build_Couzin_reservoir(P, dt; coupling=NoCoupling(), init_mode=:clustered, rng)

Build a swarm-reservoir from Couzin parameters `P` and step size `dt`.

`P` should typically come from
`Couzin_params_from_preset(preset; nondim=true, ...)`
(ABM/models/Couzin/my_Couzin_scaling.jl) rather than raw physical units:
nondimensionalising the swarm *before* coupling in an external input is what
keeps the input's spatial/temporal scale comparable to the swarm's intrinsic
response scale (DP Phase 0A, q5) — mismatched scales are a common reason a
coupling mechanism looks like it "doesn't work" when the real problem is unit
mismatch.
"""
function build_Couzin_reservoir(P::CouzinParams, dt::Float64;
    coupling::InputCoupling = NoCoupling(),
    init_mode::Symbol = :clustered,
    rng::AbstractRNG = Random.default_rng(),
)
    state = init_state_Couzin_2d(P; rng=rng, init_mode=init_mode)
    return CouzinReservoir(rng=rng, state=state, P=P, dt=dt, coupling=coupling, init_mode=init_mode)
end

# ------------------------------------------------------------
# AbstractReservoir interface
# ------------------------------------------------------------

function reset!(res::CouzinReservoir; rng::AbstractRNG=Random.default_rng())
    res.rng = rng
    res.state = init_state_Couzin_2d(res.P; rng=rng, init_mode=res.init_mode)
    return res
end

function reservoir_step!(res::CouzinReservoir, u::AbstractVector; rng::AbstractRNG=Random.default_rng())
    @assert length(u) == 2 "CouzinReservoir expects a 2D input u = (u1, u2); got length $(length(u))"
    Couzin_step_with_input!(res.state, res.P, res.dt, res.coupling, u; rng=rng)
    return res
end

function raw_state(res::CouzinReservoir)
    N = length(res.state.pos)
    Pmat = Matrix{Float64}(undef, 2, N)
    Vmat = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        Pmat[1, i] = res.state.pos[i].x
        Pmat[2, i] = res.state.pos[i].y
        Vmat[1, i] = res.state.vel[i].x
        Vmat[2, i] = res.state.vel[i].y
    end
    return (P=Pmat, V=Vmat)
end

# ------------------------------------------------------------
# Optional reservoir-analysis interface (needed by perturbation_stability)
# ------------------------------------------------------------
#
# Note: for CouzinReservoir, `state_vector` deliberately returns the raw
# dynamical state (agent positions + velocities), NOT `feature_map`'s output.
# BasicESN can use its feature vector for both because there the reservoir
# state *is* the feature vector; a swarm-reservoir's Gaussian-kernel features
# are a many-to-one projection of the state, so perturbing/comparing in
# feature space would not correspond to perturbing/comparing the underlying
# dynamics. `perturb_state!` and `state_vector` must agree on the same
# representation (see how `perturbation_stability` in RC/my_reservoir_core.jl
# uses them together); `feature_map` remains the separate, observation-layer
# representation used for training and for `sequence_separability`.

clone_reservoir(res::CouzinReservoir) = CouzinReservoir(
    rng = res.rng,
    state = copy_state(res.state),
    P = res.P,
    dt = res.dt,
    coupling = res.coupling,
    init_mode = res.init_mode,
)

function state_vector(res::CouzinReservoir, obs=nothing)
    rs = raw_state(res)
    return vcat(vec(rs.P), vec(rs.V))
end

"""
    perturb_state!(res::CouzinReservoir, ξ)

Add a perturbation `ξ` (length 4N: 2N position components then 2N velocity
components, matching `state_vector`'s layout) directly to agent positions and
velocities. Used by `perturbation_stability` to test whether the swarm's own
dynamics contract or amplify a small perturbation over time -- the physical-
reservoir analogue of the echo state property.
"""
function perturb_state!(res::CouzinReservoir, ξ::AbstractVector)
    N = length(res.state.pos)
    @assert length(ξ) == 4N "perturb_state!: expected length 4N=$(4N), got $(length(ξ))"
    @inbounds for i in 1:N
        p = res.state.pos[i]
        res.state.pos[i] = SVector2(p.x + ξ[2i - 1], p.y + ξ[2i])
    end
    @inbounds for i in 1:N
        v = res.state.vel[i]
        res.state.vel[i] = SVector2(v.x + ξ[2N + 2i - 1], v.y + ξ[2N + 2i])
    end
    return res
end

"""
    feature_map(res::CouzinReservoir, K::KernelLayer)

3M features per timestep: (Gaussian-kernel density, vx-weighted density,
vy-weighted density) at each of the M kernel centres in `K`. This is the
Gaussian-kernel observation map from Lymburn et al. (2021); Project T1-H1
compares this against alternative observation maps by swapping the `obs`
argument threaded through by RC/my_reservoir_core.jl, not by editing this
function.
"""
function feature_map(res::CouzinReservoir, K::KernelLayer)
    C = K.C
    inv2w = K.inv2w
    M = size(C, 2)
    L = res.P.L

    r1 = zeros(Float64, M)
    r2 = zeros(Float64, M)
    r3 = zeros(Float64, M)

    @inbounds for i in eachindex(res.state.pos)
        ax, ay = res.state.pos[i].x, res.state.pos[i].y
        vx, vy = res.state.vel[i].x, res.state.vel[i].y
        for m in 1:M
            dx, dy = periodic_disp(C[1, m], C[2, m], ax, ay, L)
            ψ = exp(-(dx * dx + dy * dy) * inv2w[m])
            r1[m] += ψ
            r2[m] += ψ * vx
            r3[m] += ψ * vy
        end
    end

    return vcat(r1, r2, r3)
end

# ------------------------------------------------------------
# Observation-layer construction
# ------------------------------------------------------------

"""
    build_observation_layer!(res::CouzinReservoir, Utrain; washout=0, rng, method=:spatial_gaussian, kwargs...)

Required override of the generic `build_observation_layer!` hook (see
RC/my_reservoir_core.jl); called once by `train_and_evaluate_reservoir` before
training, with the result threaded through as `obs` to every `feature_map`
call. `method = :spatial_gaussian` (default) or `:coverage_kmeans`;
`:lymburn` remains a backward-compatible alias.
"""
function build_observation_layer!(res::CouzinReservoir, Utrain::AbstractMatrix;
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    method::Symbol = :spatial_gaussian,
    obs_kwargs...
)
    if method in (:spatial_gaussian, :lymburn)
        return build_observation_layer_spatial_gaussian!(res, Utrain; washout=washout, rng=rng, obs_kwargs...)
    elseif method == :coverage_kmeans
        return build_observation_layer_coverage_kmeans!(res, Utrain; washout=washout, rng=rng, obs_kwargs...)
    else
        error("Unknown observation-layer method = $method. Use :spatial_gaussian or :coverage_kmeans.")
    end
end

"""
    build_observation_layer_spatial_gaussian!(res, Utrain; washout, rng, M=200, kneigh=5)

Lymburn-style kernel placement: for each of `M` kernels, pick a random
post-washout time and a random agent; the kernel centre is that agent's
position, and its width is set from the distance to its `kneigh`-th nearest
neighbour at that time (drives the reservoir once, teacher-forced by
`Utrain`, to harvest these samples).

`inv2w[m] = 1/(2w)` where `w` is that raw kNN distance -- confirmed
against the original authors' MATLAB (`RBF_state.m`: `exp(-d^2/(2*width))`
with `width` the kNN distance directly, i.e. used as a variance-like term,
not squared into one).
"""
function build_observation_layer_spatial_gaussian!(res::CouzinReservoir, Utrain::AbstractMatrix;
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    M::Int = 200,
    kneigh::Int = 5,
    show_progress::Bool = true,
)
    reset!(res; rng=rng)

    Nu, T = size(Utrain)
    @assert T > washout + 1 "washout too large for training length"

    N = res.P.N
    L = res.P.L
    @assert kneigh < N "kneigh must be < number of agents"

    t_samp = rand(rng, (washout + 1):T, M)
    i_samp = rand(rng, 1:N, M)

    by_t = Dict{Int, Vector{Int}}()
    for m in 1:M
        push!(get!(by_t, t_samp[m], Int[]), m)
    end

    C = zeros(2, M)
    inv2w = zeros(Float64, M)

    function kth_neighbour_dist(i::Int)
        xi, yi = res.state.pos[i].x, res.state.pos[i].y
        d = Vector{Float64}(undef, N - 1)
        idx = 1
        @inbounds for j in 1:N
            j == i && continue
            dx, dy = periodic_disp(xi, yi, res.state.pos[j].x, res.state.pos[j].y, L)
            d[idx] = hypot(dx, dy)
            idx += 1
        end
        sort!(d)
        return d[kneigh]
    end

    prog = show_progress ? Progress(T; desc="fit spatial Gaussian kernels", showspeed=true) : nothing

    for t in 1:T
        reservoir_step!(res, view(Utrain, :, t); rng=rng)

        idxs = get(by_t, t, nothing)
        if idxs !== nothing
            for m in idxs
                i = i_samp[m]
                C[1, m] = res.state.pos[i].x
                C[2, m] = res.state.pos[i].y

                w = max(kth_neighbour_dist(i), 1e-6)
                inv2w[m] = 1 / (2w)
            end
        end

        show_progress && next!(prog)
    end

    return KernelLayer(C, inv2w)
end

# Backward-compatible name used by older notebooks and the original-paper
# reproduction. The observation itself is model-independent; only distance
# geometry (periodic Couzin versus non-periodic Lymburn) dispatches by model.
build_observation_layer_lymburn!(res::CouzinReservoir, Utrain::AbstractMatrix; kwargs...) =
    build_observation_layer_spatial_gaussian!(res, Utrain; kwargs...)

"""
    build_observation_layer_coverage_kmeans!(res, Utrain; washout=0, rng, M=200, stride=5,
                                              pool_per_t=5, kσ=5, σmin=nothing, σmax=nothing)

Alternative placement rule for `build_observation_layer_spatial_gaussian!`:
k-means over the empirical swarm occupancy (positions visited during a
teacher-forced run) rather than by single-agent sampling. Requires the
`Clustering.jl` package (`using Clustering` before calling this function).
Kernel widths are the median distance to each centre's `kσ` nearest
neighbouring centres, clamped to `[σmin, σmax]`.
"""
function build_observation_layer_coverage_kmeans!(res::CouzinReservoir, Utrain::AbstractMatrix;
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    M::Int = 200,
    stride::Int = 5,
    pool_per_t::Int = 5,
    kσ::Int = 5,
    σmin = nothing,
    σmax = nothing,
    eps::Float64 = 1e-6,
    kmeans_maxiter::Int = 200,
    kmeans_tol::Float64 = 1e-4,
    show_progress::Bool = true,
)
    @assert stride ≥ 1 && pool_per_t ≥ 1 && M ≥ 1

    reset!(res; rng=rng)

    Nu, T = size(Utrain)
    @assert T > washout + 1 "washout too large for training length"

    N = res.P.N
    L = res.P.L
    @assert N ≥ 2 "need at least 2 agents"

    σmin === nothing && (σmin = 0.01 * L)
    σmax === nothing && (σmax = 0.25 * L)
    @assert 0 < σmin < σmax

    xs = Float64[]; ys = Float64[]
    sizehint!(xs, ((T - washout) ÷ stride) * pool_per_t)
    sizehint!(ys, ((T - washout) ÷ stride) * pool_per_t)

    prog = show_progress ? Progress(T; desc="build_observation_layer_coverage_kmeans!", showspeed=true) : nothing

    for t in 1:T
        reservoir_step!(res, view(Utrain, :, t); rng=rng)

        if t > washout && (t - washout) % stride == 0
            @inbounds for _ in 1:pool_per_t
                i = rand(rng, 1:N)
                push!(xs, res.state.pos[i].x)
                push!(ys, res.state.pos[i].y)
            end
        end

        show_progress && next!(prog)
    end

    S = length(xs)
    @assert S ≥ M "Not enough pooled samples ($S) to fit M=$M kernels. Increase T, reduce stride, or increase pool_per_t."

    Pmat = Matrix{Float64}(undef, 2, S)
    Pmat[1, :] .= xs
    Pmat[2, :] .= ys

    km = kmeans(Pmat, M; maxiter=kmeans_maxiter, tol=kmeans_tol, rng=rng)
    C = km.centers

    inv2w = zeros(Float64, M)
    d = Vector{Float64}(undef, M - 1)
    k_use = min(kσ, M - 1)
    @assert k_use ≥ 1 "Need M≥2 or smaller kσ."

    @inbounds for m in 1:M
        cx, cy = C[1, m], C[2, m]
        idx = 1
        for j in 1:M
            j == m && continue
            dx, dy = periodic_disp(cx, cy, C[1, j], C[2, j], L)
            d[idx] = hypot(dx, dy)
            idx += 1
        end
        sort!(d)
        σ = clamp(median(@view d[1:k_use]), σmin, σmax)
        inv2w[m] = 1 / (2 * (σ^2 + eps))
    end

    return KernelLayer(C, inv2w)
end

# ============================================================
# Minimal end-to-end example (Project T1-S1 baseline)
# ============================================================
#
#   include("ABM/load_ABM.jl")
#   include("ABM/models/Couzin/load_Couzin.jl")
#   include("RC/my_reservoir_core.jl")
#   include("SWARM_RC/my_swarmRC.jl")
#
#   scenario = Couzin_params_from_preset(:milling; nondim=true, dt=0.1)
#   coupling = PredatorCoupling(rp = 3.0 * scenario.P.Zr, alpha = 1.0)
#   res = build_Couzin_reservoir(scenario.P, scenario.dt; coupling=coupling)
#
#   u = randn(2, 4000)  # replace with a real driving signal
#   out = train_and_evaluate_reservoir(
#       res; data = u, shift = 200, train_len = 2000, predict_len = 500,
#       washout = 100, ridgeλ = 1e-3,
#       compute_reservoir_diagnostics = true,
#   )
#
#   out.memory_diag, out.separability_diag, out.stability_diag

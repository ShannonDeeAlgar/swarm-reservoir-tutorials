# Two-state toroidal boid model from Lund et al. (2026).

using Random
using LinearAlgebra

Base.@kwdef struct LundParams
    N::Int = 200
    L::Float64 = 512.0
    dt::Float64 = 0.1
    init_speed::Float64 = 1.0
    base_speed::Float64 = 1.0
    temperature_gain::Float64 = 0.30
    min_speed::Float64 = 0.1
    max_speed::Float64 = 5.0
    speed_relax::Float64 = 0.7
    inertia_alpha::Float64 = 0.4
    speed_clip::Float64 = 10.0
    noise_std::Float64 = 0.01
    inner_radius::Float64 = 15.0
    local_radius::Float64 = 35.0
    energy_threshold::Float64 = 1.70
    hysteresis_factor::Float64 = 0.10
    dispersed_radius::Float64 = 28.0
    dispersed_alignment::Float64 = 0.5
    dispersed_cohesion::Float64 = 0.5
    dispersed_separation::Float64 = 1.2
    clustered_radius::Float64 = 35.0
    clustered_alignment::Float64 = 1.0
    clustered_cohesion::Float64 = 2.0
    clustered_separation::Float64 = 1.0
    anchoring_weight::Float64 = 0.01
end

"""
    Lund_params_from_preset(preset; N=nothing)

Return the `:tutorial` parameters or the companion repository's default
two-state `:paper` parameters. Use `N` to override the default population.
"""
function Lund_params_from_preset(preset::Symbol; N::Union{Nothing,Int}=nothing)
    if preset == :tutorial
        return LundParams(N=something(N, 200), L=256.0, dt=0.1,
            energy_threshold=1.10, hysteresis_factor=0.10,
            inertia_alpha=0.4)
    elseif preset == :paper
        return LundParams(N=something(N, 200), L=512.0, dt=0.1,
            init_speed=1.0, base_speed=1.0, temperature_gain=0.30,
            min_speed=0.1, max_speed=5.0, speed_relax=0.7,
            inertia_alpha=1.0, speed_clip=10.0, noise_std=0.01,
            inner_radius=15.0, local_radius=35.0,
            energy_threshold=3.0, hysteresis_factor=0.05,
            dispersed_radius=28.0, dispersed_alignment=0.5,
            dispersed_cohesion=0.5, dispersed_separation=1.2,
            clustered_radius=35.0, clustered_alignment=1.0,
            clustered_cohesion=2.0, clustered_separation=1.0,
            anchoring_weight=0.01)
    end
    throw(ArgumentError("Unknown Lund preset :$preset. Use :tutorial or :paper."))
end

@inline lund_displacement_from_to(a::SVector2, b::SVector2, L::Real) =
    -lund_displacement(a, b, L)

"Short CPU demonstration driver; paper-scale MC runs use N≥800."
function simulate_Lund_2d(P::LundParams; steps::Int=500,
    input=zeros(steps), rng::AbstractRNG=Random.default_rng())
    length(input) >= steps || throw(ArgumentError("input is shorter than steps."))
    state = init_state_Lund_2d(P; rng=rng)
    pos_hist = Vector{Vector{SVector2}}(undef, steps + 1)
    vel_hist = Vector{Vector{SVector2}}(undef, steps + 1)
    state_hist = Vector{Vector{Int8}}(undef, steps + 1)
    dilation = zeros(steps + 1); rotation = similar(dilation); polarisation = similar(dilation)
    function save!(k)
        pos_hist[k], vel_hist[k], state_hist[k] = copy(state.pos), copy(state.vel), copy(state.agent_state)
        d, r, p, _ = order_parameters_2d(state.pos, state.vel, P.L;
            displacement=lund_displacement_from_to)
        dilation[k], rotation[k], polarisation[k] = d, r, p
    end
    save!(1)
    for t in 1:steps
        Lund_step_2d!(state, P, input[t]; rng=rng)
        save!(t + 1)
    end
    return (; t=collect(0:P.dt:steps*P.dt), pos_hist, vel_hist, state_hist,
        dilation, rotation, polarisation)
end

mutable struct LundState
    pos::Vector{SVector2}
    vel::Vector{SVector2}
    agent_state::Vector{Int8}  # 0 dispersed, 1 clustered
    local_energy::Vector{Float64}
end

function init_state_Lund_2d(P::LundParams; rng::AbstractRNG=Random.default_rng())
    pos = [P.L * SVector2(rand(rng), rand(rng)) for _ in 1:P.N]
    vel = [P.init_speed * SVector2(cos(θ), sin(θ)) for θ in 2π .* rand(rng, P.N)]
    LundState(pos, vel, zeros(Int8, P.N), zeros(P.N))
end

copy_state(state::LundState) = LundState(copy(state.pos), copy(state.vel),
    copy(state.agent_state), copy(state.local_energy))

@inline function lund_displacement(a::SVector2, b::SVector2, L::Real)
    d = a - b
    return SVector2(d.x - round(d.x / L) * L,
                    d.y - round(d.y / L) * L)
end

function _lund_local_energy_and_states!(state::LundState, P::LundParams)
    up = P.energy_threshold * (1 + P.hysteresis_factor)
    down = P.energy_threshold * (1 - P.hysteresis_factor)
    @inbounds for i in 1:P.N
        total = 0.0
        count = 0
        for j in 1:P.N
            i == j && continue
            if norm(lund_displacement(state.pos[i], state.pos[j], P.L)) < P.local_radius
                total += norm(state.vel[j])
                count += 1
            end
        end
        e = count == 0 ? norm(state.vel[i]) : total / count
        state.local_energy[i] = e
        state.agent_state[i] == 0 && e > up && (state.agent_state[i] = 1)
        state.agent_state[i] == 1 && e < down && (state.agent_state[i] = 0)
    end
    return state
end

function Lund_step_2d!(state::LundState, P::LundParams, scalar_input::Real;
    target_speed_override::Union{Nothing,Real}=nothing,
    rng::AbstractRNG=Random.default_rng())
    _lund_local_energy_and_states!(state, P)
    acc = fill(SVector2(0.0, 0.0), P.N)

    @inbounds for i in 1:P.N
        clustered = state.agent_state[i] == 1
        radius = clustered ? P.clustered_radius : P.dispersed_radius
        ka = clustered ? P.clustered_alignment : P.dispersed_alignment
        kc = clustered ? P.clustered_cohesion : P.dispersed_cohesion
        ks = clustered ? P.clustered_separation : P.dispersed_separation
        mean_v = SVector2(0.0, 0.0)
        mean_p = SVector2(0.0, 0.0)
        separation = SVector2(0.0, 0.0)
        count = 0
        for j in 1:P.N
            i == j && continue
            dij = lund_displacement(state.pos[i], state.pos[j], P.L)
            r = norm(dij)
            if r < radius
                mean_v += state.vel[j]
                mean_p += state.pos[j]  # faithful to the released implementation
                count += 1
            end
            r < P.inner_radius && (separation += dij)
        end
        if count > 0
            mean_v /= count
            mean_p /= count
            acc[i] = ks * separation + ka * (mean_v - state.vel[i]) +
                kc * (mean_p - state.pos[i])
        end
        acc[i] += P.anchoring_weight * (SVector2(P.L/2, P.L/2) - state.pos[i])
    end

    starget = target_speed_override === nothing ?
        clamp(P.base_speed + P.temperature_gain * float(scalar_input),
            P.min_speed, P.max_speed) : float(target_speed_override)
    @inbounds for i in 1:P.N
        v = state.vel[i] + P.inertia_alpha * acc[i] * P.dt
        P.noise_std > 0 && (v += P.noise_std * SVector2(randn(rng), randn(rng)))
        speed = norm(v)
        desired = speed > 1e-12 ? starget * v / speed : SVector2(starget, 0.0)
        v = (1 - P.speed_relax) * v + P.speed_relax * desired
        speed = norm(v)
        speed > P.speed_clip && (v *= P.speed_clip / speed)
        state.vel[i] = v
        p = state.pos[i] + v * P.dt
        state.pos[i] = SVector2(mod(p.x, P.L), mod(p.y, P.L))
    end
    return state
end

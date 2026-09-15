# ============================================================
# my_Lymburn_Rules2d.jl
# 2D dynamics for the Lymburn swarm model: forces (Eqs 1-4, 10), the
# sigmoidal force cap (Eq 6), and explicit-Euler integration (Eqs 8-9).
# ============================================================

using Random
using LinearAlgebra

# No periodic domain in this model -- defined locally (mirrors the same
# helper already used by ABM/models/Helbing/my_Helbing_Rules2d.jl for its
# non-periodic boundary mode).
@inline displacement_nonperiodic_2d(a::SVector2, b::SVector2, L::Real) = b - a

# ============================================================
# Initial condition
# ============================================================

"""
    init_state_Lymburn_2d(P; rng, init_radius=5.0) -> SwarmState{SVector2}

Agents start scattered uniformly within `init_radius` of the home point
`P.xh`, each moving at the target cruising speed `P.s` in a random
direction.
"""
function init_state_Lymburn_2d(P::LymburnParams; rng::AbstractRNG = Random.default_rng(), init_radius::Float64 = 5.0)
    N = P.N
    pos = [P.xh + init_radius * SVector2(2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:N]
    vel = [P.s * SVector2(cos(θ), sin(θ)) for θ in (2π * rand(rng) for _ in 1:N)]
    return SwarmState{SVector2}(pos, vel, copy(vel))
end

# ============================================================
# Forces (Eqs 1-4, 6, 10)
# ============================================================

"""
    Lymburn_total_force_2d(pos, vel, P; xp=nothing, rng=Random.default_rng()) -> Vector{SVector2}

Total capped force on each agent: repulsion (Eq 1) + alignment (Eq 2) +
homing (Eq 3) + friction (Eq 4), plus the predator term (Eq 10) if `xp` is
given and `P.Kp != 0`, plus per-step Gaussian noise if `P.noise_amp > 0`,
then the sigmoidal cap (Eq 6). Repulsion/alignment sums are raw sums over
neighbours within `P.rr`/`P.ra` (not averaged), matching the paper exactly.

**`P.cap_mode`** controls how the cap (Eq 6) is applied -- confirmed
against the paper's own released MATLAB code (`run_swarm_res.m`):
- `:componentwise` (default, matches the MATLAB exactly): `tanh` applied
  to each Cartesian component of the force separately,
  `F ↦ α·tanh(β·F)` elementwise. Not rotationally symmetric -- caps a
  force pointing along an axis differently than one at 45°, producing a
  measurable cardinal-direction density bias (confirmed present even
  fully undriven, `Kp=0`) and a smaller swarm footprint than the
  isotropic alternative. This is what actually generated the paper's
  published results, not a simplification of them.
- `:isotropic`: cap applied to the force vector's *magnitude*, preserving
  its direction: `F ↦ (α·tanh(β|F|)/|F|)·F`. Rotationally symmetric, and
  was this codebase's default for a while under the (reasonable-looking,
  but unverified until the MATLAB source was checked) assumption that the
  componentwise version was a bug. Kept available for comparison -- the
  difference between the two is itself a good demonstration of how a
  small implementation choice can measurably change emergent swarm shape.

**`P.noise_amp`**: adds `noise_amp .* SVector2(randn(rng), randn(rng))` to
the pre-cap force sum each step, matching the MATLAB's
`noise_amp*randn(size(v))` term. Zero by default, matching the paper's
own default parameters (the term exists in their code but isn't used in
the published configuration) -- available for exploring stochastically-
forced dynamics.
"""
function Lymburn_total_force_2d(
    pos::Vector{SVector2}, vel::Vector{SVector2}, P::LymburnParams;
    xp::Union{Nothing,SVector2} = nothing,
    predator_gain::Float64 = 1.0,
    predator_radius::Float64 = P.rp,
    predator_flee::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
)
    N = length(pos)
    F = Vector{SVector2}(undef, N)

    @inbounds for i in 1:N
        Fr = SVector2(0.0, 0.0)
        Fa = SVector2(0.0, 0.0)

        for j in 1:N
            j == i && continue
            d = pos[i] - pos[j]
            r = norm(d)

            if r <= P.rr && r > 1e-12
                Fr = Fr + d / (r * r)              # Eq 1
            end
            if r <= P.ra
                Fa = Fa + (vel[j] - vel[i])         # Eq 2
            end
        end

        Fh = P.xh - pos[i]                          # Eq 3

        speed = norm(vel[i])
        Ff = speed > 1e-12 ? (-vel[i] / speed) * ((speed - P.s) / P.s) : SVector2(0.0, 0.0)   # Eq 4

        Ftot = P.Ka * Fa + P.Kr * Fr + P.Kf * Ff + P.Kh * Fh

        if xp !== nothing && P.Kp != 0 && predator_gain != 0
            dp = pos[i] - xp
            rp_dist = norm(dp)
            if rp_dist <= predator_radius && rp_dist > 1e-12
                direction = predator_flee ? dp : -dp
                Ftot = Ftot + predator_gain * P.Kp * (direction / (rp_dist * rp_dist))   # Eq 10
            end
        end

        if P.noise_amp > 0
            Ftot = Ftot + P.noise_amp * SVector2(randn(rng), randn(rng))
        end

        if P.cap_mode === :componentwise
            F[i] = SVector2(P.alpha * tanh(P.beta * Ftot.x), P.alpha * tanh(P.beta * Ftot.y))   # Eq 6
        elseif P.cap_mode === :isotropic
            Fmag = norm(Ftot)
            F[i] = Fmag > 1e-12 ? (P.alpha * tanh(P.beta * Fmag) / Fmag) * Ftot : SVector2(0.0, 0.0)
        else
            error("Unknown cap_mode = $(P.cap_mode). Use :componentwise or :isotropic.")
        end
    end

    return F
end

# ============================================================
# Integration (Eqs 8-9)
# ============================================================
#
# P.integration_scheme controls whether the position update uses the OLD
# velocity v(t) (:explicit, plain Euler) or the just-updated v(t+dt)
# (:semi_implicit, symplectic/Euler-Cromer). Confirmed against the paper's
# own released MATLAB (`run_swarm_res.m`: `x(t+1) = x(t) + dt*v(t+1)`) --
# :semi_implicit generated the published results and is the default here.

function _lymburn_euler_update!(state::SwarmState{SVector2}, F::Vector{SVector2}, dt::Float64, scheme::Symbol)
    N = length(state.pos)
    if scheme === :semi_implicit
        @inbounds for i in 1:N
            v_new = state.vel[i] + F[i] * dt                # Eq 8, computed first
            state.pos[i] = state.pos[i] + v_new * dt         # Eq 9 -- uses v(t+dt)
            state.vel[i] = v_new
        end
    elseif scheme === :explicit
        @inbounds for i in 1:N
            v_old = state.vel[i]
            state.pos[i] = state.pos[i] + v_old * dt         # Eq 9 -- uses v(t)
            state.vel[i] = v_old + F[i] * dt                 # Eq 8
        end
    else
        error("Unknown integration_scheme = $scheme. Use :semi_implicit or :explicit.")
    end
    return state
end

"""
    Lymburn_step_2d!(state, P, dt; rng, update_scheme)

Undriven step -- matches the generic `simulate` driver's `step!` interface.
"""
function Lymburn_step_2d!(state::SwarmState{SVector2}, P::LymburnParams, dt::Float64;
    rng::AbstractRNG = Random.default_rng(), update_scheme::Symbol = :synchronous,
)
    F = Lymburn_total_force_2d(state.pos, state.vel, P; rng = rng)
    return _lymburn_euler_update!(state, F, dt, P.integration_scheme)
end

"""
    Lymburn_step_with_input!(state, P, dt, xp; rng, update_scheme)

Predator-driven step: `xp` is the predator's raw 2D position this
timestep (the external input signal, entering directly via Eq 10 -- no
"desired direction" indirection).
"""
function Lymburn_step_with_input!(state::SwarmState{SVector2}, P::LymburnParams, dt::Float64, xp::SVector2;
    predator_gain::Float64 = 1.0,
    predator_radius::Float64 = P.rp,
    predator_flee::Bool = true,
    rng::AbstractRNG = Random.default_rng(), update_scheme::Symbol = :synchronous,
)
    F = Lymburn_total_force_2d(state.pos, state.vel, P;
        xp = xp, predator_gain = predator_gain,
        predator_radius = predator_radius, predator_flee = predator_flee,
        rng = rng)
    return _lymburn_euler_update!(state, F, dt, P.integration_scheme)
end

# ============================================================
# Simulation driver
# ============================================================

"""
    simulate_Lymburn_2d(simcfg, P; init_radius=5.0, kwargs...) -> SimulationOutput

Wraps the generic `simulate` driver (`ABM/my_ABM_core.jl`) with the
Lymburn model's init/step/order-parameter callbacks. No periodic domain:
`displacement_nonperiodic_2d` is used for order-parameter calculations, and
`domain_size` returns `P.plot_extent` (cosmetic only -- plays no role in
the dynamics).
"""
function simulate_Lymburn_2d(
    simcfg::SimulationConfig, P::LymburnParams;
    init_mode::Symbol = :clustered,   # kept for interface symmetry with Couzin/Helbing; unused (Lymburn has one init style)
    dt_override::Union{Nothing,Float64} = nothing,
    init_radius::Float64 = 5.0,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    dt = dt_override !== nothing ? dt_override : simcfg.dt

    init_state_fn = (P; rng = Random.default_rng()) -> init_state_Lymburn_2d(P; rng = rng, init_radius = init_radius)

    return simulate(
        simcfg, P;
        dt               = dt,
        init_state       = init_state_fn,
        step!            = Lymburn_step_2d!,
        get_pos          = state -> state.pos,
        get_vel          = state -> state.vel,
        domain_size      = (P, state) -> P.plot_extent,
        displacement     = displacement_nonperiodic_2d,
        order_parameters = Lymburn_order_parameters_2d,
        collect_order    = collect_order,
        collect_history  = collect_history,
        show_progress    = show_progress,
    )
end

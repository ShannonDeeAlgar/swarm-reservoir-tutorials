# my_Helbing_Rules3d.jl

# Helbing--Molnar style social-force model in 3D.
#
# This file mirrors my_Helbing_Rules2d.jl, but uses SVector3 states.
# It assumes the generic ABM machinery already provides:
#   - SVector3
#   - SwarmState{SVector3}
#   - SimulationConfig
#   - simulate
#   - order_parameters
#   - displacement, if periodic boundaries are requested
#
# The model-specific parameters live in my_Helbing_core.jl.

using LinearAlgebra
using Random

# ============================================================
# Small helpers
# ============================================================

@inline positive_part(x::Float64) = max(0.0, x)

@inline function safe_unit_3d(v::SVector3; fallback::SVector3 = SVector3(1.0, 0.0, 0.0))
    nv = norm(v)
    return nv < 1e-12 ? fallback : v / nv
end

# Non-periodic displacement from a to b.
# This has the same argument pattern as the periodic displacement helper used
# elsewhere in the ABM code: displacement(a, b, L).
@inline displacement_nonperiodic_3d(a::SVector3, b::SVector3, L::Float64) = b - a

@inline function Helbing_vector_i_to_j_3d(pi::SVector3, pj::SVector3, P::HelbingParams)
    if P.boundary == :periodic
        return displacement(pi, pj, P.L)
    else
        return pj - pi
    end
end

@inline function clamp_speed_3d(v::SVector3, max_speed::Float64)
    s = norm(v)
    return s <= max_speed || s < 1e-12 ? v : (max_speed / s) * v
end

@inline function wrap_position_3d(p::SVector3, L::Float64)
    return SVector3(mod(p.x, L), mod(p.y, L), mod(p.z, L))
end

@inline function clamp_position_3d(p::SVector3, P::HelbingParams)
    lo = P.radius
    hi = P.L - P.radius
    return SVector3(
        clamp(p.x, lo, hi),
        clamp(p.y, lo, hi),
        clamp(p.z, lo, hi),
    )
end

# ============================================================
# Initialisation helpers
# ============================================================

function Helbing_random_point_in_ball_3d(rng::AbstractRNG, R::Float64)
    # Direction from a normal vector; radius distributed uniformly in volume.
    x = randn(rng)
    y = randn(rng)
    z = randn(rng)
    dir = safe_unit_3d(SVector3(x, y, z))
    r = R * rand(rng)^(1 / 3)
    return r * dir
end

Helbing_random_point_in_cube_3d(rng::AbstractRNG, L::Float64) =
    SVector3(L * rand(rng), L * rand(rng), L * rand(rng))

function Helbing_random_point_in_left_slab_3d(
    rng::AbstractRNG,
    L::Float64;
    xwidth::Float64 = 0.15L,
    ypad::Float64 = 0.10L,
    zpad::Float64 = 0.10L,
)
    x = xwidth * rand(rng)
    y = ypad + (L - 2ypad) * rand(rng)
    z = zpad + (L - 2zpad) * rand(rng)
    return SVector3(x, y, z)
end

function Helbing_random_point_in_right_slab_3d(
    rng::AbstractRNG,
    L::Float64;
    xwidth::Float64 = 0.15L,
    ypad::Float64   = 0.10L,
    zpad::Float64   = 0.10L,
)
    x = L - xwidth * rand(rng)
    y = ypad + (L - 2ypad) * rand(rng)
    z = zpad + (L - 2zpad) * rand(rng)
    return SVector3(x, y, z)
end
# ============================================================
# Helbing--Molnar rules in 3D
# ============================================================

function desired_direction_Helbing_3d(
    i::Int,
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    P::HelbingParams,
)
    if P.drive_mode == :goal
        P.goals === nothing &&
            error("drive_mode = :goal requires P.goals to be non-nothing.")

        to_goal = P.goals[i] - pos[i]
        return safe_unit_3d(to_goal; fallback = safe_unit_3d(vel[i]))

    elseif P.drive_mode == :heading
        P.headings === nothing &&
            error("drive_mode = :heading requires P.headings to be non-nothing.")

        return safe_unit_3d(P.headings[i]; fallback = safe_unit_3d(vel[i]))

    elseif P.drive_mode == :none
        return SVector3(0.0, 0.0, 0.0)

    else
        error("Unknown drive_mode = $(P.drive_mode). Use :goal, :heading, or :none.")
    end
end

function driving_acceleration_Helbing_3d(
    i::Int,
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    P::HelbingParams,
)
    P.drive_mode == :none && return SVector3(0.0, 0.0, 0.0)

    e_i = desired_direction_Helbing_3d(i, pos, vel, P)
    v_desired = P.v0 * e_i

    return (v_desired - vel[i]) / P.tau
end

function agent_force_Helbing_3d(
    i::Int,
    j::Int,
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    P::HelbingParams,
)
    P.interaction_mode == :none && return SVector3(0.0, 0.0, 0.0)

    d_i_to_j = Helbing_vector_i_to_j_3d(pos[i], pos[j], P)
    dist = norm(d_i_to_j)
    dist < 1e-12 && return SVector3(0.0, 0.0, 0.0)

    # n_ij points from j to i, so the force on i is repulsive.
    n_ij = -d_i_to_j / dist

    overlap = 2P.radius - dist
    g = positive_part(overlap)

    social = P.A_agent * exp(overlap / P.B_agent) * n_ij
    body = P.k_body * g * n_ij

    # In 3D there is no unique tangent direction. Use the full tangential
    # component of the relative velocity, i.e. remove the normal component.
    Δv = vel[j] - vel[i]
    Δv_t = Δv - dot(Δv, n_ij) * n_ij
    friction = P.k_friction * g * Δv_t

    if P.interaction_mode == :social_force
        return social + body + friction

    elseif P.interaction_mode == :soft_repulsion
        return social

    elseif P.interaction_mode == :contact_only
        return body + friction

    else
        error("Unknown interaction_mode = $(P.interaction_mode). Use :social_force, :soft_repulsion, :contact_only, or :none.")
    end
end

function wall_force_Helbing_3d(
    pi::SVector3,
    vi::SVector3,
    P::HelbingParams,
)
    P.boundary == :walls || return SVector3(0.0, 0.0, 0.0)
    P.wall_mode == :social_force || return SVector3(0.0, 0.0, 0.0)

    F = SVector3(0.0, 0.0, 0.0)

    # Each tuple is (distance to wall, inward normal).
    walls = (
        (pi.x,       SVector3( 1.0,  0.0,  0.0)), # left
        (P.L - pi.x, SVector3(-1.0,  0.0,  0.0)), # right
        (pi.y,       SVector3( 0.0,  1.0,  0.0)), # bottom/front
        (P.L - pi.y, SVector3( 0.0, -1.0,  0.0)), # top/back
        (pi.z,       SVector3( 0.0,  0.0,  1.0)), # lower
        (P.L - pi.z, SVector3( 0.0,  0.0, -1.0)), # upper
    )

    for (dist, n_iw) in walls
        overlap = P.radius - dist
        g = positive_part(overlap)

        social = P.A_wall * exp(overlap / P.B_wall) * n_iw
        body = P.k_body * g * n_iw

        # Wall friction opposes the component of velocity tangent to the wall.
        v_t = vi - dot(vi, n_iw) * n_iw
        friction = -P.k_friction * g * v_t

        F += social + body + friction
    end

    return F
end

function Helbing_acceleration_3d(
    i::Int,
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    P::HelbingParams,
)
    F = SVector3(0.0, 0.0, 0.0)

    @inbounds for j in eachindex(pos)
        j == i && continue
        F += agent_force_Helbing_3d(i, j, pos, vel, P)
    end

    F += wall_force_Helbing_3d(pos[i], vel[i], P)

    return driving_acceleration_Helbing_3d(i, pos, vel, P) + F / P.mass
end

function update_velocity_Helbing_3d(
    vi::SVector3,
    ai::SVector3,
    P::HelbingParams,
    dt::Float64,
)
    return clamp_speed_3d(vi + dt * ai, P.max_speed)
end

function update_position_Helbing_3d(
    pi::SVector3,
    vi::SVector3,
    P::HelbingParams,
    dt::Float64,
)
    p = pi + dt * vi

    if P.boundary == :periodic
        return wrap_position_3d(p, P.L)

    elseif P.boundary == :walls
        if P.wall_mode == :hard_clamp || P.wall_mode == :social_force
            return clamp_position_3d(p, P)
        elseif P.wall_mode == :none
            return p
        else
            error("Unknown wall_mode = $(P.wall_mode). Use :social_force, :hard_clamp, or :none.")
        end

    elseif P.boundary == :none
        return p

    else
        error("Unknown boundary = $(P.boundary). Use :walls, :periodic, or :none.")
    end
end

function Helbing_step_3d!(
    state::SwarmState{SVector3},
    P::HelbingParams,
    dt::Float64;
    rng = Random.default_rng(),
    update_scheme::Symbol = :synchronous,
)
    pos = state.pos
    vel = state.vel
    desired = state.desired

    if update_scheme == :synchronous
        acc = Vector{SVector3}(undef, length(pos))

        @inbounds for i in eachindex(pos)
            desired[i] = P.v0 * desired_direction_Helbing_3d(i, pos, vel, P)
            acc[i] = Helbing_acceleration_3d(i, pos, vel, P)
        end
        @inbounds for i in eachindex(vel)
            vel[i] = update_velocity_Helbing_3d(vel[i], acc[i], P, dt)
        end
        @inbounds for i in eachindex(pos)
            pos[i] = update_position_Helbing_3d(pos[i], vel[i], P, dt)
        end

    elseif update_scheme == :asynchronous
        order = randperm(rng, length(pos))
        @inbounds for i in order
            desired[i] = P.v0 * desired_direction_Helbing_3d(i, pos, vel, P)
            acc_i = Helbing_acceleration_3d(i, pos, vel, P)
            vel[i] = update_velocity_Helbing_3d(vel[i], acc_i, P, dt)
            pos[i] = update_position_Helbing_3d(pos[i], vel[i], P, dt)
        end

    else
        error("Unknown update_scheme = $(update_scheme). Use :synchronous or :asynchronous.")
    end

    return nothing
end

# ============================================================
# Initialisation
# ============================================================

function initial_desired_velocity_Helbing_3d(
    i::Int,
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    P::HelbingParams,
)
    if P.drive_mode == :none
        return SVector3(0.0, 0.0, 0.0)
    else
        return P.v0 * desired_direction_Helbing_3d(i, pos, vel, P)
    end
end

function place_in_domain_Helbing_3d(pi::SVector3, P::HelbingParams)
    if P.boundary == :periodic
        return wrap_position_3d(pi, P.L)

    elseif P.boundary == :walls
        return clamp_position_3d(pi, P)

    elseif P.boundary == :none
        return pi

    else
        error("Unknown boundary = $(P.boundary). Use :walls, :periodic, or :none.")
    end
end

function init_state_Helbing_3d(
    P::HelbingParams;
    rng            = Random.default_rng(),
    init_mode      ::Symbol   = :uniform,
    R0             ::Float64  = max(P.L / 10, 5P.radius),
    c0             ::SVector3 = SVector3(P.L / 4, P.L / 2, P.L / 2),
    start_at_rest  ::Bool     = true,
)
    if init_mode == :uniform && (R0 != max(P.L / 10, 5P.radius) || c0 != SVector3(P.L / 4, P.L / 2, P.L / 2))
        @warn "R0 and c0 are ignored for :uniform initialisation over the entire domain."
    end

    pos = Vector{SVector3}(undef, P.N)
    vel = Vector{SVector3}(undef, P.N)

    @inbounds for i in 1:P.N
        pi = if init_mode == :clustered
            c0 + Helbing_random_point_in_ball_3d(rng, R0)

        elseif init_mode == :uniform
            Helbing_random_point_in_cube_3d(rng, P.L)

        elseif init_mode == :left_slab
            Helbing_random_point_in_left_slab_3d(rng, P.L)

        elseif init_mode == :right_slab
            Helbing_random_point_in_right_slab_3d(rng, P.L)

        elseif init_mode == :two_slabs
            # First half starts on the left (heading right),
            # second half starts on the right (heading left).
            # Matches the two-group split used by :lane_formation
            # and :heading drive_mode presets.
            if i <= P.N ÷ 2
                Helbing_random_point_in_left_slab_3d(rng, P.L)
            else
                Helbing_random_point_in_right_slab_3d(rng, P.L)
            end

        else
            error("Unknown init_mode = $(init_mode). " *
                  "Use :uniform, :clustered, :left_slab, :right_slab, or :two_slabs.")
        end

        pos[i] = place_in_domain_Helbing_3d(pi, P)
    end

    @inbounds for i in 1:P.N
        e_i = if P.drive_mode == :goal
            P.goals === nothing &&
                error("drive_mode = :goal requires P.goals to be non-nothing.")
            safe_unit_3d(P.goals[i] - pos[i])

        elseif P.drive_mode == :heading
            P.headings === nothing &&
                error("drive_mode = :heading requires P.headings to be non-nothing.")
            safe_unit_3d(P.headings[i])

        elseif P.drive_mode == :none
            SVector3(0.0, 0.0, 0.0)

        else
            error("Unknown drive_mode = $(P.drive_mode). Use :goal, :heading, or :none.")
        end

        vel[i] = start_at_rest ? SVector3(0.0, 0.0, 0.0) : P.v0 * e_i
    end

    desired = [
        initial_desired_velocity_Helbing_3d(i, pos, vel, P)
        for i in 1:P.N
    ]

    return SwarmState{SVector3}(pos, vel, desired)
end

# ============================================================
# Simulation
# ============================================================

function simulate_Helbing_3d(
    simcfg        ::SimulationConfig,
    P             ::HelbingParams;
    init_mode     ::Symbol   = :two_slabs,
    dt_override::Union{Nothing, Float64} = nothing,
    R0            ::Float64  = max(P.L / 10, 5P.radius),
    c0            ::SVector3 = SVector3(P.L / 4, P.L / 2, P.L / 2),
    start_at_rest ::Bool     = true,
    collect_order ::Bool     = true,
    collect_history::Bool    = true,
    show_progress ::Bool     = true,
)
    # Opposing slabs create an immediate interaction; pass another init_mode
    # when a different initial layout is required.

    dt = dt_override !== nothing ? dt_override : simcfg.dt

    init_state_fn = (P; rng = Random.default_rng()) -> init_state_Helbing_3d(
        P;
        rng           = rng,
        init_mode     = init_mode,
        R0            = R0,
        c0            = c0,
        start_at_rest = start_at_rest,
    )

    displacement_fn = P.boundary == :periodic ? displacement : displacement_nonperiodic_3d

    simulate(
        simcfg, P;
        init_state       = init_state_fn,
        step!            = Helbing_step_3d!,
        get_pos          = state -> state.pos,
        get_vel          = state -> state.vel,
        domain_size      = (P, state) -> P.L,
        displacement     = displacement_fn,
        order_parameters = order_parameters,
        collect_order    = collect_order,
        collect_history  = collect_history,
        show_progress    = show_progress,
    )
end



function simulate_Helbing_3d(simcfg::SimulationConfig, scenario::HelbingScenario; kwargs...)
    simulate_Helbing_3d(simcfg, scenario.P; init_mode = scenario.init_mode, kwargs...)
end

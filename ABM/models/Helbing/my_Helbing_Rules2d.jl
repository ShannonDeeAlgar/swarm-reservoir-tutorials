# ============================================================
# Helbing--Molnar rules
# ============================================================

function desired_direction_Helbing_2d(
    i::Int,
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::HelbingParams,
)
    if P.drive_mode == :goal
        P.goals === nothing &&
            error("drive_mode = :goal requires P.goals to be non-nothing.")

        to_goal = P.goals[i] - pos[i]
        return safe_unit_2d(to_goal; fallback = safe_unit_2d(vel[i]))

    elseif P.drive_mode == :heading
        P.headings === nothing &&
            error("drive_mode = :heading requires P.headings to be non-nothing.")

        return safe_unit_2d(P.headings[i]; fallback = safe_unit_2d(vel[i]))

    elseif P.drive_mode == :none
        return SVector2(0.0, 0.0)

    else
        error("Unknown drive_mode = $(P.drive_mode). Use :goal, :heading, or :none.")
    end
end


function driving_acceleration_Helbing_2d(
    i::Int,
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::HelbingParams,
)
    P.drive_mode == :none && return SVector2(0.0, 0.0)

    e_i = desired_direction_Helbing_2d(i, pos, vel, P)
    v_desired = P.v0 * e_i

    return (v_desired - vel[i]) / P.tau
end


function agent_force_Helbing_2d(
    i::Int,
    j::Int,
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::HelbingParams,
)
    P.interaction_mode == :none && return SVector2(0.0, 0.0)

    d_i_to_j = Helbing_vector_i_to_j(pos[i], pos[j], P)
    dist = norm(d_i_to_j)
    dist < 1e-12 && return SVector2(0.0, 0.0)

    # n_ij points from j to i, so the force on i is repulsive.
    n_ij = -d_i_to_j / dist
    t_ij = SVector2(-n_ij.y, n_ij.x)

    overlap = 2P.radius - dist
    g = positive_part(overlap)

    social = P.A_agent * exp(overlap / P.B_agent) * n_ij
    body = P.k_body * g * n_ij

    Δv_t = dot(vel[j] - vel[i], t_ij)
    friction = P.k_friction * g * Δv_t * t_ij

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


function wall_force_Helbing_2d(
    pi::SVector2,
    vi::SVector2,
    P::HelbingParams,
)
    P.boundary == :walls || return SVector2(0.0, 0.0)
    P.wall_mode == :social_force || return SVector2(0.0, 0.0)

    F = SVector2(0.0, 0.0)

    walls = (
        (pi.x,       SVector2( 1.0,  0.0)), # left
        (P.L - pi.x, SVector2(-1.0,  0.0)), # right
        (pi.y,       SVector2( 0.0,  1.0)), # bottom
        (P.L - pi.y, SVector2( 0.0, -1.0)), # top
    )

    for (dist, n_iw) in walls
        overlap = P.radius - dist
        g = positive_part(overlap)

        social = P.A_wall * exp(overlap / P.B_wall) * n_iw
        body = P.k_body * g * n_iw

        t_iw = SVector2(-n_iw.y, n_iw.x)
        friction = -P.k_friction * g * dot(vi, t_iw) * t_iw

        F += social + body + friction
    end

    return F
end


function Helbing_acceleration_2d(
    i::Int,
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::HelbingParams,
)
    F = SVector2(0.0, 0.0)

    @inbounds for j in eachindex(pos)
        j == i && continue
        F += agent_force_Helbing_2d(i, j, pos, vel, P)
    end

    F += wall_force_Helbing_2d(pos[i], vel[i], P)

    return driving_acceleration_Helbing_2d(i, pos, vel, P) + F / P.mass
end


function update_velocity_Helbing_2d(
    vi::SVector2,
    ai::SVector2,
    P::HelbingParams,
    dt::Float64,
)
    return clamp_speed_2d(vi + dt * ai, P.max_speed)
end


function update_position_Helbing_2d(
    pi::SVector2,
    vi::SVector2,
    P::HelbingParams,
    dt::Float64,
)
    p = pi + dt * vi

    if P.boundary == :periodic
        return SVector2(mod(p.x, P.L), mod(p.y, P.L))

    elseif P.boundary == :walls
        if P.wall_mode == :hard_clamp || P.wall_mode == :social_force
            lo = P.radius
            hi = P.L - P.radius
            return SVector2(clamp(p.x, lo, hi), clamp(p.y, lo, hi))
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


function _helbing_apply_boundary_2d!(pos, vel, i, p_new, P)
    if P.boundary == :periodic
        pos[i] = SVector2(mod(p_new.x, P.L), mod(p_new.y, P.L))
    elseif P.boundary == :walls
        if P.wall_mode == :social_force || P.wall_mode == :hard_clamp
            pos[i], vel[i] = apply_wall_clamp_velocity_Helbing_2d(p_new, vel[i], P)
        elseif P.wall_mode == :none
            pos[i] = p_new
        else
            error("Unknown wall_mode = $(P.wall_mode). Use :social_force, :hard_clamp, or :none.")
        end
    elseif P.boundary == :none
        pos[i] = p_new
    else
        error("Unknown boundary = $(P.boundary). Use :walls, :periodic, or :none.")
    end
    return nothing
end

function Helbing_step_2d!(
    state::SwarmState{SVector2},
    P::HelbingParams,
    dt::Float64;
    rng = Random.default_rng(),
    update_scheme::Symbol = :synchronous,
)
    pos = state.pos
    vel = state.vel
    desired = state.desired

    if update_scheme == :synchronous
        acc = Vector{SVector2}(undef, length(pos))

        @inbounds for i in eachindex(pos)
            desired[i] = P.v0 * desired_direction_Helbing_2d(i, pos, vel, P)
            acc[i] = Helbing_acceleration_2d(i, pos, vel, P)
        end
        @inbounds for i in eachindex(vel)
            vel[i] = update_velocity_Helbing_2d(vel[i], acc[i], P, dt)
        end
        @inbounds for i in eachindex(pos)
            _helbing_apply_boundary_2d!(pos, vel, i, pos[i] + dt * vel[i], P)
        end

    elseif update_scheme == :asynchronous
        order = randperm(rng, length(pos))
        @inbounds for i in order
            desired[i] = P.v0 * desired_direction_Helbing_2d(i, pos, vel, P)
            acc_i = Helbing_acceleration_2d(i, pos, vel, P)
            vel[i] = update_velocity_Helbing_2d(vel[i], acc_i, P, dt)
            _helbing_apply_boundary_2d!(pos, vel, i, pos[i] + dt * vel[i], P)
        end

    else
        error("Unknown update_scheme = $(update_scheme). Use :synchronous or :asynchronous.")
    end

    return nothing
end

function apply_wall_clamp_velocity_Helbing_2d(
    p::SVector2,
    v::SVector2,
    P::HelbingParams,
)
    lo = P.radius
    hi = P.L - P.radius

    x = p.x
    y = p.y
    vx = v.x
    vy = v.y

    if x < lo
        x = lo
        vx = max(vx, 0.0)
    elseif x > hi
        x = hi
        vx = min(vx, 0.0)
    end

    if y < lo
        y = lo
        vy = max(vy, 0.0)
    elseif y > hi
        y = hi
        vy = min(vy, 0.0)
    end

    return SVector2(x, y), SVector2(vx, vy)
end
# ============================================================
# Initialisation helpers
# ============================================================

function default_heading_Helbing_2d(i::Int, P::HelbingParams)
    if P.headings !== nothing
        return safe_unit_2d(P.headings[i])
    elseif P.goals !== nothing
        # This is only a fallback; actual goal direction depends on position.
        return SVector2(1.0, 0.0)
    else
        return SVector2(1.0, 0.0)
    end
end


function initial_desired_velocity_Helbing_2d(
    i::Int,
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::HelbingParams,
)
    if P.drive_mode == :none
        return SVector2(0.0, 0.0)
    else
        return P.v0 * desired_direction_Helbing_2d(i, pos, vel, P)
    end
end


function place_in_domain_Helbing_2d(pi::SVector2, P::HelbingParams)
    if P.boundary == :periodic
        return SVector2(mod(pi.x, P.L), mod(pi.y, P.L))

    elseif P.boundary == :walls
        lo = P.radius
        hi = P.L - P.radius
        return SVector2(clamp(pi.x, lo, hi), clamp(pi.y, lo, hi))

    elseif P.boundary == :none
        return pi

    else
        error("Unknown boundary = $(P.boundary). Use :walls, :periodic, or :none.")
    end
end


# ============================================================
# Initialisation
# ============================================================

function init_state_Helbing_2d(
    P::HelbingParams;
    rng = Random.default_rng(),
    init_mode::Symbol = :left_strip,
    R0::Float64 = max(P.L / 10, 5P.radius),
    c0::SVector2 = SVector2(P.L / 4, P.L / 2),
    start_at_rest::Bool = true,
)
    if init_mode == :uniform && (R0 != max(P.L / 10, 5P.radius) || c0 != SVector2(P.L / 4, P.L / 2))
        @warn "R0 and c0 ignored for uniform initialisation over entire domain"
    end

    pos = Vector{SVector2}(undef, P.N)
    vel = Vector{SVector2}(undef, P.N)

    @inbounds for i in 1:P.N
        pi = if init_mode == :clustered
            c0 + Helbing_random_point_in_disk_2d(rng, R0)

        elseif init_mode == :uniform
            Helbing_random_point_in_square_2d(rng, P.L)

        elseif init_mode == :left_strip
            Helbing_random_point_in_left_strip_2d(rng, P.L)

        else
            error("Unknown init_mode = $(init_mode). Use :left_strip, :clustered, or :uniform.")
        end

        pos[i] = place_in_domain_Helbing_2d(pi, P)
    end

    @inbounds for i in 1:P.N
        e_i = if P.drive_mode == :goal
            P.goals === nothing &&
                error("drive_mode = :goal requires P.goals to be non-nothing.")
            safe_unit_2d(P.goals[i] - pos[i])

        elseif P.drive_mode == :heading
            P.headings === nothing &&
                error("drive_mode = :heading requires P.headings to be non-nothing.")
            safe_unit_2d(P.headings[i])

        elseif P.drive_mode == :none
            SVector2(0.0, 0.0)

        else
            error("Unknown drive_mode = $(P.drive_mode). Use :goal, :heading, or :none.")
        end

        vel[i] = start_at_rest ? SVector2(0.0, 0.0) : P.v0 * e_i
    end

    desired = [
        initial_desired_velocity_Helbing_2d(i, pos, vel, P)
        for i in 1:P.N
    ]

    return SwarmState{SVector2}(pos, vel, desired)
end


# ============================================================
# Simulation
# ============================================================

function simulate_Helbing_2d(
    simcfg::SimulationConfig,
    P::HelbingParams;
    init_mode::Symbol = :left_strip,
    dt_override::Union{Nothing, Float64} = nothing,
        R0::Float64 = max(P.L / 10, 5P.radius),
    c0::SVector2 = SVector2(P.L / 4, P.L / 2),
    start_at_rest::Bool = true,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    dt = dt_override !== nothing ? dt_override : simcfg.dt

    init_state_fn = (P; rng=Random.default_rng()) -> init_state_Helbing_2d(
        P;
        rng = rng,
        init_mode = init_mode,
        R0 = R0,
        c0 = c0,
        start_at_rest = start_at_rest,
    )

    displacement_fn = P.boundary == :periodic ? displacement : displacement_nonperiodic_2d

    simulate(
        simcfg, P;
        init_state       = init_state_fn,
        step!            = Helbing_step_2d!,
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

# # my_HelbingMolnar_Rules2d.jl

# # Helbing_random_point_in_disk_2d
# # Helbing_random_point_in_square_2d
# # Helbing_random_point_in_left_strip_2d

# # desired_direction_Helbing_2d
# # driving_acceleration_Helbing_2d
# # agent_force_Helbing_2d
# # wall_force_Helbing_2d
# # Helbing_acceleration_2d
# # update_velocity_Helbing_2d
# # update_position_Helbing_2d
# # Helbing_step_2d!

# # init_state_Helbing_2d
# # simulate_Helbing_2d

# using LinearAlgebra
# using Random
# using StaticArrays

# ============================================================
# Small helpers
# ============================================================

@inline positive_part(x::Float64) = max(0.0, x)

@inline function safe_unit_2d(v::SVector2; fallback::SVector2 = SVector2(1.0, 0.0))
    nv = norm(v)
    return nv < 1e-12 ? fallback : v / nv
end

# Non-periodic displacement from a to b.
# This has the same argument pattern as the periodic displacement helper used
# elsewhere in the ABM code: displacement(a, b, L).
@inline displacement_nonperiodic_2d(a::SVector2, b::SVector2, L::Float64) = b - a

@inline function Helbing_vector_i_to_j(pi::SVector2, pj::SVector2, P::HelbingParams)
    if P.boundary == :periodic
        return displacement(pi, pj, P.L)
    else
        return pj - pi
    end
end

@inline function clamp_speed_2d(v::SVector2, max_speed::Float64)
    s = norm(v)
    return s <= max_speed || s < 1e-12 ? v : (max_speed / s) * v
end

# ============================================================
# Initialisation helpers
# ============================================================

function Helbing_random_point_in_disk_2d(rng::AbstractRNG, R::Float64)
    θ = 2π * rand(rng)
    r = R * sqrt(rand(rng))
    return SVector2(r * cos(θ), r * sin(θ))
end

Helbing_random_point_in_square_2d(rng::AbstractRNG, L::Float64) =
    SVector2(L * rand(rng), L * rand(rng))

function Helbing_random_point_in_left_strip_2d(
    rng::AbstractRNG,
    L::Float64;
    xwidth::Float64 = 0.15L,
    ypad::Float64 = 0.10L,
)
    x = xwidth * rand(rng)
    y = ypad + (L - 2ypad) * rand(rng)
    return SVector2(x, y)
end



function simulate_Helbing_2d(simcfg::SimulationConfig, scenario::HelbingScenario; kwargs...)
    simulate_Helbing_2d(simcfg, scenario.P; init_mode = scenario.init_mode, kwargs...)
end



# # ============================================================
# # Helbing--Molnar rules
# # ============================================================

# function desired_direction_Helbing_2d(
#     i::Int,
#     pos::Vector{SVector2},
#     vel::Vector{SVector2},
#     P::HelbingParams,
# )
#     to_goal = P.goals[i] - pos[i]
#     return safe_unit_2d(to_goal; fallback = safe_unit_2d(vel[i]))
# end

# function driving_acceleration_Helbing_2d(
#     i::Int,
#     pos::Vector{SVector2},
#     vel::Vector{SVector2},
#     P::HelbingParams,
# )
#     e_i = desired_direction_Helbing_2d(i, pos, vel, P)
#     v_desired = P.v0 * e_i
#     return (v_desired - vel[i]) / P.tau
# end

# function agent_force_Helbing_2d(
#     i::Int,
#     j::Int,
#     pos::Vector{SVector2},
#     vel::Vector{SVector2},
#     P::HelbingParams,
# )
#     # d_i_to_j points from agent i to agent j.
#     d_i_to_j = Helbing_vector_i_to_j(pos[i], pos[j], P)
#     dist = norm(d_i_to_j)
#     dist < 1e-12 && return SVector2(0.0, 0.0)

#     # n_ij points from j to i, so the force on i is repulsive.
#     n_ij = -d_i_to_j / dist
#     t_ij = SVector2(-n_ij.y, n_ij.x)

#     overlap = 2P.radius - dist
#     g = positive_part(overlap)

#     social = P.A_agent * exp(overlap / P.B_agent) * n_ij
#     body = P.k_body * g * n_ij

#     # Tangential sliding/friction term. This matters only in contact.
#     Δv_t = dot(vel[j] - vel[i], t_ij)
#     friction = P.k_friction * g * Δv_t * t_ij

#     return social + body + friction
# end

# function wall_force_Helbing_2d(
#     pi::SVector2,
#     vi::SVector2,
#     P::HelbingParams,
# )
#     P.boundary == :walls || return SVector2(0.0, 0.0)

#     F = SVector2(0.0, 0.0)

#     # Each tuple is (distance to wall, inward normal).
#     walls = (
#         (pi.x,           SVector2( 1.0,  0.0)), # left
#         (P.L - pi.x,     SVector2(-1.0,  0.0)), # right
#         (pi.y,           SVector2( 0.0,  1.0)), # bottom
#         (P.L - pi.y,     SVector2( 0.0, -1.0)), # top
#     )

#     for (dist, n_iw) in walls
#         overlap = P.radius - dist
#         g = positive_part(overlap)

#         social = P.A_wall * exp(overlap / P.B_wall) * n_iw
#         body = P.k_body * g * n_iw

#         # Wall tangent. Opposes tangential sliding during contact.
#         t_iw = SVector2(-n_iw.y, n_iw.x)
#         friction = -P.k_friction * g * dot(vi, t_iw) * t_iw

#         F += social + body + friction
#     end

#     return F
# end

# function Helbing_acceleration_2d(
#     i::Int,
#     pos::Vector{SVector2},
#     vel::Vector{SVector2},
#     P::HelbingParams,
# )
#     F = SVector2(0.0, 0.0)

#     @inbounds for j in eachindex(pos)
#         j == i && continue
#         F += agent_force_Helbing_2d(i, j, pos, vel, P)
#     end

#     F += wall_force_Helbing_2d(pos[i], vel[i], P)

#     return driving_acceleration_Helbing_2d(i, pos, vel, P) + F / P.mass
# end

# function update_velocity_Helbing_2d(
#     vi::SVector2,
#     ai::SVector2,
#     P::HelbingParams,
#     dt::Float64,
# )
#     return clamp_speed_2d(vi + dt * ai, P.max_speed)
# end

# function update_position_Helbing_2d(
#     pi::SVector2,
#     vi::SVector2,
#     P::HelbingParams,
#     dt::Float64,
# )
#     p = pi + dt * vi

#     if P.boundary == :periodic
#         return SVector2(mod(p.x, P.L), mod(p.y, P.L))
#     else
#         # Keep the centre of each pedestrian inside the bounded square.
#         lo = P.radius
#         hi = P.L - P.radius
#         return SVector2(clamp(p.x, lo, hi), clamp(p.y, lo, hi))
#     end
# end

# function Helbing_step_2d!(
#     state::SwarmState{SVector2},
#     P::HelbingParams,
#     dt::Float64;
#     rng = Random.default_rng(),
# )
#     pos = state.pos
#     vel = state.vel
#     desired = state.desired

#     acc = Vector{SVector2}(undef, length(pos))

#     # Compute accelerations from the old state before updating anything.
#     @inbounds for i in eachindex(pos)
#         desired[i] = P.v0 * desired_direction_Helbing_2d(i, pos, vel, P)
#         acc[i] = Helbing_acceleration_2d(i, pos, vel, P)
#     end

#     @inbounds for i in eachindex(vel)
#         vel[i] = update_velocity_Helbing_2d(vel[i], acc[i], P, dt)
#     end

#     @inbounds for i in eachindex(pos)
#         pos[i] = update_position_Helbing_2d(pos[i], vel[i], P, dt)
#     end

#     return nothing
# end

# # ============================================================
# # Initialisation
# # ============================================================

# function init_state_Helbing_2d(
#     P::HelbingParams;
#     rng = Random.default_rng(),
#     init_mode::Symbol = :left_strip,
#     R0::Float64 = max(P.L / 10, 5P.radius),
#     c0::SVector2 = SVector2(P.L / 4, P.L / 2),
#     start_at_rest::Bool = true,
# )
#     if init_mode == :uniform && (R0 != max(P.L / 10, 5P.radius) || c0 != SVector2(P.L / 4, P.L / 2))
#         @warn "R0 and c0 ignored for uniform initialisation over entire domain"
#     end

#     pos = Vector{SVector2}(undef, P.N)
#     vel = Vector{SVector2}(undef, P.N)

#     @inbounds for i in 1:P.N
#         pi = if init_mode == :clustered
#             c0 + Helbing_random_point_in_disk_2d(rng, R0)
#         elseif init_mode == :uniform
#             Helbing_random_point_in_square_2d(rng, P.L)
#         elseif init_mode == :left_strip
#             Helbing_random_point_in_left_strip_2d(rng, P.L)
#         else
#             error("Unknown init_mode = $(init_mode). Use :left_strip, :clustered, or :uniform.")
#         end

#         if P.boundary == :periodic
#             pos[i] = SVector2(mod(pi.x, P.L), mod(pi.y, P.L))
#         else
#             lo = P.radius
#             hi = P.L - P.radius
#             pos[i] = SVector2(clamp(pi.x, lo, hi), clamp(pi.y, lo, hi))
#         end

#         e_i = safe_unit_2d(P.goals[i] - pos[i])
#         vel[i] = start_at_rest ? SVector2(0.0, 0.0) : P.v0 * e_i
#     end

#     desired = [P.v0 * safe_unit_2d(P.goals[i] - pos[i]) for i in 1:P.N]
#     return SwarmState{SVector2}(pos, vel, desired)
# end

# # ============================================================
# # Simulation
# # ============================================================

# function simulate_Helbing_2d(
#     simcfg::SimulationConfig,
#     P::HelbingParams;
#     init_mode::Symbol = :left_strip,
#     R0::Float64 = max(P.L / 10, 5P.radius),
#     c0::SVector2 = SVector2(P.L / 4, P.L / 2),
#     start_at_rest::Bool = true,
#     collect_order::Bool = true,
#     collect_history::Bool = true,
# )
#     init_state_fn = (P; rng=Random.default_rng()) -> init_state_Helbing_2d(
#         P;
#         rng = rng,
#         init_mode = init_mode,
#         R0 = R0,
#         c0 = c0,
#         start_at_rest = start_at_rest,
#     )

#     displacement_fn = P.boundary == :periodic ? displacement : displacement_nonperiodic_2d

#     simulate(
#         simcfg, P;
#         init_state       = init_state_fn,
#         step!            = Helbing_step_2d!,
#         get_pos          = state -> state.pos,
#         get_vel          = state -> state.vel,
#         domain_size      = (P, state) -> P.L,
#         displacement     = displacement_fn,
#         order_parameters = order_parameters,
#         collect_order    = collect_order,
#         collect_history  = collect_history,
#     )
# end

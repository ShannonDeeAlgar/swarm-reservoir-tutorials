# my_Couzin_Rules2d.jl
#
# 2D Couzin step logic: is_visible, random_point_in_disk_2d,
# random_point_in_square_2d, desired_direction_2d, update_velocity_2d,
# update_position_2d, Couzin_step_2d!, init_state_Couzin_2d, simulate_Couzin_2d.
# Mirrors my_Couzin_Rules3d.jl's naming (which already uses explicit _3d
# suffixes).

using LinearAlgebra
using Random
using StaticArrays

@inline function is_visible(vi_u::SVector2, d_u::SVector2, blind_half_angle::Float64)
    blind_half_angle <= 0 && return true
    blind_half_angle >= π && return false
    return dot(vi_u, d_u) >= -cos(blind_half_angle)
end

# Non-periodic displacement -- for the unbounded Couzin (2002) reproduction,
# mirroring displacement_nonperiodic_3d in my_Couzin_Rules3d.jl.
@inline displacement_nonperiodic_2d(a::SVector2, b::SVector2, ::Float64) = b - a

# ============================================================
# Initialisation helpers
# ============================================================
function random_point_in_disk_2d(rng::AbstractRNG, R::Float64)
    θ = 2π * rand(rng)
    r = R * sqrt(rand(rng))
    return SVector2(r * cos(θ), r * sin(θ))
end

random_point_in_square_2d(rng::AbstractRNG, L::Float64) =
    SVector2(L * rand(rng), L * rand(rng))



# ============================================================
# Couzin rules
# ============================================================
function desired_direction_2d(i::Int,
                           pos::Vector{SVector2},
                           vel::Vector{SVector2},
                           P::CouzinParams;
                           displacement_fn::Function = displacement,
                           combine_rule::Symbol = :couzin2002)
    pi   = pos[i]
    vi_u = unit(vel[i])

    rep = SVector2(0.0, 0.0)
    ori = SVector2(0.0, 0.0)
    att = SVector2(0.0, 0.0)

    n_rep = 0
    n_ori = 0
    n_att = 0

    @inbounds for j in eachindex(pos)
        j == i && continue

        d = displacement_fn(pi, pos[j], P.L)
        r = norm(d)
        r == 0 && continue

        if r <= P.Zr
            rep -= d / r
            n_rep += 1
            continue
        end

        d_u = d / r
        is_visible(vi_u, d_u, P.blind_half_angle) || continue

        if r <= P.Zo
            ori += unit(vel[j])
            n_ori += 1
        elseif r <= P.Za
            att += d_u
            n_att += 1
        end
    end

    if n_rep > 0
        return unit(rep)
    else
        dir =
            if n_ori > 0 && n_att > 0
                # Couzin (2002), eqns (2)-(3): d_o = Σ v̂_j (orientation zone),
                # d_a = Σ r̂_ij (attraction zone) -- each a raw, un-normalized sum
                # of unit vectors, not itself renormalized to unit length. When
                # both zones are occupied: "d_i(t+τ) = 1/2[d_o(t+τ)+d_a(t+τ)]"
                # -- a plain sum (the 1/2 doesn't affect direction, since the
                # result is unit-normalized below regardless). This is the
                # paper's actual, published rule -- verified directly against
                # the primary source (Couzin et al. 2002, J. Theor. Biol. 218).
                if combine_rule == :couzin2002
                    ori + att
                # NOT in the paper. Separately unit-normalizes each zone's
                # contribution before summing, so orientation and attraction
                # are weighted 50/50 by construction regardless of how many
                # neighbours are in each zone or how coherent their headings
                # are. Offered only as a comparison/exploration variant.
                elseif combine_rule == :normalized_zones
                    unit(ori) + unit(att)
                else
                    error("Unknown combine_rule = $combine_rule. Use :couzin2002 or :normalized_zones.")
                end
            elseif n_ori > 0
                ori
            elseif n_att > 0
                att
            else
                vi_u
            end

        return norm(dir) < 1e-12 ? vi_u : unit(dir)
    end
end

function update_velocity_2d(
    vi::SVector2,
    desired::SVector2,
    P::CouzinParams,
    dt::Float64,
    rng::AbstractRNG
)
    vi_u      = unit(vi)
    desired_u = unit(desired)

    η = sample_noise_angle(P.noise, dt, rng)
    if η != 0.0
        desired_u = rot2(desired_u, η)
    end

    θ    = signed_angle(vi_u, desired_u)
    maxΔ = P.max_turn_rate * dt
    Δ    = clamp(θ, -maxΔ, maxΔ)

    new_u = rot2(vi_u, Δ)

    return P.speed * unit(new_u)

end

function update_position_2d(pos::SVector2, vel::SVector2, P::CouzinParams, dt::Float64;
                            periodic::Bool = true)
    p = pos + dt * vel
    periodic || return p
    return SVector2(mod(p.x, P.L), mod(p.y, P.L))
end

function Couzin_step_2d!(state::SwarmState{SVector2}, P::CouzinParams, dt::Float64;
    rng = Random.default_rng(),
    update_scheme::Symbol = :synchronous,
    periodic::Bool = true,
    combine_rule::Symbol = :couzin2002,
)
    pos = state.pos
    vel = state.vel
    desired = state.desired

    if update_scheme == :synchronous
        @inbounds for i in eachindex(pos)
            desired[i] = desired_direction_2d(i, pos, vel, P;
                displacement_fn = periodic ? displacement : displacement_nonperiodic_2d,
                combine_rule = combine_rule)
        end
        @inbounds for i in eachindex(vel)
            vel[i] = update_velocity_2d(vel[i], desired[i], P, dt, rng)
        end
        @inbounds for i in eachindex(pos)
            pos[i] = update_position_2d(pos[i], vel[i], P, dt; periodic=periodic)
        end

    elseif update_scheme == :asynchronous
        order = randperm(rng, length(pos))
        @inbounds for i in order
            d = desired_direction_2d(i, pos, vel, P;
                displacement_fn = periodic ? displacement : displacement_nonperiodic_2d,
                combine_rule = combine_rule)
            desired[i] = d
            vel[i] = update_velocity_2d(vel[i], d, P, dt, rng)
            pos[i] = update_position_2d(pos[i], vel[i], P, dt; periodic=periodic)
        end

    else
        error("Unknown update_scheme = $(update_scheme). Use :synchronous or :asynchronous.")
    end

    return nothing
end

# Does every agent detect at least one neighbour (repulsion or attraction zone,
# respecting the blind angle)? Used at initialisation for the unbounded
# reproduction, where there is no box to keep a stray agent's isolation
# transient. Mirrors all_have_detectable_neighbour_3d.
function all_have_detectable_neighbour_2d(
    pos::Vector{SVector2}, vel::Vector{SVector2}, P::CouzinParams;
    periodic::Bool=true,
)
    displacement_fn = periodic ? displacement : displacement_nonperiodic_2d
    @inbounds for i in eachindex(pos)
        vi_u = unit(vel[i])
        detected = false
        for j in eachindex(pos)
            i == j && continue
            d = displacement_fn(pos[i], pos[j], P.L)
            r = norm(d)
            r == 0 && continue
            if r <= P.Zr || (r <= P.Za && is_visible(vi_u, d/r, P.blind_half_angle))
                detected = true
                break
            end
        end
        detected || return false
    end
    return true
end

# ============================================================
# Initialisation
# ============================================================

function init_state_Couzin_2d(
    P::CouzinParams;
    rng = Random.default_rng(),
    init_mode::Symbol = :clustered,
    R0::Float64 = max(P.Za, 5.0),
    c0::SVector2 = SVector2(P.L / 2, P.L / 2),
    periodic::Bool = true,
    ensure_detectable_neighbour::Bool = false,
    max_init_attempts::Int = 1000,
)
    if init_mode == :uniform && (R0 != max(P.Za, 5.0) || c0 != SVector2(P.L / 2, P.L / 2))
        @warn "R0 and c0 ignored for uniform initialisation over entire domain"
    end

    pos = Vector{SVector2}(undef, P.N)
    vel = Vector{SVector2}(undef, P.N)

    for attempt in 1:max_init_attempts
        @inbounds for i in 1:P.N
            pi = if init_mode == :clustered
                c0 + random_point_in_disk_2d(rng, R0)
            elseif init_mode == :uniform
                random_point_in_square_2d(rng, P.L)
            else
                error("Unknown init_mode = $init_mode. Use :clustered or :uniform.")
            end

            pos[i] = periodic ?
                SVector2(mod(pi.x, P.L), mod(pi.y, P.L)) : pi

            θ = 2π * rand(rng)
            vel[i] = P.speed * SVector2(cos(θ), sin(θ))
        end

        if !ensure_detectable_neighbour ||
           all_have_detectable_neighbour_2d(pos, vel, P; periodic=periodic)
            desired = copy(vel)
            return SwarmState{SVector2}(pos, vel, desired)
        end
    end

    error("Could not initialise a Couzin group in which every agent detects " *
          "at least one neighbour after $max_init_attempts attempts. " *
          "Increase R0 or max_init_attempts.")
end

# ============================================================
# Simulation
# ============================================================
function simulate_Couzin_2d(
    simcfg::SimulationConfig,
    P::CouzinParams;
    init_mode::Symbol = :clustered,
    dt_override::Union{Nothing, Float64} = nothing,

    R0::Float64 = sqrt(P.N) * (P.Zr + P.Zo) / 2, # 2D: area of disk = π R0² = N * mean_area_per_agent. Set mean_area_per_agent = π * ((Zr + Zo)/2)²  → R0 = sqrt(N) * (Zr + Zo)/2
    c0::SVector2 = SVector2(P.L / 2, P.L / 2),
    periodic::Bool = true,
    ensure_detectable_neighbour::Bool = false,
    combine_rule::Symbol = :couzin2002,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    dt = dt_override !== nothing ? dt_override : simcfg.dt

    init_state_fn = (P; rng=Random.default_rng()) -> init_state_Couzin_2d(
        P;
        rng = rng,
        init_mode = init_mode,
        R0 = R0,
        c0 = c0,
        periodic = periodic,
        ensure_detectable_neighbour = ensure_detectable_neighbour,
    )

    step_fn! = (state, P, dt; rng=Random.default_rng(), update_scheme=:synchronous) ->
        Couzin_step_2d!(state, P, dt;
            rng=rng, update_scheme=update_scheme, periodic=periodic, combine_rule=combine_rule)

    displacement_fn = periodic ? displacement : displacement_nonperiodic_2d
    order_fn = periodic ? order_parameters : order_parameters_nonperiodic_2d

    simulate(
        simcfg, P;
        dt                = dt,
        init_state        = init_state_fn,
        step!             = step_fn!,
        get_pos           = state -> state.pos,
        get_vel           = state -> state.vel,
        domain_size       = (P, state) -> P.L,
        displacement      = displacement_fn,
        order_parameters  = order_fn,
        collect_order     = collect_order,
        collect_history   = collect_history,
        show_progress     = show_progress,
    )
end

# Couzin et al. (2002) used an unbounded domain. The generic 2D order-parameter
# routine (order_parameters_2d in my_ABM_analysis.jl) assumes a periodic centre
# of mass, so the publication reproduction uses the ordinary centroid instead.
# Mirrors order_parameters_nonperiodic_3d in my_Couzin_Rules3d.jl. `rotation`
# already takes abs() after averaging, matching Couzin's |m_group| and the
# fixed generic 2D formula.
function order_parameters_nonperiodic_2d(
    pos::Vector{SVector2}, vel::Vector{SVector2}, L::Float64; displacement
)
    N = length(pos)
    N == 0 && return (0.0, 0.0, 0.0, 0.0)

    mx = 0.0
    my = 0.0
    @inbounds for i in 1:N
        v̂ = unit(vel[i])
        mx += v̂.x
        my += v̂.y
    end
    polarisation = sqrt((mx / N)^2 + (my / N)^2)

    centre = sum(pos) / N

    rot_sum = 0.0
    dil_sum = 0.0
    ang_sum = 0.0
    count = 0

    # See order_parameters_2d (ABM/my_ABM_analysis.jl) for the full rationale:
    # `rotation` normalizes r to r̂ (satisfies Couzin (2002)'s stated m_group
    # in [0,1], though the paper's eqn 5 as typeset shows no hat on r_ic);
    # `abs_angular_momentum` keeps raw r as a cross-check, and both take the
    # magnitude *after* summing over agents so opposite-handedness rotation
    # can cancel, matching eqn 5's |sum(...)| structure.
    @inbounds for i in 1:N
        r = pos[i] - centre
        ρ = norm(r)
        v = vel[i]
        vmag = norm(v)

        if ρ ≥ 1e-9 && vmag ≥ 1e-12
            r̂ = r / ρ
            t̂ = SVector2(-r̂.y, r̂.x)
            v̂ = unit(v)

            rot_sum += dot(v̂, t̂)
            dil_sum += dot(v̂, r̂)
            ang_sum += r.x * v̂.y - r.y * v̂.x
            count += 1
        end
    end

    rotation = count > 0 ? abs(rot_sum / count) : 0.0
    dilation = count > 0 ? dil_sum / count : 0.0
    abs_angular_momentum = count > 0 ? abs(ang_sum / count) : 0.0

    return (dilation, rotation, polarisation, abs_angular_momentum)
end

"""Couzin et al. (2002) two-dimensional simulation.

By default this uses the paper's unbounded space. Set `periodic=true` to retain
the codebase's periodic square while keeping the published interaction rules and
parameters. `P` selects a point in the paper's `(Delta r_o, Delta r_a)` plane;
no automatic 2D preset corrections are applied. Mirrors simulate_Couzin2002_3d
in my_Couzin_Rules3d.jl, including the `min(Zr+Zo, Za)` spawn-radius fix (see
that docstring): near `Δr_a≈0`, `Za` can be smaller than `Zr+Zo`, and
`all_have_detectable_neighbour_2d`'s detection radius is `Za`, not `Zr+Zo`.
"""
function simulate_Couzin2002_2d(
    simcfg::SimulationConfig, P::CouzinParams;
    R0::Float64 = sqrt(Float64(P.N)) * min(P.Zr + P.Zo, P.Za) / 2,
    c0::SVector2 = SVector2(0.0, 0.0),
    periodic::Bool = false,
    ensure_detectable_neighbour::Bool = true,
    combine_rule::Symbol = :couzin2002,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    return simulate_Couzin_2d(simcfg, P;
        init_mode=:clustered, R0=R0, c0=c0, periodic=periodic,
        ensure_detectable_neighbour=ensure_detectable_neighbour,
        combine_rule=combine_rule,
        collect_order=collect_order, collect_history=collect_history,
        show_progress=show_progress)
end

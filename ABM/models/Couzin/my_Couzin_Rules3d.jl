# my_Couzin_Rules3d.jl

# is_visible_3d

# unit3
# random_unit_vector_3d
# orthonormal_perp
# rodrigues_rotate
# rotate_toward_3d
# perturb_direction_3d

# desired_direction_3d
# update_velocity_3d
# update_velocity_3d_instant
# update_position_3d
# Couzin_step_3d!

# random_point_in_cube_3d
# random_point_in_ball_3d

# init_state_Couzin_3d
# simulate_Couzin_3d

using LinearAlgebra
using Random
using StaticArrays

@inline function is_visible_3d(vi_u::SVector3, d_u::SVector3, blind_half_angle::Float64)
    blind_half_angle <= 0 && return true
    blind_half_angle >= π && return false
    return dot(vi_u, d_u) >= -cos(blind_half_angle)
end

# ============================================================
# 3D vector helpers
# ============================================================
@inline function unit3(v::SVector3; atol::Float64=1e-12)
    nv = norm(v)
    nv < atol && return SVector3(1.0, 0.0, 0.0)
    return v / nv
end

function random_unit_vector_3d(rng::AbstractRNG)
    z  = 2rand(rng) - 1
    ϕ  = 2π * rand(rng)
    ρ  = sqrt(max(0.0, 1 - z^2))
    return SVector3(ρ * cos(ϕ), ρ * sin(ϕ), z)
end

function orthonormal_perp(u::SVector3)
    ref = abs(u[1]) < 0.9 ? SVector3(1.0, 0.0, 0.0) : SVector3(0.0, 1.0, 0.0)
    v = cross(u, ref)
    nv = norm(v)

    if nv < 1e-12
        ref = SVector3(0.0, 0.0, 1.0)
        v = cross(u, ref)
        nv = norm(v)
    end

    return v / nv
end

# Rodrigues' rotation formula: rotates v by angle θ about unit axis k.
function rodrigues_rotate(v::SVector3, k::SVector3, θ::Float64)
    c = cos(θ)
    s = sin(θ)
    return c * v + s * cross(k, v) + (1 - c) * dot(k, v) * k
end

function rotate_toward_3d(u::SVector3, d::SVector3, maxΔ::Float64)
    û = unit3(u)
    d̂ = unit3(d)

    c = clamp(dot(û, d̂), -1.0, 1.0)
    θ = acos(c)

    θ < 1e-12 && return û

    Δ = min(θ, maxΔ)

    ax = cross(û, d̂)
    nax = norm(ax)

    if nax < 1e-12
        if c > 0
            return û
        else
            k = orthonormal_perp(û)
            return unit3(rodrigues_rotate(û, k, Δ))
        end
    end

    k = ax / nax
    return unit3(rodrigues_rotate(û, k, Δ))
end

function perturb_direction_3d(u::SVector3, η::Float64, rng::AbstractRNG)
    abs(η) < 1e-14 && return unit3(u)

    û = unit3(u)
    e1 = orthonormal_perp(û)
    e2 = cross(û, e1)

    ϕ = 2π * rand(rng)
    axis = cos(ϕ) * e1 + sin(ϕ) * e2

    return unit3(rodrigues_rotate(û, axis, η))
end

# ============================================================
# Couzin rules (3D)
# ============================================================
function desired_direction_3d(i::Int,
                              pos::Vector{SVector3},
                              vel::Vector{SVector3},
                              P::CouzinParams;
                              displacement_fn::Function = displacement,
                              combine_rule::Symbol = :couzin2002)
    pi   = pos[i]
    vi_u = unit3(vel[i])

    rep = SVector3(0.0, 0.0, 0.0)
    ori = SVector3(0.0, 0.0, 0.0)
    att = SVector3(0.0, 0.0, 0.0)

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
        is_visible_3d(vi_u, d_u, P.blind_half_angle) || continue

        if r <= P.Zo
            ori += unit3(vel[j])
            n_ori += 1
        elseif r <= P.Za
            att += d_u
            n_att += 1
        end
    end

    if n_rep > 0
        return unit3(rep)
    else
        dir =
            if n_ori > 0 && n_att > 0
                # Couzin (2002), eqns (2)-(3): raw, un-normalized sums of unit
                # vectors, combined as d_i = 1/2[d_o+d_a] when both zones are
                # occupied -- verified directly against the primary source
                # (Couzin et al. 2002, J. Theor. Biol. 218). Mirrors the 2D
                # implementation's combine_rule in my_Couzin_Rules2d.jl.
                if combine_rule == :couzin2002
                    ori + att
                # NOT in the paper -- separately unit-normalizes each zone
                # before summing, weighting orientation/attraction 50/50
                # regardless of neighbour count or coherence. Comparison
                # variant only.
                elseif combine_rule == :normalized_zones
                    unit3(ori) + unit3(att)
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

        return norm(dir) < 1e-12 ? vi_u : unit3(dir)
    end
end

function update_velocity_3d(
    vi::SVector3,
    desired::SVector3,
    P::CouzinParams,
    dt::Float64,
    rng::AbstractRNG
)
    vi_u      = unit3(vi)
    desired_u = unit3(desired)

    η = sample_noise_angle(P.noise, dt, rng)
    if η != 0.0
        desired_u = perturb_direction_3d(desired_u, η, rng)
    end

    maxΔ  = P.max_turn_rate * dt
    new_u = rotate_toward_3d(vi_u, desired_u, maxΔ)

    return P.speed * new_u
end

function update_velocity_3d_instant(
    vi::SVector3,
    desired::SVector3,
    P::CouzinParams,
    dt::Float64,
    rng::AbstractRNG
)
    desired_u = unit3(desired)

    η = sample_noise_angle(P.noise, dt, rng)
    if η != 0.0
        desired_u = perturb_direction_3d(desired_u, η, rng)
    end

    return P.speed * desired_u
end

function update_position_3d(pos::SVector3, vel::SVector3, P::CouzinParams, dt::Float64;
                            periodic::Bool = true)
    p = pos + dt * vel
    periodic || return p
    return SVector3(mod(p[1], P.L), mod(p[2], P.L), mod(p[3], P.L))
end

function Couzin_step_3d!(state::SwarmState{SVector3}, P::CouzinParams, dt::Float64;
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
            desired[i] = desired_direction_3d(i, pos, vel, P;
                displacement_fn = periodic ? displacement : displacement_nonperiodic_3d,
                combine_rule = combine_rule)
        end
        @inbounds for i in eachindex(vel)
            vel[i] = update_velocity_3d(vel[i], desired[i], P, dt, rng)
        end
        @inbounds for i in eachindex(pos)
            pos[i] = update_position_3d(pos[i], vel[i], P, dt; periodic=periodic)
        end

    elseif update_scheme == :asynchronous
        order = randperm(rng, length(pos))
        @inbounds for i in order
            d = desired_direction_3d(i, pos, vel, P;
                displacement_fn = periodic ? displacement : displacement_nonperiodic_3d,
                combine_rule = combine_rule)
            desired[i] = d
            vel[i] = update_velocity_3d(vel[i], d, P, dt, rng)
            pos[i] = update_position_3d(pos[i], vel[i], P, dt; periodic=periodic)
        end

    else
        error("Unknown update_scheme = $(update_scheme). Use :synchronous or :asynchronous.")
    end

    return nothing
end

# ============================================================
# Initialisation (3D)
# ============================================================
random_point_in_cube_3d(rng::AbstractRNG, L::Float64) =
    SVector3(L * rand(rng), L * rand(rng), L * rand(rng))

@inline displacement_nonperiodic_3d(a::SVector3, b::SVector3, ::Float64) = b - a

function random_point_in_ball_3d(rng::AbstractRNG, R::Float64)
    u = random_unit_vector_3d(rng)
    r = R * rand(rng)^(1/3)
    return r * u
end

function all_have_detectable_neighbour_3d(
    pos::Vector{SVector3}, vel::Vector{SVector3}, P::CouzinParams;
    periodic::Bool=true,
)
    displacement_fn = periodic ? displacement : displacement_nonperiodic_3d
    @inbounds for i in eachindex(pos)
        vi_u = unit3(vel[i])
        detected = false
        for j in eachindex(pos)
            i == j && continue
            d = displacement_fn(pos[i], pos[j], P.L)
            r = norm(d)
            r == 0 && continue
            if r <= P.Zr || (r <= P.Za && is_visible_3d(vi_u, d/r, P.blind_half_angle))
                detected = true
                break
            end
        end
        detected || return false
    end
    return true
end

function init_state_Couzin_3d(
    P::CouzinParams;
    rng = Random.default_rng(),
    init_mode::Symbol = :clustered,
    R0::Float64 = cbrt(Float64(P.N)) * (P.Zr + P.Zo) / 2,
    c0::SVector3 = SVector3(P.L / 2, P.L / 2, P.L / 2),
    periodic::Bool = true,
    ensure_detectable_neighbour::Bool = false,
    max_init_attempts::Int = 1000,
)
    pos = Vector{SVector3}(undef, P.N)
    vel = Vector{SVector3}(undef, P.N)

    for attempt in 1:max_init_attempts
        @inbounds for i in 1:P.N
            pi = if init_mode == :clustered
                c0 + random_point_in_ball_3d(rng, R0)
            elseif init_mode == :uniform
                random_point_in_cube_3d(rng, P.L)
            else
                error("Unknown init_mode = $init_mode. Use :clustered or :uniform.")
            end

            pos[i] = periodic ?
                SVector3(mod(pi.x, P.L), mod(pi.y, P.L), mod(pi.z, P.L)) : pi
            vel[i] = P.speed * random_unit_vector_3d(rng)
        end

        if !ensure_detectable_neighbour ||
           all_have_detectable_neighbour_3d(pos, vel, P; periodic=periodic)
            desired = copy(vel)
            return SwarmState{SVector3}(pos, vel, desired)
        end
    end

    error("Could not initialise a Couzin group in which every agent detects " *
          "at least one neighbour after $max_init_attempts attempts. " *
          "Increase R0 or max_init_attempts.")
end

# ============================================================
# Simulation (3D)
# ============================================================
function simulate_Couzin_3d(
    simcfg::SimulationConfig,
    P::CouzinParams;
    init_mode::Symbol = :clustered,
    dt_override::Union{Nothing, Float64} = nothing,
    R0::Float64 = cbrt(P.N) * (P.Zr + P.Zo) / 2,
    c0::SVector3 = SVector3(P.L / 2, P.L / 2, P.L / 2),
    periodic::Bool = true,
    ensure_detectable_neighbour::Bool = false,
    combine_rule::Symbol = :couzin2002,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    dt = dt_override !== nothing ? dt_override : simcfg.dt

    init_state_fn = (P; rng=Random.default_rng()) -> init_state_Couzin_3d(
        P;
        rng = rng,
        init_mode = init_mode,
        R0 = R0,
        c0 = c0,
        periodic = periodic,
        ensure_detectable_neighbour = ensure_detectable_neighbour,
    )

    step_fn! = (state, P, dt; rng=Random.default_rng(), update_scheme=:synchronous) ->
        Couzin_step_3d!(state, P, dt;
            rng=rng, update_scheme=update_scheme, periodic=periodic, combine_rule=combine_rule)

    displacement_fn = periodic ? displacement : displacement_nonperiodic_3d
    order_fn = periodic ? order_parameters : order_parameters_nonperiodic_3d

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

# Couzin et al. (2002) used an unbounded 3D domain.  The generic 3D order
# parameter routine assumes a periodic centre of mass, so the publication
# reproduction uses the ordinary centroid instead.
function order_parameters_nonperiodic_3d(
    pos::Vector{SVector3}, vel::Vector{SVector3}, L::Float64; displacement
)
    N = length(pos)
    N == 0 && return (0.0, 0.0, 0.0, 0.0)

    vhat = unit3.(vel)
    polarisation = norm(sum(vhat)) / N
    centre = sum(pos) / N

    angular = SVector3(0.0, 0.0, 0.0)
    dilation = 0.0
    count = 0

    # Raw-r (not r̂) accumulator, see order_parameters_2d (ABM/my_ABM_analysis.jl)
    # for the full rationale on the r vs r̂ ambiguity and why this is summed
    # as a vector (cancels opposite-handedness rotation) before norm().
    abs_angular_vec = SVector3(0.0, 0.0, 0.0)

    @inbounds for i in eachindex(pos)
        r = pos[i] - centre
        rho = norm(r)
        vmag = norm(vel[i])
        (rho < 1e-9 || vmag < 1e-12) && continue
        rhat = r / rho
        angular += cross(rhat, vhat[i])
        dilation += dot(vhat[i], rhat)
        abs_angular_vec += cross(r, vhat[i])
        count += 1
    end

    rotation = count > 0 ? norm(angular / count) : 0.0
    dilation_mean = count > 0 ? dilation / count : 0.0
    abs_angular = count > 0 ? norm(abs_angular_vec / count) : 0.0
    return (dilation_mean, rotation, polarisation, abs_angular)
end

"""Couzin et al. (2002) three-dimensional simulation.

By default this uses the paper's unbounded space. Set `periodic=true` to retain
the codebase's periodic cube while keeping the published interaction rules and
parameters. `P` selects a point in the paper's `(Delta r_o, Delta r_a)` plane;
no automatic 3D preset corrections are applied.

`R0` (initial spawn radius) uses `min(Zr+Zo, Za)`, not just `Zr+Zo`: near
`Δr_a≈0` (small attraction zone), `Za` can be *smaller* than `Zr+Zo`, and
`all_have_detectable_neighbour_3d`'s detection radius is `Za`. Sizing R0 from
`Zr+Zo` alone can spawn agents too sparsely to detect each other within `Za`,
causing `init_state_Couzin_3d` to exhaust `max_init_attempts` -- this is
Couzin's own region (e) (low `Δr_o`/`Δr_a`, paper: ">50% chance of
fragmenting"), so failures there reflect the model, not just initialisation,
but the *spawn* should still be dense enough that a well-formed group has a
fair chance to be found. Confirmed this doesn't change R0 for any of the four
`paper_regimes_3d` points above (Za is always well above Zr+Zo there).
"""
function simulate_Couzin2002_3d(
    simcfg::SimulationConfig, P::CouzinParams;
    R0::Float64 = cbrt(Float64(P.N)) * min(P.Zr + P.Zo, P.Za) / 2,
    c0::SVector3 = SVector3(0.0, 0.0, 0.0),
    periodic::Bool = false,
    ensure_detectable_neighbour::Bool = true,
    combine_rule::Symbol = :couzin2002,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    return simulate_Couzin_3d(simcfg, P;
        init_mode=:clustered, R0=R0, c0=c0, periodic=periodic,
        ensure_detectable_neighbour=ensure_detectable_neighbour,
        combine_rule=combine_rule,
        collect_order=collect_order, collect_history=collect_history,
        show_progress=show_progress)
end

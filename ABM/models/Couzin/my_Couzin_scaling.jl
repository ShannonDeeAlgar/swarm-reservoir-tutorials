# my_Couzin_scaling.jl
#
# Nondimensionalisation + scaling utilities: warn_if_step_too_large,
# nondimensionalise!, audit_nondim, scale_time!, scale_space!,
# scale_swarm_dynamics!.
#
# Despite the trailing `!` (kept for consistent naming with the rest of the
# codebase), none of these mutate their `CouzinParams` argument in place --
# `CouzinParams` is immutable. Each returns a *new* (P_new, dt_new) pair:
#   nondimensionalise!(P, dt)         -> (P_nd, dt_nd)
#   scale_time!(P, dt; f=...)         -> (P_new, dt_new)
#   scale_space!(P, dt; s=...)        -> (P_new, dt_new)
#   scale_swarm_dynamics!(P, dt; g=...) -> (P_new, dt_new)
# ============================================================

# --------------------------
# Step-size warning (shared)
# --------------------------
function warn_if_step_too_large(P::CouzinParams, dt_eff::Float64;
                                thresh_warn::Float64 = 0.2,
                                thresh_fail::Float64 = 0.5,
                                context::String = "")
    ε = (P.speed * dt_eff) / P.Zr

    if ε > thresh_fail
        @warn """
        Couzin step size too large$(isempty(context) ? "" : " ($context)"):
            dt_eff = $(dt_eff)
            ε = (speed · dt_eff) / Zr = $(round(ε, digits=3)) > $(thresh_fail)

        Neighbour topology will change abruptly; milling is very likely to fail.
        Reduce dt, or use substepping / record with stride.
        """
    elseif ε > thresh_warn
        @warn """
        Couzin step size borderline$(isempty(context) ? "" : " ($context)"):
            dt_eff = $(dt_eff)
            ε = (speed · dt_eff) / Zr = $(round(ε, digits=3)) > $(thresh_warn)

        Behaviour may shift (milling may weaken/disappear).
        """
    end

    return ε
end

# --------------------------
# 1) Nondimensionalise params + dt (single function)
# --------------------------
"""
    nondimensionalise!(P::CouzinParams, dt_phys::Float64) -> (P_nd, dt_nd)

Uses length unit Zr and time unit Zr/speed.

- Zr -> 1
- Zo, Za, L -> divide by Zr
- speed -> 1
- max_turn_rate -> max_turn_rate * (Zr/speed)
Noise:
- Diffusion: Dθ -> Dθ * (Zr/speed)
- WrappedStepGaussian: unchanged
Angles (blind_half_angle) unchanged.
Also returns dt_nd = (speed/Zr) * dt_phys.

If P.is_nondim==true, returns (P, dt_phys) unchanged.
"""
function nondimensionalise!(P::CouzinParams, dt_phys::Float64)
    if P.is_nondim
        return P, dt_phys
    end

    @assert P.Zr > 0 "Zr must be > 0"
    @assert P.speed > 0 "speed must be > 0"
    @assert dt_phys > 0 "dt must be > 0"

    ℓ = P.Zr
    τ = P.Zr / P.speed

    # dt in nondim units
    dt_nd = (P.speed / P.Zr) * dt_phys

    # noise
    noise_new = P.noise
    if noise_new.mode == Diffusion
        noise_new = update_CouzinNoise(noise_new; Dθ = noise_new.Dθ * τ)
    end

    P_nd = update_CouzinParams(
        P;
        Zr = 1.0,
        Zo = P.Zo / ℓ,
        Za = P.Za / ℓ,
        L  = P.L  / ℓ,

        speed = 1.0,
        max_turn_rate = P.max_turn_rate * τ,
        noise = noise_new,

        blind_half_angle = P.blind_half_angle,

        is_nondim = true,
        Zr_ref = P.Zr,
        v_ref  = P.speed
    )

    return P_nd, dt_nd
end

function audit_nondim(P::CouzinParams, dt::Float64; atol=1e-8, rtol=1e-6)
    @assert P.is_nondim "audit_nondim called with is_nondim = false"

    @assert isapprox(P.Zr, 1.0; atol=atol, rtol=rtol) "Zr ≠ 1"
    @assert isapprox(P.speed, 1.0; atol=atol, rtol=rtol) "speed ≠ 1"

    ϵ = P.speed * dt / P.Zr
    @assert isapprox(ϵ, dt; atol=atol, rtol=rtol) "dt not dimensionless"

    Δθ = P.max_turn_rate * dt
    @assert Δθ < π "turning per step too large"

    if P.noise.mode == Diffusion
        σ = sqrt(P.noise.Dθ * dt)
        @assert σ ≥ 0 "invalid diffusion noise"
    end

    return true
end



# --------------------------
# 2) Time scaling params + dt (single function)
# --------------------------
"""
    scale_time!(P::CouzinParams, dt::Float64; f=1.0, warn_step=true, ...) -> (P_new, dt_new)

Fast-forward / slow-mo of the discrete map:
- dt -> f*dt
- max_turn_rate -> max_turn_rate / f
- Diffusion Dθ -> Dθ / f
- WrappedStepGaussian: unchanged

Warns if ε = (speed*dt_new)/Zr is too large.
"""
function scale_time!(P::CouzinParams, dt::Float64;
                     f::Float64 = 1.0,
                     warn_step::Bool = true,
                     thresh_warn::Float64 = 0.2,
                     thresh_fail::Float64 = 0.5)
    @assert f > 0
    @assert dt > 0

    dt_new = dt * f

    if warn_step
        warn_if_step_too_large(P, dt_new;
                               thresh_warn=thresh_warn, thresh_fail=thresh_fail,
                               context="after time scaling (dt→f·dt)")
    end

    noise_new = P.noise
    if noise_new.mode == Diffusion
        noise_new = CouzinNoise(mode=Diffusion, s=noise_new.s, Dθ=noise_new.Dθ / f)
    end

    P_new = update_CouzinParams(
        P;
        max_turn_rate = P.max_turn_rate / f,
        noise = noise_new
    )

    return P_new, dt_new
end

##Better than scale_time...
# function simulate_Couzin_strided(simcfg::SimulationConfig,
#                                  P::CouzinParams;
#                                  stride::Int = 1,
#                                  compute_order::Bool=true)

#     @assert stride ≥ 1

#     # run full simulation
#     full = simulate_Couzin(simcfg, P;
#                            compute_order=compute_order,
#                            store_hist=true)

#     keep = 1:stride:length(full.t)

#     return SimulationOutput(
#         full.simcfg,
#         full.params,
#         full.focal,

#         full.t[keep],

#         compute_order ? full.dilation[keep]     : Float32[],
#         compute_order ? full.rotation[keep]     : Float32[],
#         compute_order ? full.polarisation[keep] : Float32[],

#         full.pos_hist[keep],
#         full.vel_hist[keep]
#     )
# end

# --------------------------
# 3) Space scaling params + dt (single function)
# --------------------------
"""
    scale_space!(P::CouzinParams, dt::Float64; s=1.0, preserve_density=true, preserve_step_stats=true, ...) -> (P_new, dt_new)

Space rescaling by factor s with kinematic similarity:
- lengths: Zr,Zo,Za,L -> s*(...)
- dt -> s*dt (keeps speed*dt/Zr invariant)
- preserve_density: N -> round(N*s^2)
- preserve_step_stats: max_turn_rate -> max_turn_rate / s
  and Diffusion Dθ -> Dθ / s (so per-step variance invariant)

Warns if ε = (speed*dt_new)/Zr is too large.
"""
function scale_space!(P::CouzinParams, dt::Float64;
                      s::Float64 = 1.0,
                      preserve_density::Bool = true,
                      preserve_step_stats::Bool = true,
                      warn_step::Bool = true,
                      thresh_warn::Float64 = 0.2,
                      thresh_fail::Float64 = 0.5)
    @assert s > 0
    @assert dt > 0

    # N scaling
    Nnew = preserve_density ? max(1, round(Int, P.N * s^2)) : P.N
    if Nnew < 10
        @warn "N = $Nnew after rescaling. Proper flocking behaviour is unlikely with fewer than ~10 agents."
    end

    # dt scaling (kinematic similarity)
    dt_new = dt * s

    # rate/noise scaling if preserving per-step stats
    max_turn_rate_new = preserve_step_stats ? (P.max_turn_rate / s) : P.max_turn_rate

    noise_new = P.noise
    if preserve_step_stats && noise_new.mode == Diffusion
        noise_new = CouzinNoise(mode=Diffusion, s=noise_new.s, Dθ=noise_new.Dθ / s)
    end

    P_new = update_CouzinParams(
        P;
        N = Nnew,
        Zr = P.Zr * s,
        Zo = P.Zo * s,
        Za = P.Za * s,
        L  = P.L  * s,
        max_turn_rate = max_turn_rate_new,
        noise = noise_new
        # blind_half_angle stays unchanged via defaults
    )

    if warn_step
        warn_if_step_too_large(P_new, dt_new;
                               thresh_warn=thresh_warn, thresh_fail=thresh_fail,
                               context="after space scaling (dt→s·dt)")
    end

    return P_new, dt_new
end



"""
    scale_dynamics!(P::CouzinParams, dt::Float64; g=1.0) -> (P_new, dt_new)

Change the physical speed of the swarm relative to external inputs
while preserving the Couzin behavioural regime.

- speed -> g*speed
- interaction ranges -> g*(...)
- dt unchanged
- ε = v*dt/Zr invariant → milling preserved

This changes how fast the swarm moves in real space
without changing decision geometry.

Use this to match swarm response rate to an external driver.
"""
function scale_swarm_dynamics!(P::CouzinParams, dt::Float64;
                         g::Float64 = 1.0,
                         warn_step::Bool = true,
                         thresh_warn::Float64 = 0.2,
                         thresh_fail::Float64 = 0.5)

    @assert g > 0
    @assert dt > 0

    P_new = update_CouzinParams(
        P;
        Zr = P.Zr * g,
        Zo = P.Zo * g,
        Za = P.Za * g,
        speed = P.speed * g,
        L = P.L * g
        # keep turning rate & noise unchanged
    )

    if warn_step
        warn_if_step_too_large(P_new, dt;
                               thresh_warn=thresh_warn,
                               thresh_fail=thresh_fail,
                               context="after dynamics scaling")
    end

    return P_new, dt
end


# ============================================================
# Rescale summary printer
# ============================================================

_fmtval(v::Int)     = string(v)
_fmtval(v::Float64) = string(round(v; sigdigits = 5))
_fmtval(v)          = string(v)

function _print_rescale_summary(
    scale_info::NamedTuple,
    P0::CouzinParams, dt0::Float64,
    P1::CouzinParams, dt1::Float64,
)
    type = scale_info.type
    factor_str = if type == :space
        "s = $(scale_info.s)"
    elseif type == :time
        "f = $(scale_info.f)"
    elseif type == :dynamics
        "g = $(scale_info.g)"
    else
        string(scale_info)
    end

    rows = Tuple{String, String, String}[]

    function maybe(name, v0, v1)
        v0 != v1 && push!(rows, (name, _fmtval(v0), _fmtval(v1)))
    end

    maybe("N",             P0.N,             P1.N)
    maybe("Zr",            P0.Zr,            P1.Zr)
    maybe("Zo",            P0.Zo,            P1.Zo)
    maybe("Za",            P0.Za,            P1.Za)
    maybe("L",             P0.L,             P1.L)
    maybe("speed",         P0.speed,         P1.speed)
    maybe("max_turn_rate", P0.max_turn_rate, P1.max_turn_rate)
    P0.noise.Dθ != P1.noise.Dθ &&
        push!(rows, ("noise.Dθ", _fmtval(P0.noise.Dθ), _fmtval(P1.noise.Dθ)))
    maybe("dt",            dt0,              dt1)

    isempty(rows) && return

    w = maximum(length(r[1]) for r in rows)
    println("Rescale :$(type)  ($factor_str)")
    for (name, before, after) in rows
        println("  $(rpad(name, w)) : $before  →  $after")
    end
end

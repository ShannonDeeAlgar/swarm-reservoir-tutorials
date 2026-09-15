# ============================================================
# my_predator.jl
# Optional utilities for generating a predator-style input signal (from a
# driving dynamical system, e.g. Lorenz) and matching its spatial/temporal
# scale to a CouzinReservoir before use with PredatorCoupling
# (SWARM_RC/my_swarmRC.jl). Not required for a minimal swarm-reservoir --
# only needed if you want a chaotic driving signal rather than e.g. randn().
# ============================================================
#
# `predator_input` requires a `lorenz_data(; rng, tspan, dt_data)` function
# to be defined (e.g. from TIME_SERIES/my_systems.jl) if cfg.system == :lorenz.

using Statistics
using StaticArrays
using ProgressMeter

# ----------------------------
# Types
# ----------------------------

"""
Configuration for generating predator input data.

- `system`: `:lorenz`, `:rossler`, `:hyper_rossler`, `:logistic`, or
  `:mackey_glass`
- `seed`, `dt_data`: sampling of the continuous-time system
- `T`: number of samples to return (after any embedding)
- `dims`: coordinates of the attractor projection (1-indexed), e.g. `(1,2)`
  for a 2D swarm or `(1,2,3)` for a 3D swarm
- `embed`: set to `nothing` to disable embedding, or a NamedTuple like:
    `(m=3, τ=40, source_dim=1, take=(1,2), horizon=1)`
  where `source_dim` chooses the scalar series and `take` chooses coordinates
  from its delay embedding.
- `system_kwargs`: optional generator-specific keyword arguments.
"""
Base.@kwdef struct PredatorConfig
    system::Symbol = :lorenz
    seed::Int = 100
    dt_data::Float64 = 0.01
    T::Int = 2000

    dims::Tuple{Vararg{Int}} = (1, 2)

    embed::Union{Nothing,NamedTuple} = nothing
    system_kwargs::NamedTuple = NamedTuple()
end

"""
Returned predator input.

- `U`: input_dim×T matrix suitable for reservoir input (columns are time)
- `raw`: raw system data matrix (d×N_raw)
- `info`: Dict with helpful metadata
"""
struct PredatorInput
    U::Matrix{Float64}
    raw::Matrix{Float64}
    info::Dict{Symbol,Any}
end

# ----------------------------
# Internals: system dispatch
# ----------------------------

# You said you'll keep system generators in my_systems.jl.
# This keeps Predator.jl clean: you include my_systems.jl in your main, or you can include it here.
#
# If you prefer Predator.jl to be standalone, uncomment the next line and ensure the path is correct:
# include("my_systems.jl")

function _generate_system(cfg::PredatorConfig)
    rng = MersenneTwister(cfg.seed)
    emb_extra = cfg.embed === nothing ? 0 :
        (cfg.embed.m - 1) * cfg.embed.τ + get(cfg.embed, :horizon, 0)
    nraw = cfg.T + emb_extra

    if cfg.system == :lorenz
        tspan = (0.0, (nraw - 1) * cfg.dt_data)
        out = lorenz_data(; merge((rng=rng, tspan=tspan, dt_data=cfg.dt_data),
                                  cfg.system_kwargs)...)
        tgrid = collect(range(tspan[1]; step=cfg.dt_data, length=nraw))
        raw = Matrix{Float64}(Array(out.sol(tgrid)))
    elseif cfg.system == :rossler
        tspan = (0.0, (nraw - 1) * cfg.dt_data)
        out = rossler_data(; merge((tspan=tspan, dt_data=cfg.dt_data),
                                   cfg.system_kwargs)...)
        raw = Matrix{Float64}(out.data)
    elseif cfg.system == :hyper_rossler
        tspan = (0.0, (nraw - 1) * cfg.dt_data)
        out = hyper_rossler_data(; merge((tspan=tspan, dt_data=cfg.dt_data),
                                         cfg.system_kwargs)...)
        raw = Matrix{Float64}(out.data)
    elseif cfg.system == :logistic
        out = logistic_data(; merge((T=nraw,), cfg.system_kwargs)...)
        raw = Matrix{Float64}(out.data)
    elseif cfg.system == :mackey_glass
        out = mackey_glass_data(; merge((n_out=nraw,), cfg.system_kwargs)...)
        raw = Matrix{Float64}(out.data)
    else
        error("Unknown system $(cfg.system). Use :lorenz, :rossler, " *
              ":hyper_rossler, :logistic, or :mackey_glass.")
    end
    size(raw, 2) >= nraw || error(
        "$(cfg.system) returned $(size(raw,2)) samples; need at least $nraw.")
    dt_used = cfg.system in (:logistic, :mackey_glass) ? 1.0 : cfg.dt_data
    return raw[:, 1:nraw], Dict{Symbol,Any}(
        :system => cfg.system, :dt_data => dt_used, :seed => cfg.seed,
        :system_kwargs => cfg.system_kwargs,
    )
end

# ----------------------------
# Internals: embedding + selection
# ----------------------------

# Select an arbitrary 2D/3D projection from a state or embedding matrix.
@inline function _take_dims(X::AbstractMatrix, dims::Tuple{Vararg{Int}}, T::Int)
    @assert !isempty(dims) "Select at least one input coordinate."
    @assert all(d -> 1 <= d <= size(X,1), dims) "Projection dims=$dims exceed data dimension $(size(X,1))."
    N = size(X,2)
    @assert T <= N "Requested T=$T exceeds available samples N=$N."
    return Matrix{Float64}(X[collect(dims), 1:T])
end

# If you have a Takens function elsewhere, this wrapper will call it.
# Expected signature: takens_embed(x; m, τ, horizon) -> (U, Y, idx) or similar
function _takens_embed_input(raw::AbstractMatrix, emb::NamedTuple, T::Int)
    m        = emb.m
    τ        = emb.τ
    src      = get(emb, :source_dim, 1)
    take     = get(emb, :take, (1,2))
    horizon  = get(emb, :horizon, 1)

    @assert 1 <= src <= size(raw,1) "source_dim=$(src) out of range for raw dimension."
    x = vec(raw[src, :])

    # You supply this elsewhere.
    @assert @isdefined(takens_embed) "takens_embed is not defined. Either define it or set cfg.embed=nothing."

    U_emb, Y, idx = takens_embed(x; m=m, τ=τ, horizon=horizon)

    # Common convention: embedded U is (m×N_emb) or (N_emb×m). Handle both.
    if size(U_emb, 1) == m
        X = U_emb
    elseif size(U_emb, 2) == m
        X = permutedims(U_emb)  # make it m×N
    else
        error("takens_embed returned size $(size(U_emb)), cannot infer orientation (m=$m).")
    end

    U = _take_dims(X, Tuple(take), T)
    return U, Dict{Symbol,Any}(
        :embed => true,
        :m => m,
        :τ => τ,
        :source_dim => src,
        :take => take,
        :horizon => horizon,
        :embedded_len_available => size(X,2),
    )
end

# ----------------------------
# Public API
# ----------------------------

"""
    predator_input(cfg::PredatorConfig) -> PredatorInput

Generate a projected attractor or delay-embedded trajectory as an
`input_dim×T` reservoir input matrix.

Options:
- No embedding: choose raw system coordinates via `cfg.dims`.
- With embedding: set `cfg.embed = (m=..., τ=..., source_dim=..., take=(...), horizon=...)`.
"""
function predator_input(cfg::PredatorConfig)
    @assert cfg.T >= 2 "Need at least 2 time points."

    raw, sysinfo = _generate_system(cfg)

    if cfg.embed === nothing
        U = _take_dims(raw, cfg.dims, cfg.T)
        info = merge(sysinfo, Dict{Symbol,Any}(
            :embed => false,
            :dims => cfg.dims,
            :T => cfg.T,
            :raw_len_available => size(raw,2),
        ))
        return PredatorInput(U, raw, info)
    else
        U, embinfo = _takens_embed_input(raw, cfg.embed, cfg.T)
        info = merge(sysinfo, embinfo, Dict{Symbol,Any}(
            :T => cfg.T,
            :raw_len_available => size(raw,2),
        ))
        return PredatorInput(U, raw, info)
    end
end

# end # module Predator


function choose_sigma_target(U::AbstractMatrix, L::Float64;
                                       q::Float64 = 0.99,
                                       margin::Float64 = 0.05L)
    @assert size(U,1) == 2
    μ = mean(U, dims=2)
    σ = std(U,  dims=2) .+ 1e-12
    Z = (U .- μ) ./ σ              # standardised
    r = sqrt.(Z[1,:].^2 .+ Z[2,:].^2)
    r_q = quantile(r, q)
    R_desired = (L/2 - margin)
    sigma_target = R_desired / r_q
    return sigma_target
end



function predator_steps_with_stride(U::AbstractMatrix;
                                    L::Float64,
                                    centre::NTuple{2,Float64},
                                    sigma_target::Float64,
                                    stride::Int)

    @assert size(U,1) == 2 "U must be 2×T"
    @assert stride >= 1

    # downsample
    U2 = U[:, 1:stride:end]
    @assert size(U2,2) >= 2 "Need at least 2 samples after downsampling (reduce stride or provide longer U)."

    # standardise
    μ  = mean(U2, dims=2)
    σ  = std(U2,  dims=2) .+ 1e-12
    X  = (U2 .- μ) ./ σ .* sigma_target

    # map into box + periodic wrap
    cx, cy = centre
    px = mod.(cx .+ vec(X[1, :]), L)
    py = mod.(cy .+ vec(X[2, :]), L)

    # compute periodic step sizes
    Tn = length(px)
    steps = Vector{Float64}(undef, Tn-1)
    @inbounds for t in 2:Tn
        dx, dy = periodic_disp(px[t-1], py[t-1], px[t], py[t], L)
        steps[t-1] = hypot(dx, dy)
    end

    return steps
end

function choose_stride_for_matching(U::AbstractMatrix,
                                    target_step::Float64;
                                    L::Float64,
                                    centre::NTuple{2,Float64},
                                    sigma_target::Float64 = 2.0,
                                    stride_candidates = 1:200)

    best_k   = first(stride_candidates)
    best_err = Inf
    best_med = NaN

    for k in stride_candidates
        steps = predator_steps_with_stride(U;
                                           L=L,
                                           centre=centre,
                                           sigma_target=sigma_target,
                                           stride=k)

        med = median(steps)
        err = abs(med - target_step)

        if err < best_err
            best_err = err
            best_k   = k
            best_med = med
        end
    end

    if best_k == 1 && best_med > target_step * (1 + 1e-6)
        ratio = best_med / target_step
        @warn """
        Predator still too fast at minimum stride (k=1).

          predator median step = $(best_med)
          swarm median step    = $(target_step)
          ratio (pred/swarm)   = $(ratio)

        Downsampling cannot slow the predator further.

        Next steps:
          • Ensure predator data is sufficiently long.
          • If possible: regenerate input with smaller dt_data (denser sampling).
          • Otherwise: rescale the swarm in time to match predator.
        """
    end

    return best_k, best_med, best_err
end


# U is 2×T, columns are positions
function predator_step_sizes_periodic(U::AbstractMatrix{<:Real}, L::Real)
    T = size(U, 2)
    steps = Vector{Float64}(undef, T-1)
    @inbounds for t in 1:(T-1)
        p1 = SVector2(float(U[1,t]),   float(U[2,t]))
        p2 = SVector2(float(U[1,t+1]), float(U[2,t+1]))
        d  = displacement(p1, p2, L)     # your existing periodic displacement
        steps[t] = norm(d)
    end
    return steps
end

# ============================================================
# predator_to_couzin_reservoir.jl
# Predator → CouzinReservoir similarity matching (NO swarm scaling, NO g)
#
# Requires:
#   res::CouzinReservoir has fields: P::CouzinParams, dt::Float64
#   res.P has fields: L::Float64, speed::Float64
#
# Produces:
#   α = Rs/Rp     (space scale)
#   β = (α*Vp)/Vs (time scale)
# and a resampled predator trajectory U_new embedded in [0,L) (periodic box)
#
# Notes:
# - dt_native: native sampling interval of the predator data U
# - dt_swarm:  timestep used to step the reservoir; defaults to res.dt
# ============================================================

using LinearAlgebra, Statistics

# ------------------------------------------------------------
# Predator characteristic scales (native units)
# ------------------------------------------------------------

"""
    predator_scales_axisrange(U; dt=1.0)

Predator characteristic extent + speed in native units.

Rp: max coordinate range across dimensions (box-side proxy).
Vp: mean step speed = mean(norm(Δx))/dt.

Returns (Rp, Vp, range_vec).
"""
function predator_scales_axisrange(U::AbstractMatrix; dt::Real=1.0)
    d, T = size(U)
    @assert d ≥ 1
    @assert T ≥ 2
    @assert dt > 0

    mins = vec(mapslices(minimum, U; dims=2))
    maxs = vec(mapslices(maximum, U; dims=2))
    range_vec = maxs .- mins
    Rp = maximum(range_vec)

    dU = @views U[:, 2:end] .- U[:, 1:end-1]
    Vp = mean([norm(@views dU[:, k]) for k in 1:size(dU, 2)]) / dt

    return (; Rp, Vp, range_vec)
end

# ------------------------------------------------------------
# Linear interpolation resampling
# ------------------------------------------------------------

"""
    resample_linear(U, t_old, t_new)

Linear interpolation of a d×T trajectory U given increasing time grids.
Assumes t_new lies within [t_old[1], t_old[end]].
"""
function resample_linear(U::AbstractMatrix, t_old::AbstractVector, t_new::AbstractVector)
    d, T = size(U)
    Tnew = length(t_new)
    @assert length(t_old) == T
    @assert T ≥ 2
    @assert Tnew ≥ 1

    Unew = Matrix{Float64}(undef, d, Tnew)
    j = 1
    @inbounds for k in 1:Tnew
        tk = t_new[k]
        while (j < T-1) && (t_old[j+1] < tk)
            j += 1
        end
        t0, t1 = t_old[j], t_old[j+1]
        w = (tk - t0) / (t1 - t0)
        for i in 1:d
            Unew[i, k] = (1 - w) * U[i, j] + w * U[i, j+1]
        end
    end
    return Unew
end

# ------------------------------------------------------------
# Reservoir target scales (reference frame = CouzinReservoir)
# ------------------------------------------------------------

"""
    couzin_targets(res; extent_target=:domain, Rs=nothing, Vs=nothing, dt_swarm=nothing)

Define the target scales in *reservoir units*:

- Vs defaults to res.P.speed
- Rs defaults to:
    extent_target = :domain -> Rs = res.P.L (predator roams the whole domain)
    extent_target = :swarm  -> Rs must be provided (measured swarm extent)

- dt_swarm defaults to res.dt (Couzin integration timestep)

Returns NamedTuple: (Rs, Vs, L, dt_swarm).
"""
function couzin_targets(res::CouzinReservoir;
    extent_target::Symbol = :domain,
    Rs = nothing,
    Vs = nothing,
    dt_swarm = nothing,
)
    @assert extent_target in (:domain, :swarm)

    L = float(res.P.L)
    Vs_val = Vs === nothing ? float(res.P.speed) : float(Vs)

    Rs_val = if extent_target == :domain
        L
    else
        @assert Rs !== nothing "extent_target=:swarm requires Rs (measured swarm extent) to be provided."
        float(Rs)
    end

    dt_val = dt_swarm === nothing ? float(res.dt) : float(dt_swarm)

    @assert L > 0 && Rs_val > 0 && Vs_val > 0 && dt_val > 0
    return (; Rs=Rs_val, Vs=Vs_val, L=L, dt_swarm=dt_val)
end

# ------------------------------------------------------------
# Main: predator -> CouzinReservoir matching and embedding
# ------------------------------------------------------------

"""
    match_predator_to_couzin(res, U; dt_native, dt_swarm=nothing, extent_target=:domain, Rs=nothing, Vs=nothing)

Compute similarity scales (α,β) mapping predator -> reservoir units so that:

    α*Rp ≈ Rs
    (α/β)*Vp ≈ Vs

Then build U_new:
- centre predator (remove mean)
- scale space by α
- scale time by t' = β t (β>1 => slower in reservoir time)
- resample to dt_swarm (defaults to res.dt)
- shift to middle of box and wrap periodic to [0,L)

Arguments:
- U: d×T predator trajectory in native units
- dt_native: native sampling interval of U
- dt_swarm:  reservoir timestep; if not provided uses res.dt
- extent_target:
    :domain -> Rs = L  (predator spans the whole domain)
    :swarm  -> Rs provided externally (e.g. measured swarm RMS radius)

Returns NamedTuple including:
- α,β and diagnostics
- U_new (d×Tnew) in reservoir units, periodic box [0,L)
- t_new (Tnew) corresponding reservoir-time grid
"""
function match_predator_to_couzin(
    res::CouzinReservoir,
    U::AbstractMatrix;
    dt_native::Real,
    dt_swarm::Union{Nothing,Real}=nothing,
    extent_target::Symbol=:domain,
    Rs=nothing,
    Vs=nothing
)

    @assert dt_native > 0
    d, T = size(U)
    @assert T ≥ 2

    # ------------------------------------------------------------
    # Reservoir scales
    # ------------------------------------------------------------

    L = float(res.P.L)

    Vs_val = Vs === nothing ? float(res.P.speed) : float(Vs)

    Rs_val = if extent_target == :domain
        L
    else
        @assert Rs !== nothing "extent_target=:swarm requires Rs."
        float(Rs)
    end

    dt_res = dt_swarm === nothing ? float(res.dt) : float(dt_swarm)

    # ------------------------------------------------------------
    # Predator scales
    # ------------------------------------------------------------

    ps = predator_scales_axisrange(U; dt=dt_native)
    Rp, Vp = ps.Rp, ps.Vp

    # ------------------------------------------------------------
    # Similarity transform
    # ------------------------------------------------------------

    α = Rs_val / Rp
    β = (α * Vp) / Vs_val

    # ------------------------------------------------------------
    # Space transform
    # ------------------------------------------------------------

    μ = mean(U; dims=2)
    U0 = U .- μ
    Uspace = α .* U0

    # ------------------------------------------------------------
    # Time transform
    # ------------------------------------------------------------

    t_native = (0:T-1) .* dt_native
    t_old = β .* t_native

    t_new = collect(0.0:dt_res:t_old[end])
    Unew = resample_linear(Uspace, t_old, t_new)

    # ------------------------------------------------------------
    # Embed in periodic box
    # ------------------------------------------------------------

    Unew .+= L/2

    @inbounds for k in 1:size(Unew,2), i in 1:size(Unew,1)
        Unew[i,k] = mod(Unew[i,k], L)
    end

    return (;
        L=L,
        Rs=Rs_val,
        Vs=Vs_val,
        dt_swarm=dt_res,
        extent_target=extent_target,

        Rp=Rp,
        Vp=Vp,
        range_vec=ps.range_vec,
        dt_native=dt_native,

        alpha=α,
        beta=β,

        Rp_scaled = α*Rp,
        Vp_scaled = (α/β)*Vp,

        U_new = Unew,
        t_new = t_new
    )
end

"""
    match_predator_to_lymburn(res::LymburnReservoir, U; dt_native, dt_swarm=nothing, Rs, Vs=nothing)

`LymburnReservoir` counterpart to `match_predator_to_couzin`. The Lymburn
model has no periodic domain -- the swarm is held near `res.P.xh` by a
homing force rather than wrapped in a box -- so this centres the scaled
predator trajectory on `res.P.xh` instead of wrapping into `[0,L)`. `Rs` is
required (there is no domain size to default to; pass roughly the swarm's
own spatial extent, e.g. 2-3x its typical radius from `res.P.xh`, so the
predator's roaming region actually overlaps where the swarm lives -- see
the `extent_target=:swarm` discussion for `match_predator_to_couzin` in
`Tutorial_swarmRC.ipynb`'s input-selection section for why this matters).
"""
function match_predator_to_lymburn(
    res::AbstractReservoir,
    U::AbstractMatrix;
    dt_native::Real,
    dt_swarm::Union{Nothing,Real}=nothing,
    Rs::Real,
    Vs=nothing,
)
    nameof(typeof(res)) == :LymburnReservoir ||
        throw(ArgumentError("match_predator_to_lymburn requires a LymburnReservoir; got $(typeof(res))."))
    @assert dt_native > 0
    d, T = size(U)
    @assert T ≥ 2

    Vs_val = Vs === nothing ? float(res.P.s) : float(Vs)
    dt_res = dt_swarm === nothing ? float(res.dt) : float(dt_swarm)

    ps = predator_scales_axisrange(U; dt=dt_native)
    Rp, Vp = ps.Rp, ps.Vp

    α = float(Rs) / Rp
    β = (α * Vp) / Vs_val

    μ = mean(U; dims=2)
    U0 = U .- μ
    Uspace = α .* U0

    t_native = (0:T-1) .* dt_native
    t_old = β .* t_native

    t_new = collect(0.0:dt_res:t_old[end])
    Unew = resample_linear(Uspace, t_old, t_new)

    # Centre on the home point instead of wrapping into a periodic box --
    # this model has no domain to wrap into.
    Unew[1, :] .+= res.P.xh.x
    Unew[2, :] .+= res.P.xh.y

    return (;
        Rs=float(Rs), Vs=Vs_val, dt_swarm=dt_res,
        Rp=Rp, Vp=Vp, range_vec=ps.range_vec, dt_native=dt_native,
        alpha=α, beta=β,
        Rp_scaled=α*Rp, Vp_scaled=(α/β)*Vp,
        U_new=Unew, t_new=t_new,
    )
end


# -------------------------
# Pretty printer
# -------------------------

function print_matching_report(res::CouzinReservoir,
    U_native::AbstractMatrix,
    out_match;  # output from match_predator_to_couzin(...)
    dt_native::Real,
    extent_target::Symbol=:domain,
)

    targ = couzin_targets(res; extent_target=extent_target, Rs=get(out_match, :Rs, nothing), Vs=get(out_match, :Vs, nothing))

    # predator native stats (native time units)
    pn = predator_scales_axisrange(U_native; dt=dt_native)

    # predator matched stats (reservoir time units)
    U_new = out_match.U_new
    pm = predator_scales_axisrange(U_new; dt=targ.dt_swarm)

    println("\n================ Matching report ================")
    println("Reference frame: CouzinReservoir")
    println("  L (domain side)      = $(targ.L)")
    println("  Rs (extent target)   = $(targ.Rs)   (extent_target = $(extent_target))")
    println("  Vs (speed target)    = $(targ.Vs)")
    println("  dt_swarm (res.dt)    = $(targ.dt_swarm)")
    println("-------------------------------------------------")
    println("Predator (native):")
    println("  dt_native            = $(dt_native)")
    println("  Rp_native            = $(pn.Rp)   (max axis range)")
    println("  range_vec_native     = $(pn.range_vec)")
    println("  Vp_native            = $(pn.Vp)")
    println("-------------------------------------------------")
    println("Similarity scales used:")
    println("  alpha                = $(out_match.alpha)")
    println("  beta                 = $(out_match.beta)")
    println("  (predicted) Rp_scaled= $(out_match.Rp_scaled)   (target ~ Rs)")
    println("  (predicted) Vp_scaled= $(out_match.Vp_scaled)   (target ~ Vs)")
    println("-------------------------------------------------")
    println("Predator (after scaling + resampling + wrap):")
    println("  Rp_matched           = $(pm.Rp)   (max axis range)")
    println("  range_vec_matched    = $(pm.range_vec)")
    println("  Vp_matched           = $(pm.Vp)")
    println("-------------------------------------------------")
    println("Ratios (matched / target):")
    println("  Rp_matched / Rs = $(pm.Rp / targ.Rs)")
    println("  Vp_matched / Vs = $(pm.Vp / targ.Vs)")
    println("=================================================\n")

    return (; targets=targ, predator_native=pn, predator_matched=pm)
end


"""
    run_offline_predator_drive!(res::AbstractReservoir, Ppred; dt, T=size(Ppred,2), rng=res.rng,
                                 order_parameters=order_parameters_2d, displacement=displacement, L=nothing)

Drive the swarm-reservoir `res` (`CouzinReservoir` or `LymburnReservoir`)
with an offline 2×T predator trajectory `Ppred` (e.g. from
`match_predator_to_couzin`), logging the raw state after each step
(compatible with `animate_from_states` in my_swarmRC_plotting.jl) and the
swarm's order parameters. Returns a NamedTuple with fields `states`,
`pred_hist`, `t`, `dilation`, `rotation`, `polarisation`.

Defaults (`order_parameters_2d`, periodic `displacement`, `L=res.P.L`)
match `CouzinReservoir`'s periodic-domain model; pass
`order_parameters=Lymburn_order_parameters_2d,
displacement=displacement_nonperiodic_2d, L=res.P.plot_extent` for a
`LymburnReservoir` (or any `L` value -- it is cosmetic for that model).
"""
function run_offline_predator_drive!(res::AbstractReservoir,
                                    Ppred::AbstractMatrix;
                                    dt::Real,
                                    T::Int = size(Ppred, 2),
                                    rng::AbstractRNG = res.rng,
                                    order_parameters::Function = order_parameters_2d,
                                    displacement::Function = displacement,
                                    L::Union{Nothing,Real} = nothing,
                                    show_progress::Bool = true)

    @assert size(Ppred, 1) == 2 "Ppred must be 2×T"
    @assert T >= 1 "T must be ≥ 1"
    @assert T <= size(Ppred,2) "T exceeds available columns in Ppred"
    @assert dt > 0

    Luse = L === nothing ? res.P.L : L

    states       = Vector{Any}(undef, T)
    pred_hist    = Vector{SVector2}(undef, T)
    dilation     = Vector{Float32}(undef, T)
    rotation     = Vector{Float32}(undef, T)
    polarisation = Vector{Float32}(undef, T)

    prog = show_progress ? Progress(T; desc="run_offline_predator_drive!", showspeed=true) : nothing

    for t in 1:T
        reservoir_step!(res, view(Ppred, :, t); rng=rng)

        states[t] = raw_state(res)
        pred_hist[t] = SVector2(Ppred[1, t], Ppred[2, t])

        dil, rot, pol, _ = order_parameters(res.state.pos, res.state.vel, Luse; displacement=displacement)
        dilation[t]     = Float32(dil)
        rotation[t]     = Float32(rot)
        polarisation[t] = Float32(pol)

        show_progress && next!(prog)
    end

    return (;
        states = states,
        pred_hist = pred_hist,
        t = Float32.(0:dt:dt*(T-1)),
        dilation = dilation,
        rotation = rotation,
        polarisation = polarisation,
    )
end

"""
    run_offline_input_drive!(res, U; dt, order_parameters, displacement, L)

Model-independent counterpart to `run_offline_predator_drive!`. It accepts an
input matrix of whatever row dimension the selected coupling requires and logs
states, order parameters and the input itself. Use this for global scalar
couplings such as `TemperatureSpeedCoupling`.
"""
function run_offline_input_drive!(res::AbstractReservoir, U::AbstractMatrix;
    dt::Real, T::Int = size(U, 2), rng::AbstractRNG = res.rng,
    order_parameters::Function = order_parameters_2d,
    displacement::Function = displacement,
    L::Union{Nothing,Real} = nothing,
    show_progress::Bool = true)

    T >= 1 || throw(ArgumentError("T must be at least 1."))
    T <= size(U, 2) || throw(ArgumentError("T exceeds the available input columns."))
    dt > 0 || throw(ArgumentError("dt must be positive."))
    Luse = L === nothing ? res.P.L : L

    states = Vector{Any}(undef, T)
    dilation = Vector{Float32}(undef, T)
    rotation = Vector{Float32}(undef, T)
    polarisation = Vector{Float32}(undef, T)
    prog = show_progress ? Progress(T; desc="run_offline_input_drive!", showspeed=true) : nothing

    for t in 1:T
        reservoir_step!(res, view(U, :, t); rng=rng)
        states[t] = raw_state(res)
        dil, rot, pol, _ = order_parameters(res.state.pos, res.state.vel, Luse;
            displacement=displacement)
        dilation[t], rotation[t], polarisation[t] = Float32(dil), Float32(rot), Float32(pol)
        show_progress && next!(prog)
    end

    return (; states, input=Matrix(U[:, 1:T]),
        t=Float32.(0:dt:dt*(T-1)), dilation, rotation, polarisation)
end


function plot_predator_step_sizes(Ppred::AbstractMatrix, L::Float64, dt::Float64;
                                  title::String="Predator step sizes")
    @assert size(Ppred,1) == 2
    T = size(Ppred,2)

    steps = zeros(Float64, T-1)
    speeds = zeros(Float64, T-1)
    for t in 2:T
        dx, dy = periodic_disp(Ppred[1,t-1], Ppred[2,t-1], Ppred[1,t], Ppred[2,t], L)
        s = hypot(dx, dy)
        steps[t-1] = s
        speeds[t-1] = s / dt
    end

    fig = Figure(size=(900, 400))
    ax  = Axis(fig[1,1], title=title, xlabel="t (step index)", ylabel="distance per step")
    lines!(ax, steps, linewidth=2)

    fig
end


"""
median_step_length(P::AbstractMatrix, L)

P must be 2×T trajectory inside periodic box of size L.
Returns median displacement per step.
"""
function median_step_length(P::AbstractMatrix, L::Float64)
    @assert size(P,1) == 2
    T = size(P,2)
    @assert T ≥ 2

    steps = Vector{Float64}(undef, T-1)

    @inbounds for t in 2:T
        dx, dy = periodic_disp(P[1,t-1], P[2,t-1],
                               P[1,t],   P[2,t],   L)
        steps[t-1] = hypot(dx, dy)
    end

    return median(steps)
end

"""
Plot a single input trajectory U (Nu×T) with:
- left column: time series for each dimension (Nu rows)
- right column: phase portrait for chosen dims, spanning all rows

This mirrors the structure of `plot_downsample`, but for a single U only.
"""
function plot_input_trajectory(U::AbstractMatrix;
    phase_dims::Tuple{Int,Int} = (1, 2),
    dt::Real = 1.0,
    title::String = "Input trajectory",
    labels::Union{Nothing,Vector{String}} = nothing,
    show_points::Bool = false,
    every::Int = 1,
    phase_every::Int = every,
)

    Nu, T = size(U)
    d1, d2 = phase_dims
    @assert 1 ≤ d1 ≤ Nu && 1 ≤ d2 ≤ Nu "phase_dims must be valid row indices of U."
    @assert every ≥ 1 && phase_every ≥ 1
    @assert dt > 0

    labels === nothing && (labels = ["u_$d" for d in 1:Nu])

    fig = Figure(size = (1200, 700))

    # ---- Left column: time series (Nu rows) ----
    for d in 1:Nu
        ax = Axis(fig[d, 1],
            xlabel = d == Nu ? "time" : "",
            ylabel = labels[d],
            title  = d == 1 ? title : ""
        )

        idx = 1:every:T
        t = (idx .- 1) .* dt
        u = vec(@view U[d, idx])

        lines!(ax, t, u, linewidth=2)

        if show_points
            scatter!(ax, t, u, markersize=3)
        end
    end

    # ---- Right column: phase portrait spanning all rows ----
    axP = Axis(fig[1:Nu, 2],
        xlabel = labels[d1],
        ylabel = labels[d2],
        title  = "Phase space"
    )

    idxP = 1:phase_every:T
    x = vec(@view U[d1, idxP])
    y = vec(@view U[d2, idxP])

    lines!(axP, x, y, linewidth=2)
    if show_points
        scatter!(axP, x, y, markersize=3)
    end

    # layout balance like your diagnostic plots
    colsize!(fig.layout, 1, Relative(0.52))
    colsize!(fig.layout, 2, Relative(0.48))

    return fig
end

"""
Downsampling diagnostic figure:
- Left column: time series for each u_d (Nu rows)
- Right column: phase portrait (u_{d1} vs u_{d2}) spanning all rows
"""
function plot_downsample(U_raw::AbstractMatrix;
                                      factors::Vector{Int} = [1, 5, 10, 20, 50],
                                      phase_dims::Tuple{Int,Int} = (1, 2),
                                      dt::Real = 1.0,
                                      title::String = "Lorenz input with downsampling",
                                      labels::Union{Nothing,Vector{String}} = nothing)

    Nu, T = size(U_raw)
    d1, d2 = phase_dims
    @assert 1 ≤ d1 ≤ Nu && 1 ≤ d2 ≤ Nu "phase_dims must be valid row indices of U_raw."

    # axis labels
    labels === nothing && (labels = ["u_$d" for d in 1:Nu])

    fig = Figure(size = (1200, 700))

    # ---- Left column: time series (Nu rows) ----
    for d in 1:Nu
        ax = Axis(fig[d, 1],
            xlabel = d == Nu ? "time" : "",
            ylabel = labels[d],
            title  = d == 1 ? title : ""
        )

        for f in factors
            idx = 1:f:T
            t_ds = (idx .- 1) .* dt
            U_ds = @view U_raw[d, idx]

            lines!(ax, t_ds, vec(U_ds),
                label = f == 1 ? "original (×1)" : "downsample ×$f"
            )
        end

        d == 1 && axislegend(ax, position = :rb)
    end

    # ---- Right column: phase portrait spanning all rows ----
    axP = Axis(fig[1:Nu, 2],
        xlabel = labels[d1],
        ylabel = labels[d2],
        title  = "Phase space"
    )

    for f in factors
        idx = 1:f:T
        x = vec(@view U_raw[d1, idx])
        y = vec(@view U_raw[d2, idx])

        lines!(axP, x, y,
            label = f == 1 ? "original (×1)" : "downsample ×$f"
        )
    end
    axislegend(axP, position = :rb)

    # layout balance like your TF diagnostic
    colsize!(fig.layout, 1, Relative(0.52))
    colsize!(fig.layout, 2, Relative(0.48))

    return fig
end


using LinearAlgebra

function rescale_time(U::AbstractMatrix, α::Real)
    T = size(U,2)
    t_old = collect(1:T)
    t_new = range(1, T; length = Int(round(T/α)))

    Uα = zeros(2, length(t_new))

    for d in 1:2
        Uα[d,:] = interp_linear(t_old, U[d,:], t_new)
    end

    return Uα
end

"""
    interp_linear(x_old, y_old, x_new)

Piecewise linear interpolation.

x_old : sorted vector (monotone increasing)
y_old : same length values
x_new : query points (can be non-integer)

Returns y(x_new).

Clamps outside domain to endpoints (important for trajectory edges).
"""
function interp_linear(x_old::AbstractVector,
                       y_old::AbstractVector,
                       x_new::AbstractVector)

    @assert length(x_old) == length(y_old)
    @assert issorted(x_old)

    n = length(x_old)
    y_new = similar(x_new, Float64)

    j = 1  # moving pointer (O(N) total)

    @inbounds for i in eachindex(x_new)
        x = x_new[i]

        # ---- clamp outside domain ----
        if x <= x_old[1]
            y_new[i] = y_old[1]
            continue
        elseif x >= x_old[end]
            y_new[i] = y_old[end]
            continue
        end

        # ---- advance interval pointer ----
        while j < n-1 && x > x_old[j+1]
            j += 1
        end

        # ---- linear interpolate ----
        x0 = x_old[j]
        x1 = x_old[j+1]
        y0 = y_old[j]
        y1 = y_old[j+1]

        t = (x - x0) / (x1 - x0)
        y_new[i] = (1-t)*y0 + t*y1
    end

    return y_new
end

"""
    moving_average_smooth(U, window)

Apply a centred moving average independently to each row of `U`.
`window` is the half-width in samples; non-positive values return a copy.
"""
function moving_average_smooth(U::AbstractMatrix, window::Int)
    window <= 0 && return copy(U)
    D, T = size(U)
    Usmooth = similar(U)
    @inbounds for d in 1:D, i in 1:T
        lo = max(1, i - window)
        hi = min(T, i + window)
        Usmooth[d, i] = mean(@view U[d, lo:hi])
    end
    return Usmooth
end

"""
    zscore_rescale(X; target_std=2.0)

Centre and rescale each row of `X` independently to standard deviation
`target_std`, matching the per-coordinate normalisation used by Lymburn
et al.
"""
function zscore_rescale(X::AbstractMatrix; target_std::Real = 2.0)
    D, T = size(X)
    Y = similar(X, Float64)
    @inbounds for d in 1:D
        row = @view X[d, :]
        μ = mean(row)
        σ = std(row)
        Y[d, :] .= target_std .* (row .- μ) ./ σ
    end
    return Y
end

# ============================================================
# predator_interaction_report / plot_predator_interaction_*
# An objective, non-visual alternative/companion to eyeballing
# animate_from_states -- see Tutorial_swarmRC.ipynb.
# ============================================================

function _lagged_corr(a::AbstractVector, b::AbstractVector, lags::AbstractVector{<:Integer})
    n = length(a)
    out = Vector{Float64}(undef, length(lags))
    @inbounds for (k, l) in enumerate(lags)
        if l >= 0
            out[k] = cor(view(a, 1:n-l), view(b, 1+l:n))
        else
            out[k] = cor(view(a, 1-l:n), view(b, 1:n+l))
        end
    end
    return out
end

"""
    predator_interaction_report(states, pred_hist, rp;
                                dt=1.0, max_lag_time=2.0)

Measure predator contact, predator-to-swarm distances and lagged coupling
to the swarm centre of mass. Returns the time series and scalar summaries
used by the companion plotting functions.
"""
function predator_interaction_report(states, pred_hist, rp::Real; dt::Real = 1.0, max_lag_time::Real = 2.0)
    T = length(states)
    @assert length(pred_hist) == T "states and pred_hist must have the same length"
    N = size(states[1].P, 2)

    contact_fraction = zeros(T)
    dist_nearest = zeros(T)
    dist_com = zeros(T)
    comx = zeros(T); comy = zeros(T)
    px = zeros(T); py = zeros(T)

    @inbounds for t in 1:T
        Pm = states[t].P
        pxt, pyt = pred_hist[t][1], pred_hist[t][2]
        d = sqrt.((view(Pm, 1, :) .- pxt) .^ 2 .+ (view(Pm, 2, :) .- pyt) .^ 2)
        contact_fraction[t] = count(<(rp), d) / N
        dist_nearest[t] = minimum(d)
        cx, cy = mean(view(Pm, 1, :)), mean(view(Pm, 2, :))
        comx[t], comy[t] = cx, cy
        dist_com[t] = sqrt((pxt - cx)^2 + (pyt - cy)^2)
        px[t], py[t] = pxt, pyt
    end

    max_lag = max(1, round(Int, max_lag_time / dt))
    lags = collect(-max_lag:max_lag)
    xcorr_x = _lagged_corr(px, comx, lags)
    xcorr_y = _lagged_corr(py, comy, lags)
    ibx = argmax(abs.(xcorr_x))
    iby = argmax(abs.(xcorr_y))
    izero = max_lag + 1   # lags[izero] == 0

    return (;
        contact_fraction, dist_nearest, dist_com, comx, comy, px, py,
        mean_contact_fraction = mean(contact_fraction),
        frac_timesteps_any_contact = count(>(0), contact_fraction) / T,
        mean_dist_nearest = mean(dist_nearest),
        mean_dist_com = mean(dist_com),
        lags = lags .* dt,
        xcorr_x, xcorr_y,
        best_lag_x = lags[ibx] * dt, best_corr_x = xcorr_x[ibx],
        best_lag_y = lags[iby] * dt, best_corr_y = xcorr_y[iby],
        zero_lag_corr_x = xcorr_x[izero], zero_lag_corr_y = xcorr_y[izero],
    )
end

function _hist2d(xs::AbstractVector, ys::AbstractVector, xedges::AbstractRange, yedges::AbstractRange)
    nx, ny = length(xedges) - 1, length(yedges) - 1
    counts = zeros(Int, nx, ny)
    xlo, xhi = first(xedges), last(xedges)
    ylo, yhi = first(yedges), last(yedges)
    dx, dy = step(xedges), step(yedges)
    @inbounds for k in eachindex(xs)
        x, y = xs[k], ys[k]
        (xlo <= x < xhi && ylo <= y < yhi) || continue
        i = clamp(floor(Int, (x - xlo) / dx) + 1, 1, nx)
        j = clamp(floor(Int, (y - ylo) / dy) + 1, 1, ny)
        counts[i, j] += 1
    end
    return counts
end

"""
    plot_predator_interaction_density(states, pred_hist; bins=60, center=(0.0,0.0), half_extent=nothing)

Figure 1 of the interaction report: side-by-side 2D occupancy heatmaps of
where the swarm spent its time (every agent, every timestep) vs. where the
predator spent its time, on identical, linked axes -- the direct visual
answer to "how much do their footprints even overlap", independent of any
single frame you might eyeball in an animation.

`half_extent` defaults to just covering both point clouds; pass e.g.
`half_extent=P.plot_extent, center=(0.0,0.0)` for a Lymburn view matching
`animate_from_states`'s `center=(-P.plot_extent,-P.plot_extent)` window.
"""
function plot_predator_interaction_density(states, pred_hist;
    bins::Int = 60,
    center::Tuple{<:Real,<:Real} = (0.0, 0.0),
    half_extent::Union{Nothing,Real} = nothing,
)
    T = length(states)
    N = size(states[1].P, 2)

    allx = Vector{Float64}(undef, N * T)
    ally = Vector{Float64}(undef, N * T)
    idx = 1
    for t in 1:T
        Pm = states[t].P
        @views allx[idx:idx+N-1] .= Pm[1, :]
        @views ally[idx:idx+N-1] .= Pm[2, :]
        idx += N
    end
    px = [pred_hist[t][1] for t in 1:T]
    py = [pred_hist[t][2] for t in 1:T]

    he = half_extent === nothing ?
        1.05 * max(maximum(abs.(allx .- center[1])), maximum(abs.(ally .- center[2])),
                   maximum(abs.(px .- center[1])), maximum(abs.(py .- center[2]))) :
        Float64(half_extent)

    xedges = range(center[1] - he, center[1] + he; length = bins + 1)
    yedges = range(center[2] - he, center[2] + he; length = bins + 1)
    xc = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    yc = (yedges[1:end-1] .+ yedges[2:end]) ./ 2

    swarm_counts = _hist2d(allx, ally, xedges, yedges)
    pred_counts = _hist2d(px, py, xedges, yedges)

    fig = Figure(size = (900, 420))
    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Swarm occupancy (all agents, all t)", aspect = DataAspect())
    hm1 = heatmap!(ax1, xc, yc, swarm_counts; colormap = :viridis)
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3], xlabel = "x", ylabel = "y", title = "Predator occupancy", aspect = DataAspect())
    hm2 = heatmap!(ax2, xc, yc, pred_counts; colormap = :viridis)
    Colorbar(fig[1, 4], hm2)

    linkaxes!(ax1, ax2)
    return fig
end

"""
    plot_predator_interaction_timeseries(report; dt=1.0)

Figure 2 of the interaction report, built from a
`predator_interaction_report(...)` result: contact fraction and
predator-to-swarm distance over time (left column), and the lagged
predator-vs-CoM cross-correlation curves per axis (right) -- where the
correlation peak actually sits tells you whether any distance/contact
signal above is a real response or an incidental one.
"""
function plot_predator_interaction_timeseries(report; dt::Real = 1.0)
    T = length(report.contact_fraction)
    t = (0:T-1) .* dt

    fig = Figure(size = (950, 480))

    ax1 = Axis(fig[1, 1], xlabel = "t", ylabel = "contact fraction", title = "Contact fraction (agents within rp)")
    lines!(ax1, t, report.contact_fraction; color = :firebrick)

    ax2 = Axis(fig[2, 1], xlabel = "t", ylabel = "distance", title = "Predator distance: nearest agent / centre of mass")
    lines!(ax2, t, report.dist_nearest; label = "nearest agent", color = :steelblue)
    lines!(ax2, t, report.dist_com; label = "centre of mass", color = :firebrick)
    axislegend(ax2)

    ax3 = Axis(fig[1:2, 2], xlabel = "lag (time units)", ylabel = "correlation",
        title = "Predator-CoM cross-correlation\n(peak far from lag=0 ⟹ likely incidental, not a response)")
    lines!(ax3, report.lags, report.xcorr_x; label = "x", color = :steelblue)
    lines!(ax3, report.lags, report.xcorr_y; label = "y", color = :firebrick)
    vlines!(ax3, [0.0]; color = :gray, linestyle = :dash)
    axislegend(ax3)

    return fig
end

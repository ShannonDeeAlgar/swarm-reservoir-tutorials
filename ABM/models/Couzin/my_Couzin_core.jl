# my_Couzin_core.jl
#
# Couzin 2002 topological social-force model: parameters, presets, and
# scenario construction.
#
# This file assumes the generic ABM machinery already provides:
#   - NoiseMode, NoNoise, WrappedStepGaussian, Diffusion
#   - SimulationConfig
#   - mean_order_parameters
#   - wrap_angle

using Random
using Statistics
using ProgressMeter
ProgressMeter.ijulia_behavior(:clear)


# ============================================================
# CouzinNoise
# ============================================================

Base.@kwdef struct CouzinNoise
    mode::NoiseMode = NoNoise
    s::Float64      = 0.0
    Dθ::Float64     = 0.0
end

@inline function sample_noise_angle(noise::CouzinNoise, dt::Float64, rng::AbstractRNG)
    noise.mode == NoNoise && return 0.0

    if noise.mode == WrappedStepGaussian
        return wrap_angle(randn(rng) * noise.s * sqrt(dt / 0.1)) #Weiner process scaling
    elseif noise.mode == Diffusion
        return randn(rng) * sqrt(noise.Dθ * dt)
    else
        error("Unknown noise mode $(noise.mode).")
    end
end

function update_CouzinNoise(
    noise::CouzinNoise;
    mode::NoiseMode = noise.mode,
    s::Float64      = noise.s,
    Dθ::Float64     = noise.Dθ,
)
    return CouzinNoise(mode=mode, s=s, Dθ=Dθ)
end


# ============================================================
# CouzinParams
# ============================================================

Base.@kwdef struct CouzinParams
    N::Int
    Zr::Float64
    Zo::Float64
    Za::Float64
    blind_half_angle::Float64 = 0.0   # radians; 0 = no blind zone

    L::Float64                        # domain size

    speed::Float64
    max_turn_rate::Float64

    noise::CouzinNoise = CouzinNoise()

    # bookkeeping
    is_nondim::Bool  = false
    Zr_ref::Float64  = 1.0
    v_ref::Float64   = 1.0
end

function update_CouzinParams(
    P::CouzinParams;
    N::Int                    = P.N,
    Zr::Float64               = P.Zr,
    Zo::Float64               = P.Zo,
    Za::Float64               = P.Za,
    blind_half_angle::Float64 = P.blind_half_angle,
    L::Float64                = P.L,
    speed::Float64            = P.speed,
    max_turn_rate::Float64    = P.max_turn_rate,
    noise::CouzinNoise        = P.noise,
    is_nondim::Bool           = P.is_nondim,
    Zr_ref::Float64           = P.Zr_ref,
    v_ref::Float64            = P.v_ref,
)
    return CouzinParams(
        N=N, Zr=Zr, Zo=Zo, Za=Za,
        blind_half_angle=blind_half_angle,
        L=L,
        speed=speed, max_turn_rate=max_turn_rate,
        noise=noise,
        is_nondim=is_nondim, Zr_ref=Zr_ref, v_ref=v_ref,
    )
end


# ============================================================
# CouzinScenario
# ============================================================
#
# Wraps CouzinParams with the preset name, mirroring HelbingScenario.
# init_mode is not needed here — Couzin always initialises uniformly.
# The preset field is useful for plot/animation labels and for
# generate_validated_preset to know which acceptance criterion to apply.

struct CouzinScenario
    P      :: CouzinParams
    preset :: Symbol
    dt     :: Float64               # effective dt after nondim / rescale transforms

    # transform bookkeeping
    nondim      :: Bool             # was nondimensionalise applied?
    scale_info  :: Union{Nothing, NamedTuple}  # records s, g, f if rescaling was applied
end


# ============================================================
# Baseline params  (Couzin 2002 paper values)
# ============================================================

function Couzin2002_base_params(; L::Float64 = 100.0)
    return CouzinParams(
        N                = 100,
        Zr               = 1.0,
        Zo               = 2.0,
        Za               = 16.0,
        blind_half_angle = deg2rad(45.0),
        L                = L,
        speed            = 3.0,
        max_turn_rate    = deg2rad(40.0),
        noise            = CouzinNoise(mode=WrappedStepGaussian, s=0.05),
        is_nondim        = false,
    )
end


# ============================================================
# COUZIN_REGIME_PRESETS
# ============================================================
#
# Each entry overrides only the zone radii that distinguish the regime. All
# other parameters come from Couzin2002_base_params. Values (except
# :fragmented, untested) are empirically validated in 2D: each passes
# preset_accepts under the unbounded simulate_Couzin2002_2d at N=100, with
# paper-scale settling (steps=5000, transient_steps=4000). 3D uses the same
# (Zr, Zo, Za) with COUZIN_3D_OVERRIDES applied on top (see below).

const COUZIN_REGIME_PRESETS = Dict(

    # Low cohesion: agents interact only weakly; no coherent collective motion.
    :swarm => (
        Zr = 1.0,
        Zo = 1.1,
        Za = 16.0,
    ),

    # Toroidal milling: high rotation order, low polarisation.
    :milling => (
        Zr = 1.0,
        Zo = 5.0,
        Za = 15.0,
    ),

    # Polarised schooling: aligned parallel motion, but fluid -- density and
    # neighbours keep changing (Couzin 2002's own distinction from :highly_parallel).
    # Validated under periodic=true, L=150 (2D) -- see preset_accepts' comment
    # on this preset for why, unlike the other three regimes here.
    :dynamic_parallel => (
        Zr = 1.0,
        Zo = 7.0,
        Za = 13.0,
    ),

    # Highly polarised schooling: very tight alignment.
    :highly_parallel => (
        Zr = 1.0,
        Zo = 11.0,
        Za = 21.0,
    ),

    # Fragmented: small attraction zone; group breaks apart.
    :fragmented => (
        Zr = 1.0,
        Zo = 1.5,
        Za = 4.0,
    ),
)


# ============================================================
# 3D parameter overrides
# ============================================================
#
# In 3D the orientation zone is a sphere, not a disc, which makes alignment
# much stronger than in 2D at the same (Zo, Za): a sphere holds ~10x the
# neighbours of the equivalent disc at the same density, and the 45° blind
# cone blocks a smaller fraction of a sphere (~15%) than of a circle (~25%).
# A 2D-validated (Zo, Za) pair can therefore land in the wrong 3D regime
# entirely. Each entry below is an independently re-validated (Zo, Za) pair
# for 3D (passes preset_accepts, N=100, steps=5000, transient_steps=4000)
# rather than a correction formula applied to the 2D value — `nothing` means
# the 2D value already works. :swarm/:milling/:highly_parallel are validated
# under simulate_Couzin2002_3d (unbounded, periodic=false); :dynamic_parallel
# is validated under simulate_Couzin_3d with periodic=true, L=75 -- see
# preset_accepts' comment on that preset for why it alone needs periodic
# boundaries to stay bounded rather than disperse.
#
# Applied automatically when dim=3 is passed to Couzin_params_from_preset.
# User kwargs passed to that function can still override any of these.

struct _3DOverride
    Zo    :: Union{Nothing, Float64}
    Za    :: Union{Nothing, Float64}
    noise :: Union{Nothing, CouzinNoise}
end
_3DOverride(; Zo=nothing, Za=nothing, noise=nothing) = _3DOverride(Zo, Za, noise)

const COUZIN_3D_OVERRIDES = Dict{Symbol, _3DOverride}(

    # Swarm: 2D value already validates in 3D.
    :swarm => _3DOverride(),

    # Milling (torus): 2D's small Zo mills too tightly in 3D at Za=15 unless
    # Zo also grows a little to keep the rotating shell from collapsing to a
    # rigid parallel lock.
    :milling => _3DOverride(Zo = 4.0, Za = 15.0),

    # Dynamic parallel: 3D's stronger alignment needs a substantially smaller
    # Zo than 2D to keep temporal fluctuations (density/position exchange
    # within the group) rather than locking rigid. Validated with periodic=true
    # (see preset_accepts) -- pol settles ~0.78-0.87, group radius bounded
    # (~6-12) rather than growing, over a 12000-step run.
    :dynamic_parallel => _3DOverride(Zo = 3.0, Za = 7.0),

    # Highly parallel: needs a larger Za in 3D to stay a single cohesive
    # group at this Zo rather than fragmenting.
    :highly_parallel => _3DOverride(Zo = 15.0, Za = 19.0),

    # Fragmented: no structural change; small Za already limits cohesion.
    :fragmented => _3DOverride(),
)


# ============================================================
# Couzin_params_from_preset  →  CouzinScenario
# ============================================================

function Couzin_params_from_preset(
    preset::Symbol;
    dim::Int     = 2,
    L::Float64   = 100.0,
    dt::Float64  = 0.1,            # physical dt — will be transformed if nondim/rescale
    nondim::Bool = false,
    rescale::Union{Nothing, NamedTuple} = nothing,
    kwargs...,
)
    canonical = if preset == :swarming
        :swarm
    elseif preset == :polarised || preset == :polarized
        :dynamic_parallel
    else
        preset
    end

    haskey(COUZIN_REGIME_PRESETS, canonical) || error(
        "Unknown Couzin preset = $preset. " *
        "Valid presets are $(sort(collect(keys(COUZIN_REGIME_PRESETS))))."
    )

    # In 3D, default L=100 gives near-zero density.
    # Use L=50 so the expected number of Zo-neighbours is comparable to 2D.
    L_use = (dim == 3 && L == 100.0) ? 50.0 : L

    vals = COUZIN_REGIME_PRESETS[canonical]
    P0   = Couzin2002_base_params(; L=L_use)
    P    = update_CouzinParams(P0; Zr=vals.Zr, Zo=vals.Zo, Za=vals.Za)

    # Apply 3D overrides (Zo/Za/noise) before user kwargs so that
    # the user can still override any of these explicitly.
    if dim == 3 && haskey(COUZIN_3D_OVERRIDES, canonical)
        ov = COUZIN_3D_OVERRIDES[canonical]
        ov.Zo    !== nothing && (P = update_CouzinParams(P; Zo    = ov.Zo))
        ov.Za    !== nothing && (P = update_CouzinParams(P; Za    = ov.Za))
        ov.noise !== nothing && (P = update_CouzinParams(P; noise = ov.noise))
    end

    P    = isempty(kwargs) ? P : update_CouzinParams(P; kwargs...)

    # Apply transforms in order: nondim first, then rescale.
    dt_out     = dt
    scale_info = nothing

    if nondim
        P, dt_out = nondimensionalise!(P, dt_out)
    end

    if rescale !== nothing
        type = get(rescale, :type, :none)

        P_pre_rescale  = P
        dt_pre_rescale = dt_out

        if type == :space
            s          = Float64(rescale.s)
            P, dt_out  = scale_space!(P, dt_out;
                s                   = s,
                preserve_density    = get(rescale, :preserve_density,    true),
                preserve_step_stats = get(rescale, :preserve_step_stats, true),
                warn_step           = get(rescale, :warn_step,           true),
            )
            scale_info = (type=:space, s=s)

        elseif type == :time
            f          = Float64(rescale.f)
            P, dt_out  = scale_time!(P, dt_out; f=f,
                warn_step  = get(rescale, :warn_step, true),
            )
            scale_info = (type=:time, f=f)

        elseif type == :dynamics
            g          = Float64(rescale.g)
            P, dt_out  = scale_swarm_dynamics!(P, dt_out; g=g,
                warn_step  = get(rescale, :warn_step, true),
            )
            scale_info = (type=:dynamics, g=g)

        else
            error("Unknown rescale type = $type. Use :space, :time, or :dynamics.")
        end

        _print_rescale_summary(scale_info, P_pre_rescale, dt_pre_rescale, P, dt_out)
    end

    return CouzinScenario(P, canonical, dt_out, nondim, scale_info)
end


# ============================================================
# Public entry points
# ============================================================

# Primary notebook entry point — returns a CouzinScenario.
function default_Couzin_scenario(
    preset::Symbol = :milling;
    dim::Int     = 2,
    dt::Float64  = 0.1,
    nondim::Bool = false,
    rescale::Union{Nothing, NamedTuple} = nothing,
    kwargs...,
)
    return Couzin_params_from_preset(preset;
        dim     = dim,
        dt      = dt,
        nondim  = nondim,
        rescale = rescale,
        kwargs...,
    )
end

# Backwards-compatible alias — returns bare CouzinParams.
function default_Couzin_params(; preset::Symbol = :base, dt::Float64 = 0.1, kwargs...)
    if preset == :base
        P0 = Couzin2002_base_params()
        return isempty(kwargs) ? P0 : update_CouzinParams(P0; kwargs...)
    else
        return Couzin_params_from_preset(preset; dt=dt, kwargs...).P
    end
end


# ============================================================
# Preset acceptance criteria  (used by generate_validated_preset)
# ============================================================

function preset_accepts(preset::Symbol, stats)
    pol  = stats.polarisation_mean
    rot  = abs(stats.rotation_mean)
    dil  = abs(stats.dilation_mean)

    if preset == :swarm
        return pol ≤ 0.35 && rot ≤ 0.35 && dil ≤ 0.25

    elseif preset == :milling
        return rot ≥ 0.60 && pol ≤ 0.35 && dil ≤ 0.25

    elseif preset == :dynamic_parallel
        # Under the unbounded (periodic=false) reproduction, every
        # intermediate-polarisation point found empirically was unbounded --
        # confirmed via actual group radius (not just this dilation proxy):
        # several 0.5-0.9-polarisation candidates grew 20-30x over 5000 steps
        # rather than settling, i.e. genuine dispersal toward fragmentation,
        # not the density/neighbour fluidity Couzin (2002) describes. Under
        # periodic boundaries the group stays bounded and settles to
        # pol~0.92-0.94 (below :highly_parallel's ~0.998) with decaying but
        # nonzero rotation -- hence :dynamic_parallel is validated with
        # periodic=true (see COUZIN_REGIME_PRESETS/COUZIN_3D_OVERRIDES),
        # unlike the other three regimes, which stay bounded either way.
        # Ceiling kept below 0.95 so this never overlaps :highly_parallel.
        return 0.70 ≤ pol ≤ 0.94 && rot ≤ 0.20 && dil ≤ 0.20

    elseif preset == :highly_parallel
        return pol ≥ 0.95 && rot ≤ 0.10 && dil ≤ 0.15

    elseif preset == :fragmented
        return pol ≤ 0.30 && rot ≤ 0.20

    else
        error("Unknown preset for acceptance test: $preset.")
    end
end


# ============================================================
# Validated preset generation
# ============================================================
#
# Runs the simulator up to `tries` times with different seeds until the
# time-averaged order parameters satisfy the acceptance criterion for
# the requested preset.  Returns a NamedTuple with the accepted run, or
# a failure record if no seed passed within the attempt budget.

function generate_validated_preset(
    simulator        :: Function,
    simcfg           :: SimulationConfig,
    preset           :: Symbol;
    tries            :: Int    = 20,
    transient_steps  :: Int    = 200,
    L                :: Float64 = 50.0,
    collect_order    :: Bool   = true,
    collect_history  :: Bool   = true,
    show_progress    :: Bool   = true,
    seed0            :: Int    = simcfg.seed,
    param_kwargs             = NamedTuple(),
    sim_kwargs...,
)
    for k in 0:(tries - 1)
        seed = seed0 + k

        simcfg_try = SimulationConfig(
            steps         = simcfg.steps,
            dt            = simcfg.dt,
            seed          = seed,
            focal         = simcfg.focal,
            update_scheme = simcfg.update_scheme,
        )

        scenario = Couzin_params_from_preset(preset; L=L, param_kwargs...)

        # `show_progress` above controls only the per-trial acceptance diagnostics
        # printed below, not the simulator's own per-step bar -- simulate_Couzin2002_2d/3d
        # default that to true independently, so without this every one of up to `tries`
        # inner runs would spawn its own Progress bar. Default it off here (still
        # overridable via sim_kwargs), same fix as run_Couzin_once in
        # my_Couzin_experiments.jl.
        sim_kwargs = merge((show_progress = false,), NamedTuple(sim_kwargs))

        # The simulator expects bare CouzinParams, not CouzinScenario.
        out = simulator(
            simcfg_try,
            scenario.P;
            collect_order   = collect_order,
            collect_history = collect_history,
            sim_kwargs...,
        )

        stats    = mean_order_parameters(out; transient_steps=transient_steps)
        accepted = preset_accepts(preset, stats)

        if show_progress
            print_preset_trial_diagnostics(
                preset, stats;
                accepted = accepted,
                trial    = k + 1,
                tries    = tries,
                seed     = seed,
            )
        end

        if accepted
            return (
                accepted = true,
                preset   = preset,
                trial    = k + 1,
                seed     = seed,
                scenario = scenario,
                simcfg   = simcfg_try,
                out      = out,
                stats    = stats,
            )
        end
    end

    return (
        accepted = false,
        preset   = preset,
        trial    = tries,
        seed     = seed0 + tries - 1,
        scenario = nothing,
        simcfg   = nothing,
        out      = nothing,
        stats    = nothing,
    )
end


# ============================================================
# Trial diagnostics printer
# ============================================================

function print_preset_trial_diagnostics(
    preset   :: Symbol,
    stats;
    accepted :: Bool,
    trial    :: Int,
    tries    :: Int,
    seed     :: Int,
)
    pol  = stats.polarisation_mean
    rot  = stats.rotation_mean
    arot = abs(rot)
    dil  = stats.dilation_mean
    adil = abs(dil)

    println("preset=$preset  trial=$trial/$tries  seed=$seed  accepted=$accepted")
    println("  pol=$(round(pol,digits=4))  rot=$(round(rot,digits=4))  " *
            "|rot|=$(round(arot,digits=4))  dil=$(round(dil,digits=4))  " *
            "|dil|=$(round(adil,digits=4))")

    if !accepted
        if preset == :swarm
            println("  checks: pol≤0.35=$(pol≤0.35)  |rot|≤0.35=$(arot≤0.35)  |dil|≤0.25=$(adil≤0.25)")
        elseif preset == :milling
            println("  checks: |rot|≥0.60=$(arot≥0.60)  pol≤0.35=$(pol≤0.35)  |dil|≤0.25=$(adil≤0.25)")
        elseif preset == :dynamic_parallel
            println("  checks: pol≥0.75=$(pol≥0.75)  |rot|≤0.20=$(arot≤0.20)  |dil|≤0.20=$(adil≤0.20)")
        elseif preset == :highly_parallel
            println("  checks: pol≥0.95=$(pol≥0.95)  |rot|≤0.10=$(arot≤0.10)  |dil|≤0.15=$(adil≤0.15)")
        elseif preset == :fragmented
            println("  checks: pol≤0.30=$(pol≤0.30)  |rot|≤0.20=$(arot≤0.20)")
        end
    end
end

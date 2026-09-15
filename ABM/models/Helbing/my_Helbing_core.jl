# my_Helbing_core.jl
#
# Helbing--Molnar social force model: parameters, presets, and scenario construction.
#
# This file assumes the generic ABM machinery already provides:
#   - SVector2, SVector3
#   - SwarmState{V}
#   - SimulationConfig
#   - simulate
#   - order_parameters
#   - displacement   (for periodic boundary displacement)
#
# Rules live in my_Helbing_Rules2d.jl and my_Helbing_Rules3d.jl.

using LinearAlgebra
using StaticArrays


# ============================================================
# HelbingParams
# ============================================================

struct HelbingParams{V}
    # Population and domain scale
    N::Int
    L::Float64
    # Individual motion
    mass::Float64
    radius::Float64
    v0::Float64
    tau::Float64
    max_speed::Float64
    # Agent-agent social force
    A_agent::Float64
    B_agent::Float64
    k_body::Float64
    k_friction::Float64
    # Wall force
    A_wall::Float64
    B_wall::Float64
    # Switches
    #   boundary:         :walls | :periodic | :none
    #   drive_mode:       :goal  | :heading  | :none
    #   interaction_mode: :social_force | :soft_repulsion | :contact_only | :none
    #   wall_mode:        :social_force | :hard_clamp | :none
    boundary::Symbol
    drive_mode::Symbol
    interaction_mode::Symbol
    wall_mode::Symbol
    # Per-agent targets (V = SVector2 or SVector3).
    # Nothing when drive_mode = :none or the field is unused.
    goals::Union{Nothing, Vector{V}}
    headings::Union{Nothing, Vector{V}}
end


# ============================================================
# HelbingScenario
# ============================================================
#
# Wraps HelbingParams with two fields the params struct cannot carry:
#   init_mode – how agents are placed at t = 0
#   preset    – originating preset name (useful for plot labels)

struct HelbingScenario
    P         :: HelbingParams
    init_mode :: Symbol
    preset    :: Symbol
    dt        :: Float64               # effective dt after nondim / rescale transforms

    # transform bookkeeping
    nondim      :: Bool
    scale_info  :: Union{Nothing, NamedTuple}
end


# ============================================================
# Validation helpers
# ============================================================

function validate_Helbing_symbols(
    boundary::Symbol,
    drive_mode::Symbol,
    interaction_mode::Symbol,
    wall_mode::Symbol,
)
    boundary in (:walls, :periodic, :none) ||
        error("boundary must be :walls, :periodic, or :none. Got $boundary.")

    drive_mode in (:goal, :heading, :none) ||
        error("drive_mode must be :goal, :heading, or :none. Got $drive_mode.")

    interaction_mode in (:social_force, :soft_repulsion, :contact_only, :none) ||
        error("interaction_mode must be :social_force, :soft_repulsion, :contact_only, or :none. Got $interaction_mode.")

    wall_mode in (:social_force, :hard_clamp, :none) ||
        error("wall_mode must be :social_force, :hard_clamp, or :none. Got $wall_mode.")

    if boundary != :walls && wall_mode != :none
        @warn "wall_mode = $wall_mode has no effect unless boundary = :walls."
    end

    return nothing
end


function _normalise_Helbing_vector(v)
    nv = norm(v)
    nv < 1e-12 && error("Cannot normalise a near-zero Helbing heading vector.")
    return v / nv
end


function normalise_Helbing_headings(headings, N::Int)
    headings === nothing && return nothing
    length(headings) == N ||
        error("length(headings) must equal N = $N, got $(length(headings)).")
    return [_normalise_Helbing_vector(h) for h in headings]
end


# ============================================================
# HelbingParams keyword constructor
# ============================================================

function HelbingParams(;
    N::Int         = 50,
    L::Float64     = 50.0,

    mass::Float64      = 80.0,
    radius::Float64    = 0.30,
    v0::Float64        = 1.34,
    tau::Float64       = 0.50,
    max_speed::Float64 = 1.3 * v0,

    A_agent::Float64    = 2.0e3,
    B_agent::Float64    = 0.08,
    k_body::Float64     = 1.2e5,
    k_friction::Float64 = 2.4e5,

    A_wall::Float64 = A_agent,
    B_wall::Float64 = B_agent,

    boundary::Symbol         = :walls,
    drive_mode::Symbol       = :goal,
    interaction_mode::Symbol = :social_force,
    wall_mode::Union{Nothing, Symbol} = nothing,

    goals    = nothing,
    headings = nothing,
)
    chosen_wall_mode = if wall_mode === nothing
        boundary == :walls ? :social_force : :none
    else
        wall_mode
    end

    validate_Helbing_symbols(boundary, drive_mode, interaction_mode, chosen_wall_mode)

    goals_vec = if goals !== nothing
        length(goals) == N ||
            error("length(goals) must equal N = $N, got $(length(goals)).")
        collect(goals)
    else
        nothing
    end

    headings_vec = if headings !== nothing
        normalise_Helbing_headings(headings, N)
    else
        nothing
    end

    # Infer vector type from goals or headings; fall back to SVector2.
    V = if goals_vec !== nothing
        eltype(goals_vec)
    elseif headings_vec !== nothing
        eltype(headings_vec)
    else
        SVector2
    end

    return HelbingParams{V}(
        N, L,
        mass, radius, v0, tau, max_speed,
        A_agent, B_agent, k_body, k_friction,
        A_wall, B_wall,
        boundary, drive_mode, interaction_mode, chosen_wall_mode,
        goals_vec, headings_vec,
    )
end


# ============================================================
# update_HelbingParams
# ============================================================

function update_HelbingParams(
    P::HelbingParams;
    N::Int         = P.N,
    L::Float64     = P.L,

    mass::Float64      = P.mass,
    radius::Float64    = P.radius,
    v0::Float64        = P.v0,
    tau::Float64       = P.tau,
    max_speed::Float64 = P.max_speed,

    A_agent::Float64    = P.A_agent,
    B_agent::Float64    = P.B_agent,
    k_body::Float64     = P.k_body,
    k_friction::Float64 = P.k_friction,

    A_wall::Float64 = P.A_wall,
    B_wall::Float64 = P.B_wall,

    boundary::Symbol         = P.boundary,
    drive_mode::Symbol       = P.drive_mode,
    interaction_mode::Symbol = P.interaction_mode,
    wall_mode::Symbol        = P.wall_mode,

    goals    = P.goals,
    headings = P.headings,
)
    return HelbingParams(
        N = N, L = L,
        mass = mass, radius = radius, v0 = v0, tau = tau, max_speed = max_speed,
        A_agent = A_agent, B_agent = B_agent, k_body = k_body, k_friction = k_friction,
        A_wall = A_wall, B_wall = B_wall,
        boundary = boundary, drive_mode = drive_mode,
        interaction_mode = interaction_mode, wall_mode = wall_mode,
        goals = goals, headings = headings,
    )
end


# ============================================================
# HELBING_REGIME_PRESETS
# ============================================================
#
# Every entry owns the complete scenario specification:
#   force params   – mass, radius, v0, tau, A_agent, …
#   boundary       – spatial boundary condition
#   drive_mode     – how the desired velocity is computed
#   wall_mode      – wall force treatment (only active when boundary = :walls)
#   interaction_mode – which agent-agent force terms are active
#   init_mode      – how agents are placed at t = 0

const HELBING_REGIME_PRESETS = Dict(

    # Paper-like baseline: single unidirectional group.
    :base => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 2.0e3,
        B_agent          = 0.08,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.0e3,
        B_wall           = 0.08,
        boundary         = :periodic,
        drive_mode       = :heading,
        wall_mode        = :none,
        interaction_mode = :social_force,
        init_mode        = :uniform,
    ),

    # Near-undisturbed walking: weak interactions, low density.
    :free_flow => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 5.0e2,
        B_agent          = 0.08,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 5.0e2,
        B_wall           = 0.08,
        boundary         = :periodic,
        drive_mode       = :heading,
        wall_mode        = :none,
        interaction_mode = :soft_repulsion,
        init_mode        = :uniform,
    ),

    # Repulsion-dominated avoidance: single group, moderate density.
    :avoidance => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 2.0e3,
        B_agent          = 0.08,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.0e3,
        B_wall           = 0.08,
        boundary         = :periodic,
        drive_mode       = :heading,
        wall_mode        = :none,
        interaction_mode = :social_force,
        init_mode        = :two_slabs,
    ),

    # Bidirectional corridor: lanes emerge above a critical density.
    :lane_formation => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 2.0e3,
        B_agent          = 0.20,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.0e3,
        B_wall           = 0.20,
        boundary         = :periodic,
        drive_mode       = :heading,
        wall_mode        = :none,
        interaction_mode = :social_force,
        init_mode        = :two_slabs,
    ),

    # Lane formation with periodic boundaries; lanes persist indefinitely.
    :stable_lanes => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 2.0e3,
        B_agent          = 0.20,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.0e3,
        B_wall           = 0.20,
        boundary         = :periodic,
        drive_mode       = :heading,
        wall_mode        = :none,
        interaction_mode = :social_force,
        init_mode        = :two_slabs,
    ),

    # Opposing groups through a narrow door; flow direction can oscillate.
    :bottleneck => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 2.0e3,
        B_agent          = 0.08,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.0e3,
        B_wall           = 0.08,
        boundary         = :walls,
        drive_mode       = :goal,
        wall_mode        = :social_force,
        interaction_mode = :social_force,
        init_mode        = :two_slabs,
    ),

    # More constrained bottleneck; stronger blocking, clearer alternation.
    :narrow_bottleneck => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.34,
        tau              = 0.50,
        max_speed_factor = 1.3,
        A_agent          = 2.5e3,
        B_agent          = 0.08,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.5e3,
        B_wall           = 0.08,
        boundary         = :walls,
        drive_mode       = :goal,
        wall_mode        = :social_force,
        interaction_mode = :social_force,
        init_mode        = :two_slabs,
    ),

    # Shorter relaxation time; sharper correction toward desired velocity.
    :aggressive => (
        L                = 50.0,
        mass             = 80.0,
        radius           = 0.30,
        v0               = 1.50,
        tau              = 0.25,
        max_speed_factor = 1.3,
        A_agent          = 2.0e3,
        B_agent          = 0.08,
        k_body           = 1.2e5,
        k_friction       = 2.4e5,
        A_wall           = 2.0e3,
        B_wall           = 0.08,
        boundary         = :periodic,
        drive_mode       = :heading,
        wall_mode        = :none,
        interaction_mode = :social_force,
        init_mode        = :two_slabs,
    ),
)


# ============================================================
# Goal and heading helpers  (dim-aware)
# ============================================================

function make_Helbing_goals(preset::Symbol, N::Int, L::Float64; dim::Int = 2)
    dim in (2, 3) || error("dim must be 2 or 3, got $dim.")

    if dim == 2
        if preset in (:lane_formation, :stable_lanes, :bottleneck, :narrow_bottleneck, :aggressive)
            left_goal  = SVector2(0.0, L / 2)
            right_goal = SVector2(L,   L / 2)
            return [i <= N ÷ 2 ? right_goal : left_goal for i in 1:N]
        else
            return fill(SVector2(L, L / 2), N)
        end
    else
        if preset in (:lane_formation, :stable_lanes, :bottleneck, :narrow_bottleneck, :aggressive)
            left_goal  = SVector3(0.0, L / 2, L / 2)
            right_goal = SVector3(L,   L / 2, L / 2)
            return [i <= N ÷ 2 ? right_goal : left_goal for i in 1:N]
        else
            return fill(SVector3(L, L / 2, L / 2), N)
        end
    end
end


function default_Helbing_headings(preset::Symbol, N::Int; dim::Int = 2)
    dim in (2, 3) || error("dim must be 2 or 3, got $dim.")

    if dim == 2
        if preset in (:lane_formation, :stable_lanes, :bottleneck, :narrow_bottleneck, :aggressive)
            return [i <= N ÷ 2 ? SVector2(1.0, 0.0) : SVector2(-1.0, 0.0) for i in 1:N]
        else
            return fill(SVector2(1.0, 0.0), N)
        end
    else
        if preset in (:lane_formation, :stable_lanes, :bottleneck, :narrow_bottleneck, :aggressive)
            return [i <= N ÷ 2 ? SVector3(1.0, 0.0, 0.0) : SVector3(-1.0, 0.0, 0.0) for i in 1:N]
        else
            return fill(SVector3(1.0, 0.0, 0.0), N)
        end
    end
end


# ============================================================
# Helbing_params_from_preset  →  HelbingScenario
# ============================================================

function Helbing_params_from_preset(
    preset::Symbol;
    N::Int                              = 100,
    dim::Int                            = 2,
    dt::Float64                         = 0.1,

    nondim::Bool                        = false,
    rescale::Union{Nothing, NamedTuple} = nothing,

    L_override::Union{Nothing, Float64} = nothing,

    # Scenario-level overrides — all default to the preset's own value.
    boundary::Union{Nothing, Symbol}         = nothing,
    drive_mode::Union{Nothing, Symbol}       = nothing,
    interaction_mode::Union{Nothing, Symbol} = nothing,
    wall_mode::Union{Nothing, Symbol}        = nothing,
    init_mode::Union{Nothing, Symbol}        = nothing,

    goals    = nothing,
    headings = nothing,

    kwargs...,   # scalar param overrides (v0, tau, A_agent, …) → update_HelbingParams
)
    dim in (2, 3) || error("dim must be 2 or 3, got $dim.")

    haskey(HELBING_REGIME_PRESETS, preset) || error(
        "Unknown Helbing preset = $preset. " *
        "Valid presets are $(sort(collect(keys(HELBING_REGIME_PRESETS))))."
    )

    vals = HELBING_REGIME_PRESETS[preset]

    # Caller wins; otherwise use the preset default.
    L               = L_override       === nothing ? vals.L                : L_override
    chosen_boundary = boundary         === nothing ? vals.boundary         : boundary
    chosen_drive    = drive_mode       === nothing ? vals.drive_mode       : drive_mode
    chosen_imode    = interaction_mode === nothing ? vals.interaction_mode : interaction_mode
    chosen_init     = init_mode        === nothing ? vals.init_mode        : init_mode

    # Presets store one init_mode value, but init_state_Helbing_2d and
    # init_state_Helbing_3d use different names for the equivalent "two
    # groups colliding head-on" layout (:left_strip in 2D, :two_slabs in
    # 3D) -- most presets here were written with the 3D name, which
    # init_state_Helbing_2d doesn't recognise at all. Remap only when the
    # caller didn't explicitly ask for a specific init_mode.
    if init_mode === nothing && dim == 2 && chosen_init == :two_slabs
        chosen_init = :left_strip
    end

    chosen_wall = if wall_mode !== nothing
        wall_mode
    elseif chosen_boundary == :walls
        vals.wall_mode
    else
        :none
    end

    goals_vec = if goals !== nothing
        goals
    elseif chosen_drive == :goal
        make_Helbing_goals(preset, N, L; dim = dim)
    else
        nothing
    end

    headings_vec = if headings !== nothing
        headings
    elseif chosen_drive == :heading
        default_Helbing_headings(preset, N; dim = dim)
    else
        nothing
    end

    P = HelbingParams(
        N             = N,
        L             = L,
        mass          = vals.mass,
        radius        = vals.radius,
        v0            = vals.v0,
        tau           = vals.tau,
        max_speed     = vals.max_speed_factor * vals.v0,
        A_agent       = vals.A_agent,
        B_agent       = vals.B_agent,
        k_body        = vals.k_body,
        k_friction    = vals.k_friction,
        A_wall        = vals.A_wall,
        B_wall        = vals.B_wall,
        boundary      = chosen_boundary,
        drive_mode    = chosen_drive,
        interaction_mode = chosen_imode,
        wall_mode     = chosen_wall,
        goals         = goals_vec,
        headings      = headings_vec,
    )

    P_final = isempty(kwargs) ? P : update_HelbingParams(P; kwargs...)

    # Transform block — Helbing scaling not yet implemented
    dt_out     = dt
    scale_info = nothing

    if nondim
        error("Helbing nondimensionalisation is not yet implemented. " *
              "Set nondim=false or implement nondimensionalise_Helbing!.")
    end

    if rescale !== nothing
        error("Helbing rescaling is not yet implemented. " *
              "Set rescale=nothing or implement scale_Helbing functions.")
    end

    return HelbingScenario(P_final, chosen_init, preset, dt_out, nondim, scale_info)

end


# ============================================================
# Public entry points
# ============================================================

# Primary notebook entry point — returns a HelbingScenario.
function default_Helbing_scenario(
    preset::Symbol = :base;
    dt::Float64  = 0.1,
    nondim::Bool = false,
    rescale::Union{Nothing, NamedTuple} = nothing,
    kwargs...,
)
    return Helbing_params_from_preset(preset;
        dt      = dt,
        nondim  = nondim,
        rescale = rescale,
        kwargs...,
    )
end

# Backwards-compatible alias — returns the bare HelbingParams.
# Existing code that passes P directly to rules functions keeps working.
function default_Helbing_params(; preset::Symbol = :base, kwargs...)
    return Helbing_params_from_preset(preset; kwargs...).P
end

# Older alias kept for compatibility.
Helbing1995_base_params(; kwargs...) =
    Helbing_params_from_preset(:base; kwargs...).P
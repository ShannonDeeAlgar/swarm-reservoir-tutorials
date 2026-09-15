# ============================================================
# my_ABM_core.jl
# Generic ABM simulation core
# ============================================================
#
# This file defines the generic state/config/output types and the simulation
# driver. It deliberately knows nothing about Couzin, Vicsek, Boids, Helbing,
# or any other specific model.
#
# Models plug into `simulate` by supplying:
#   - init_state
#   - step!
#   - get_pos / get_vel
#   - domain_size
#   - displacement
#   - order_parameters
#
# The core can optionally collect:
#   - position/velocity history
#   - global order-parameter time series
#   - mean agent speed time series
#
# ============================================================

# SwarmState
# copy_state

# SimulationConfig
# SimulationOutput

# has_history
# has_order
# has_speed

# simulate
# thin_history


using Random
using LinearAlgebra
using ProgressMeter


# ============================================================
# 1. Generic state container
# ============================================================

mutable struct SwarmState{V}
    pos::Vector{V}
    vel::Vector{V}
    desired::Vector{V}
end


function copy_state(state::SwarmState{V}) where {V}
    return SwarmState{V}(
        copy(state.pos),
        copy(state.vel),
        copy(state.desired),
    )
end


# ============================================================
# 2. Simulation configuration
# ============================================================

Base.@kwdef struct SimulationConfig
    steps::Int          = 400
    dt::Float64         = 0.1
    seed::Int           = 1
    focal::Int          = 1
    update_scheme::Symbol = :synchronous   # :synchronous or :asynchronous
end


# ============================================================
# 3. Simulation output
# ============================================================

struct SimulationOutput{P,V}
    simcfg::SimulationConfig
    params::P
    focal::Int

    t::Vector{Float32}

    dilation::Vector{Float32}
    rotation::Vector{Float32}
    polarisation::Vector{Float32}
    abs_angular_momentum::Vector{Float32}
    speed::Vector{Float32}

    pos_hist::Vector{Vector{V}}
    vel_hist::Vector{Vector{V}}
end


has_history(out) = !isempty(out.pos_hist) && !isempty(out.vel_hist)
has_order(out)   = !isempty(out.dilation)
has_speed(out)   = hasproperty(out, :speed) && !isempty(out.speed)


# ============================================================
# 4. Generic speed diagnostic
# ------------------------------------------------------------
# This is deliberately model-agnostic: it only assumes velocities
# have a norm.
# ============================================================

function mean_agent_speed(vel)
    isempty(vel) && return 0.0

    s = 0.0

    @inbounds for v in vel
        s += norm(v)
    end

    return s / length(vel)
end


# ============================================================
# 5. Generic simulation driver
# ============================================================

function simulate(simcfg::SimulationConfig, P;
    init_state,
    step!,
    get_pos,
    get_vel,
    domain_size,
    displacement,
    order_parameters,
    collect_history::Bool = true,
    collect_order::Bool = true,
    dt::Float64 = simcfg.dt,
    show_progress::Bool = true,
)
    rng   = MersenneTwister(simcfg.seed)
    state = init_state(P; rng = rng)

    pos0 = get_pos(state)
    vel0 = get_vel(state)

    V = eltype(pos0)

    N = length(pos0)
    focal = clamp(simcfg.focal, 1, max(N, 1))

    t = collect(Float32, 0:dt:(simcfg.steps * dt))

    dilation = collect_order ?
        zeros(Float32, simcfg.steps + 1) :
        Float32[]

    rotation = collect_order ?
        zeros(Float32, simcfg.steps + 1) :
        Float32[]

    polarisation = collect_order ?
        zeros(Float32, simcfg.steps + 1) :
        Float32[]

    abs_angular_momentum = collect_order ?
        zeros(Float32, simcfg.steps + 1) :
        Float32[]

    speed = collect_order ?
        zeros(Float32, simcfg.steps + 1) :
        Float32[]

    pos_hist = collect_history ?
        Vector{Vector{V}}(undef, simcfg.steps + 1) :
        Vector{Vector{V}}()

    vel_hist = collect_history ?
        Vector{Vector{V}}(undef, simcfg.steps + 1) :
        Vector{Vector{V}}()


    function snapshot!(k::Int)
        pos = get_pos(state)
        vel = get_vel(state)

        if collect_history
            pos_hist[k] = copy(pos)
            vel_hist[k] = copy(vel)
        end

        if collect_order
            L = domain_size(P, state)

            dil, rot, pol, mabs = order_parameters(
                pos,
                vel,
                L;
                displacement = displacement,
            )

            dilation[k]             = Float32(dil)
            rotation[k]             = Float32(rot)
            polarisation[k]         = Float32(pol)
            abs_angular_momentum[k] = Float32(mabs)
            speed[k]                = Float32(mean_agent_speed(vel))
        end

        return nothing
    end


    snapshot!(1)

    prog = show_progress ?
        Progress(simcfg.steps; desc = "simulate", showspeed = true) :
        nothing

    for k in 1:simcfg.steps
        step!(state, P, dt; rng = rng, update_scheme = simcfg.update_scheme)
        snapshot!(k + 1)
        show_progress && next!(prog)
    end

    return SimulationOutput{typeof(P),V}(
        simcfg,
        P,
        focal,
        t,
        dilation,
        rotation,
        polarisation,
        abs_angular_momentum,
        speed,
        pos_hist,
        vel_hist,
    )
end


# ============================================================
# 6. Thin stored history/time series
# ============================================================

function thin_history(out; stride::Int = 5)
    keep = 1:stride:length(out.t)

    return SimulationOutput(
        out.simcfg,
        out.params,
        out.focal,
        out.t[keep],

        isempty(out.dilation) ?
            Float32[] :
            out.dilation[keep],

        isempty(out.rotation) ?
            Float32[] :
            out.rotation[keep],

        isempty(out.polarisation) ?
            Float32[] :
            out.polarisation[keep],

        isempty(out.abs_angular_momentum) ?
            Float32[] :
            out.abs_angular_momentum[keep],

        isempty(out.speed) ?
            Float32[] :
            out.speed[keep],

        isempty(out.pos_hist) ?
            typeof(out.pos_hist)() :
            out.pos_hist[keep],

        isempty(out.vel_hist) ?
            typeof(out.vel_hist)() :
            out.vel_hist[keep],
    )
end
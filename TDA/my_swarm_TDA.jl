# Common swarm-state adapter for topological data analysis.
# This file deliberately has no dependency on a particular ABM or reservoir type.

using Statistics
using LinearAlgebra

"A model-independent snapshot of the agents visible to the TDA layer."
struct SwarmSnapshot
    positions::Matrix{Float64}                 # N × D
    velocities::Union{Nothing,Matrix{Float64}} # N × D
    agent_state::Union{Nothing,Vector}
    time::Union{Nothing,Float64}
    frame::Union{Nothing,Int}
end

function _agents_by_dimension(X::AbstractMatrix; name::AbstractString, dimensions_first::Bool=false)
    A = Matrix{Float64}(X)
    # Reservoir raw_state uses D×N. A conventional point cloud uses N×D.
    if dimensions_first
        size(A, 1) in (2, 3) || throw(ArgumentError(
            "$name in D×N layout must have D=2 or 3; got $(size(A))."))
        return permutedims(A)
    elseif size(A, 1) in (2, 3) && size(A, 2) > size(A, 1)
        return permutedims(A)
    elseif size(A, 2) in (2, 3)
        return A
    end
    throw(ArgumentError("$name must be D×N or N×D with D=2 or 3; got $(size(A))."))
end

function _agents_by_dimension(xs::AbstractVector; name::AbstractString, dimensions_first::Bool=false)
    isempty(xs) && throw(ArgumentError("$name cannot be empty."))
    p = first(xs)
    D = hasproperty(p, :z) ? 3 : 2
    X = Matrix{Float64}(undef, length(xs), D)
    @inbounds for i in eachindex(xs)
        X[i, 1] = Float64(getproperty(xs[i], :x))
        X[i, 2] = Float64(getproperty(xs[i], :y))
        D == 3 && (X[i, 3] = Float64(getproperty(xs[i], :z)))
    end
    return X
end

function _snapshot_from_state(state; time=nothing, frame=nothing)
    dimensions_first = hasproperty(state, :P)
    pos_source = dimensions_first ? getproperty(state, :P) :
        (hasproperty(state, :pos) ? getproperty(state, :pos) : nothing)
    pos_source === nothing && throw(ArgumentError(
        "Cannot find agent positions. Expected `P` or `pos` in the supplied state."))
    positions = _agents_by_dimension(pos_source; name="positions", dimensions_first)

    vel_source = hasproperty(state, :V) ? getproperty(state, :V) :
        (hasproperty(state, :vel) ? getproperty(state, :vel) : nothing)
    velocities = vel_source === nothing ? nothing :
        _agents_by_dimension(vel_source; name="velocities", dimensions_first)
    velocities !== nothing && size(velocities) != size(positions) &&
        throw(DimensionMismatch("positions have size $(size(positions)); velocities have size $(size(velocities))."))

    labels_source = hasproperty(state, :agent_state) ? getproperty(state, :agent_state) : nothing
    labels = labels_source === nothing ? nothing : collect(labels_source)
    labels !== nothing && length(labels) != size(positions, 1) &&
        throw(DimensionMismatch("agent_state has $(length(labels)) entries for $(size(positions, 1)) agents."))

    return SwarmSnapshot(positions, velocities, labels,
        time === nothing ? nothing : Float64(time), frame)
end

function _snapshot_time(source, t::Int)
    if hasproperty(source, :t)
        ts = getproperty(source, :t)
        return ts isa AbstractVector && t <= length(ts) ? Float64(ts[t]) : nothing
    elseif hasproperty(source, :simcfg) && hasproperty(source.simcfg, :dt)
        return (t - 1) * Float64(source.simcfg.dt)
    elseif hasproperty(source, :dt)
        return (t - 1) * Float64(source.dt)
    end
    return nothing
end

"""
    swarm_snapshot(source[, t]) -> SwarmSnapshot

Expose a current reservoir/ABM state or one frame of a simulation result through
one TDA-facing interface. Supported shapes are:

- a reservoir or model with `state.pos` and optionally `state.vel`;
- a simulation result with `pos_hist`/`vel_hist`;
- a driven-reservoir result with `states[t]`, where a state has `P`/`V`;
- a raw state or point-cloud-like object with `P`/`V` or `pos`/`vel`.

Matrices returned by `raw_state` may be `D×N`; `SwarmSnapshot.positions` and
`.velocities` are always normalised to `N×D`. Supply `t` for a trajectory;
omit it for a current state.
"""
function swarm_snapshot(source, t::Union{Nothing,Int}=nothing)
    source isa SwarmSnapshot && return source

    if hasproperty(source, :pos_hist)
        t === nothing && throw(ArgumentError("A trajectory requires a frame index `t`."))
        1 <= t <= length(source.pos_hist) || throw(BoundsError(source.pos_hist, t))
        state = (P=source.pos_hist[t],
            V=hasproperty(source, :vel_hist) ? source.vel_hist[t] : nothing,
            agent_state=hasproperty(source, :state_hist) ? source.state_hist[t] : nothing)
        return _snapshot_from_state(state; time=_snapshot_time(source, t), frame=t)
    elseif hasproperty(source, :states)
        t === nothing && throw(ArgumentError("A driven trajectory requires a frame index `t`."))
        1 <= t <= length(source.states) || throw(BoundsError(source.states, t))
        return _snapshot_from_state(source.states[t]; time=_snapshot_time(source, t), frame=t)
    elseif hasproperty(source, :state) && hasproperty(source.state, :pos)
        return _snapshot_from_state(source.state;
            time=t === nothing ? nothing : _snapshot_time(source, t), frame=t)
    end

    return _snapshot_from_state(source; frame=t)
end

"Return the snapshot's point cloud using positions or combined position–velocity coordinates."
function swarm_pointcloud(source, t::Union{Nothing,Int}=nothing;
    representation::Symbol=:position, standardize::Bool=false)
    snap = swarm_snapshot(source, t)
    X = if representation == :position
        copy(snap.positions)
    elseif representation == :position_velocity
        snap.velocities === nothing && throw(ArgumentError(
            "representation=:position_velocity requires velocities."))
        hcat(snap.positions, snap.velocities)
    else
        throw(ArgumentError("representation must be :position or :position_velocity."))
    end
    if standardize
        for col in eachcol(X)
            μ, σ = mean(col), std(col)
            σ > 0 && (col .= (col .- μ) ./ σ)
        end
    end
    return X
end

"Pairwise distances on a selected swarm point-cloud representation."
function swarm_dissimilarity(source, t::Union{Nothing,Int}=nothing;
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing)
    X = swarm_pointcloud(source, t;
        representation=representation, standardize=standardize)
    periodic && representation != :position && throw(ArgumentError(
        "periodic distances are currently defined only for :position."))
    periodic && L === nothing && throw(ArgumentError("periodic=true requires L."))
    N, D = size(X)
    distances = zeros(Float64, N, N)
    @inbounds for i in 1:N, j in (i + 1):N
        δ = view(X, j, :) .- view(X, i, :)
        dij = periodic ? sqrt(sum((δ[d] - round(δ[d] / L) * L)^2 for d in 1:D)) : norm(δ)
        distances[i, j] = distances[j, i] = dij
    end
    return distances
end

"Physical time attached to a snapshot, or a caller-supplied fallback."
snapshot_time(s::SwarmSnapshot; fallback=nothing) =
    s.time === nothing ? fallback : s.time

"Physical time for frame `t`; defaults to the zero-based frame coordinate."
swarm_time(source, t::Int) = snapshot_time(swarm_snapshot(source, t); fallback=Float64(t - 1))

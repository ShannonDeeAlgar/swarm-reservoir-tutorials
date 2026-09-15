# ============================================================
# my_TDA.jl
# TDA helpers for model-independent swarm snapshots over time
# ============================================================
#
# Workflow (point cloud -> persistence diagram -> vectorised feature):
#   swarm_snapshot(source, t)   -> common state for Couzin/Lymburn/Mizzi/Lund
#   positions_at(out, t)        -> N×D matrix of agent positions at time t
#   ph_snapshot(out; t, maxdim) -> persistent homology (Vietoris-Rips), one
#                                  diagram per homology dimension 0:maxdim
#   bd_lifetime / total_persistence / persistence_entropy
#                                -> scalar summaries of a single diagram
#   betti_curve(diag, εs)        -> vectorised Betti curve (the observation
#                                   map used by Project T3-H1)
#   crocker(out; t_idxs, εs)     -> Betti curves stacked over time (a
#                                   CROCKER matrix)
#   diagram_distance(dgmA, dgmB) -> exact Wasserstein/Bottleneck distance
#                                   between two diagrams (via PersistenceDiagrams.jl)
#
# The default filtration is Euclidean Vietoris-Rips on agent positions.
# `spatial_dissimilarity` provides the equivalent matrix explicitly and,
# when requested, uses minimum-image distances in a periodic box.
# `ph_snapshot`'s `filtration_fn` argument keeps the agent-based Rips
# construction fixed and accepts a different edge dissimilarity `f`, via
# `interaction_graph_dissimilarity` (graph-
# shortest-path on the Couzin active interaction network) or
# `alignment_dissimilarity` (heading similarity); both flow through to
# `betti_curves_snapshot` too. `density_grid_ph` is a *different K*
# entirely (a spatial-mesh/density-grid cubical complex, not the agent
# point cloud), kept as its own function rather than a `filtration_fn`
# since it doesn't fit the "N×N matrix on N agents" contract. No vineyards
# / feature tracking through time here regardless of filtration choice --
# that's still the subject of Projects T3-P1/T3-P2/T3-H2/T3-P5, not
# something this file solves.

using Ripserer
using StaticArrays
using Statistics
using LinearAlgebra
using CairoMakie
using ProgressMeter
using PersistenceDiagrams
using Graphs

include(joinpath(@__DIR__, "my_swarm_TDA.jl"))

# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

to_points2(X::AbstractMatrix) = [SVector{2,Float64}(X[i, 1], X[i, 2]) for i in axes(X, 1)]

function to_points(X::AbstractMatrix)
    D = size(X, 2)
    D > 0 || throw(ArgumentError("point matrix must have at least one coordinate column"))
    return [SVector{D,Float64}(ntuple(j -> Float64(X[i, j]), D)) for i in axes(X, 1)]
end

"""
    positions_at(out, t) -> N×D Matrix{Float64}

Extract all spatial coordinates from snapshot `t` through `swarm_snapshot`.
Couzin, Lymburn, Mizzi and Lund outputs are supported in 2D; 3D state-vector
outputs are supported as well.
"""
positions_at(out, t::Int) = swarm_snapshot(out, t).positions
positions_at(snapshot::SwarmSnapshot) = snapshot.positions

"""
    bd_lifetime(diag) -> (B, D, L)

Extract finite birth times `B`, death times `D`, and lifetimes `L`
from a persistence diagram.
"""
function bd_lifetime(diag)
    B = Float64[]
    D = Float64[]
    L = Float64[]

    for itv in diag
        b = getproperty(itv, :birth)
        d = getproperty(itv, :death)

        if isfinite(d) && d > b
            bb = Float64(b)
            dd = Float64(d)
            push!(B, bb)
            push!(D, dd)
            push!(L, dd - bb)
        end
    end

    return B, D, L
end

"""
    total_persistence(diag; p=1)

Compute total persistence of a diagram using only finite lifetimes.
"""
function total_persistence(diag; p::Real=1)
    _, _, L = bd_lifetime(diag)
    isempty(L) ? 0.0 : sum(L .^ p)
end

"""
    persistence_entropy(diag; ϵ=1e-12)

Compute persistence entropy from finite lifetimes.
"""
function persistence_entropy(diag; ϵ::Real=1e-12)
    _, _, L = bd_lifetime(diag)
    isempty(L) && return 0.0

    s = sum(L)
    s <= 0 && return 0.0

    probs = L ./ s
    return -sum(probs .* log.(probs .+ ϵ))
end

"""
    ph_snapshot(out; t=1, maxdim=2, thresh=nothing, filtration_fn=nothing)

Compute persistent homology for the swarm at snapshot `t`. This function
always uses an agent-based Vietoris--Rips construction; `filtration_fn`
changes its edge dissimilarity. Constructions with a genuinely different
`K`, such as `density_grid_ph`'s cubical grid, use a separate entry point.

By default (`filtration_fn === nothing`), this is the original behaviour:
agent-based Vietoris--Rips persistence with raw Euclidean distance between
positions as the edge filtration value.

`filtration_fn`, if given, is a function `(out, t) -> D` returning an `N×N`
dissimilarity matrix (`N` = agent count at snapshot `t`); persistence is
then computed on `D` directly (`ripserer` accepts either raw points or a
precomputed matrix). This changes `f` while retaining the Rips/flag
construction on the same `N` agents. See
`interaction_graph_dissimilarity` and `alignment_dissimilarity`
below for two ready-made options; a `filtration_fn` closing over extra
arguments (e.g. `(out, t) -> interaction_graph_dissimilarity(out, t; P=P)`)
is the usual way to supply one.

Requires at least 2 agents at time `t`; the default position-based path
also requires at least 2 *distinct* positions (a single point, or all
agents co-located, gives no well-defined Vietoris-Rips filtration). This
distinctness check is skipped for a supplied `filtration_fn`, since a
non-spatial `D` (e.g. graph distance) can be well-defined even when
positions coincide.
"""
function ph_snapshot(out; t::Int=1, maxdim::Int=2, thresh=nothing, filtration_fn=nothing)
    if filtration_fn === nothing
        X = positions_at(out, t)
        pts = to_points(X)

        length(pts) < 2 && throw(ArgumentError(
            "ph_snapshot: need at least 2 points at t=$t, got $(length(pts))."))
        length(unique(pts)) < 2 && throw(ArgumentError(
            "ph_snapshot: all $(length(pts)) agent positions coincide at t=$t; " *
            "Vietoris-Rips homology is not informative here."))

        return thresh === nothing ?
            ripserer(pts; dim_max=maxdim) :
            ripserer(pts; dim_max=maxdim, threshold=thresh)
    end

    Dmat = filtration_fn(out, t)
    N1, N2 = size(Dmat)
    N1 == N2 || throw(ArgumentError(
        "ph_snapshot: filtration_fn must return a square N×N matrix, got $(N1)×$(N2)."))
    N1 < 2 && throw(ArgumentError("ph_snapshot: need at least 2 points at t=$t, got $N1."))

    return thresh === nothing ?
        ripserer(Dmat; dim_max=maxdim) :
        ripserer(Dmat; dim_max=maxdim, threshold=thresh)
end

"""
    ph_swarm_snapshot(source; t=nothing, representation=:position, ...)

Compute persistent homology directly from a current swarm/reservoir state or
from frame `t` of a trajectory. Unlike the historical `ph_snapshot` entry
point, this accepts every source supported by `swarm_snapshot`.
"""
function ph_swarm_snapshot(source; t::Union{Nothing,Int}=nothing,
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing, maxdim::Int=1, thresh=nothing)
    X = swarm_pointcloud(source, t;
        representation=representation, standardize=standardize)
    size(X, 1) >= 2 || throw(ArgumentError("ph_swarm_snapshot requires at least two agents."))
    length(unique(to_points(X))) >= 2 || throw(ArgumentError(
        "ph_swarm_snapshot requires at least two distinct agent points."))

    input = periodic ? swarm_dissimilarity(source, t;
        representation=representation, standardize=standardize,
        periodic=true, L=L) : to_points(X)
    return thresh === nothing ? ripserer(input; dim_max=maxdim) :
        ripserer(input; dim_max=maxdim, threshold=thresh)
end

"""
    ph_swarm_snapshot_reps(source; t, representation=:position, standardize=false,
                             periodic=false, L=nothing, maxdim=1, thresh=nothing)
                             -> (dgm, filtration)

Same as `ph_swarm_snapshot`, but also attaches representative cocycles
(`reps=true`) and returns the underlying Rips filtration alongside the
diagram. The filtration is needed by `representative_cycle` to reconstruct a
plottable 1-cycle for a chosen H1 feature -- it is not stored on the diagram
itself, so keep both return values if you want to visualise a cycle later.
"""
function ph_swarm_snapshot_reps(source; t::Union{Nothing,Int}=nothing,
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing, maxdim::Int=1, thresh=nothing)
    X = swarm_pointcloud(source, t;
        representation=representation, standardize=standardize)
    size(X, 1) >= 2 || throw(ArgumentError("ph_swarm_snapshot_reps requires at least two agents."))
    length(unique(to_points(X))) >= 2 || throw(ArgumentError(
        "ph_swarm_snapshot_reps requires at least two distinct agent points."))

    input = periodic ? swarm_dissimilarity(source, t;
        representation=representation, standardize=standardize,
        periodic=true, L=L) : to_points(X)
    flt = Ripserer.Rips(input)
    dgm = thresh === nothing ? ripserer(flt; dim_max=maxdim, reps=true) :
        ripserer(flt; dim_max=maxdim, threshold=thresh, reps=true)
    return dgm, flt
end

"""
    representative_cycle(flt, interval) -> Vector{Tuple{Int,Int}}

Reconstruct the shortest representative 1-cycle for an H1 `interval` (an
entry of `dgm[2]` from a diagram computed with `reps=true`, e.g. via
`ph_swarm_snapshot_reps`) as a list of agent-index edge pairs, ready to plot
on the swarm's positions with `plot_cycle_on_swarm!`. `flt` is the
filtration returned alongside the diagram by `ph_swarm_snapshot_reps`.

Uses `Ripserer.reconstruct_cycle`, which finds the shortest cycle at the
feature's *birth* time -- this is what makes the drawn loop look like the
actual geometric gap that opened, rather than the raw cohomology
representative (which can include long, visually meaningless chords near
the feature's death time).
"""
function representative_cycle(flt, interval)
    length(interval) == 0 && return Tuple{Int,Int}[]  # not fired for a valid interval, but guards misuse
    cyc = Ripserer.reconstruct_cycle(flt, interval)
    return [Tuple(Ripserer.vertices(sx)) for sx in cyc]
end

"""
    plot_cycle_on_swarm!(ax, positions, cycle_edges; periodic=false, L=nothing,
                          color=:crimson, linewidth=3)

Draw a representative cycle's edges (agent-index pairs from
`representative_cycle`) over a swarm's `positions` (N×D matrix). In a
periodic box, each edge is drawn using the shortest wrapped displacement
between its two agents, so a cycle edge that is actually short across the
boundary is not drawn as a spurious line crossing the whole box.
"""
function plot_cycle_on_swarm!(ax, positions::AbstractMatrix, cycle_edges;
    periodic::Bool=false, L=nothing, color=:crimson, linewidth::Real=3)
    size(positions, 2) == 2 || throw(ArgumentError(
        "plot_cycle_on_swarm! currently supports 2D positions only."))
    periodic && L === nothing && throw(ArgumentError("periodic=true requires L."))
    # Plain scalar wrap (same formula swarm_dissimilarity's periodic branch uses),
    # not the ABM module's `displacement` -- that only accepts its own hand-rolled
    # SVector2/SVector3 struct, not the plain StaticArrays.SVector this file's
    # to_points builds, so the two are not interchangeable here.
    wrap(δ) = periodic ? δ - L * round(δ / L) : δ
    for (i, j) in cycle_edges
        xi, yi = positions[i, 1], positions[i, 2]
        dx = wrap(positions[j, 1] - xi)
        dy = wrap(positions[j, 2] - yi)
        lines!(ax, [xi, xi + dx], [yi, yi + dy]; color=color, linewidth=linewidth)
    end
    agent_ids = unique(vcat(first.(cycle_edges), last.(cycle_edges)))
    scatter!(ax, positions[agent_ids, 1], positions[agent_ids, 2];
        color=color, markersize=10)
    return ax
end

"Return scalar H0/H1 persistence summaries for one current swarm state."
function tda_summary_snapshot(source; t::Union{Nothing,Int}=nothing,
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing, maxdim::Int=1, thresh=nothing)
    dgm = ph_swarm_snapshot(source; t, representation, standardize,
        periodic, L, maxdim, thresh)
    tp0, tp1, ent1, n1, maxL1 = _h0_h1_summary(dgm; maxdim)
    return (; tp0, tp1, ent1, n1, maxL1)
end

"""
    tda_feature_map(source; εs, dims=0:1, ...)

Fixed-length observation vector made by concatenating selected Betti curves.
This is the TDA counterpart of a spatial-kernel `feature_map` and can therefore
be evaluated at every reservoir step before fitting the linear readout.
"""
function tda_feature_map(source; εs::AbstractVector{<:Real},
    dims=0:1, t::Union{Nothing,Int}=nothing,
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing, thresh=nothing,
    min_lifetime::Real=0.0)
    dims_vec = collect(dims)
    isempty(dims_vec) && throw(ArgumentError("dims cannot be empty."))
    minimum(dims_vec) >= 0 || throw(ArgumentError("homology dimensions must be nonnegative."))
    dgm = ph_swarm_snapshot(source; t, representation, standardize,
        periodic, L, maxdim=maximum(dims_vec), thresh)
    return reduce(vcat,
        (Float64.(betti_curve(dgm[d + 1], εs; min_lifetime)) for d in dims_vec))
end

include(joinpath(@__DIR__, "my_TDA_observation.jl"))

"""
    spatial_dissimilarity(out, t; periodic=false, L=nothing)

Pairwise spatial-distance matrix for snapshot `t`. With `periodic=false`,
this is ordinary Euclidean distance. With `periodic=true`, it uses the
minimum-image displacement in a square/cubic periodic box of side length
`L`, so agents on opposite sides of the stored coordinate cut are treated
as nearby.

Use the periodic form as a `ph_snapshot` filtration function when analysing
a periodic simulation:

```julia
fn = (out, t) -> spatial_dissimilarity(out, t; periodic=true, L=P.L)
ph_snapshot(out; t=t, filtration_fn=fn)
```
"""
function spatial_dissimilarity(out, t::Int; periodic::Bool=false, L=nothing)
    X = swarm_snapshot(out, t).positions
    N, D = size(X)
    periodic && L === nothing && throw(ArgumentError(
        "spatial_dissimilarity: periodic=true requires the box side length L"))

    Dmat = zeros(Float64, N, N)
    @inbounds for i in 1:N, j in (i + 1):N
        δ = view(X, j, :) .- view(X, i, :)
        dij = periodic ? sqrt(sum((δ[d] - round(δ[d] / L) * L)^2 for d in 1:D)) : norm(δ)
        Dmat[i, j] = dij
        Dmat[j, i] = dij
    end
    return Dmat
end

"""
    _graph_shortest_path_matrix(g)

Convert an undirected graph to a finite hop-distance matrix. Unreachable
pairs receive one more than the largest finite off-diagonal distance; an
edgeless graph uses `2.0`.
"""
function _graph_shortest_path_matrix(g)
    N = nv(g)
    dists = Graphs.floyd_warshall_shortest_paths(g).dists
    unreachable = dists .== typemax(eltype(dists))
    Dmat = Float64.(dists)
    off_diag = trues(N, N)
    for i in 1:N
        off_diag[i, i] = false
    end
    reachable_off_diag = off_diag .& .!unreachable
    fill_val = any(reachable_off_diag) ? maximum(Dmat[reachable_off_diag]) + 1.0 : 2.0
    Dmat[unreachable] .= fill_val
    for i in 1:N
        Dmat[i, i] = 0.0
    end
    return Dmat
end

"""
    radius_graph_dissimilarity(out, t; radius, periodic=false, L=nothing)

Model-agnostic counterpart to `interaction_graph_dissimilarity`: builds an
undirected neighbour graph from raw Euclidean distance alone (an edge
between agents `i` and `j` iff their distance is `<= radius`), then
returns the graph's shortest-path (hop-count) distance matrix, wrapped in
a Rips filtration as usual. Unlike `interaction_graph_dissimilarity`,
this doesn't depend on any model's own interaction rules (Couzin zones,
Lymburn's `rr`/`ra`, ...) -- just positions and a chosen `radius`, so it
applies to any model's `swarm_snapshot`. For models whose own rules do
define a natural interaction radius (e.g. Lymburn's `rr`/`ra`, both 1.0
in every preset), `radius` is the free parameter that plays the role
Couzin's named zones played: pick it relative to that radius (e.g.
`0.5×`, `1×`, `2×`, `4×`) to see how the graph topology depends on scale.

Agent pairs with no path between them are assigned one more than the
largest finite graph distance found, same sentinel convention as
`interaction_graph_dissimilarity` (see that function's docstring for why
`Graphs.jl`'s `typemax(Int)` sentinel needs handling before use).
"""
function radius_graph_dissimilarity(out, t::Int; radius::Real, periodic::Bool=false, L=nothing)
    X = swarm_snapshot(out, t).positions
    N, D = size(X)
    periodic && L === nothing && throw(ArgumentError(
        "radius_graph_dissimilarity: periodic=true requires the box side length L"))

    g = SimpleGraph(N)
    @inbounds for i in 1:N, j in (i + 1):N
        δ = view(X, j, :) .- view(X, i, :)
        dij = periodic ? sqrt(sum((δ[d] - round(δ[d] / L) * L)^2 for d in 1:D)) : norm(δ)
        dij <= radius && add_edge!(g, i, j)
    end

    return _graph_shortest_path_matrix(g)
end

"""
    temporal_proximity_support(out, t; radius, window=100, periodic=false, L=nothing)
        -> N×N Matrix{Float64}

Model-agnostic, undirected counterpart to Couzin's notebook-local
`temporal_interaction_support` (built on `interaction_layers`, so specific
to that model): over the `window` frames ending at `t`, builds a proximity
edge between every agent pair at each frame (`dist <= radius`, same rule as
`radius_graph_dissimilarity`) and averages that 0/1 adjacency across the
window. Entry `(i,j)` of the result is the fraction of frames during which
`i` and `j` were within `radius` of each other -- *who stayed close*,
independent of heading. Threshold with `support_graph_at`/
`support_flag_dissimilarity` to turn this into a genuinely restricted
carrier `K` (as opposed to every other filtration in this file, which fixes
`K` = the complete graph on all agents and only varies `f_t`).
"""
function temporal_proximity_support(out, t::Int; radius::Real, window::Int=100,
    periodic::Bool=false, L=nothing)
    periodic && L === nothing && throw(ArgumentError(
        "temporal_proximity_support: periodic=true requires the box side length L"))
    first_t = max(1, t - window + 1)
    tids = first_t:t
    N, D = size(swarm_snapshot(out, t).positions)
    counts = zeros(Int, N, N)
    for tt in tids
        X = swarm_snapshot(out, tt).positions
        @inbounds for i in 1:N, j in (i + 1):N
            δ = view(X, j, :) .- view(X, i, :)
            dij = periodic ? sqrt(sum((δ[d] - round(δ[d] / L) * L)^2 for d in 1:D)) : norm(δ)
            if dij <= radius
                counts[i, j] += 1
                counts[j, i] += 1
            end
        end
    end
    return counts ./ length(tids)
end

"""
    temporal_alignment_support(out, t; align_cutoff, window=100) -> N×N Matrix{Float64}

Orientation-interaction counterpart to `temporal_proximity_support`: at
each frame in the window, an edge `(i,j)` is "active" iff the two agents'
headings are aligned beyond `align_cutoff` (unit-velocity dot product
`>= align_cutoff`, i.e. `alignment_dissimilarity(out,tt)[i,j] <= 1 -
align_cutoff`), regardless of spatial distance. Averaging this indicator
over the window gives the fraction of frames during which `i` and `j` were
mutually aligned -- a carrier built from *who consistently moves the same
way*, as opposed to *who stays spatially close*
(`temporal_proximity_support`). This is the "orientation interaction graph"
`K`-construction: threshold it with `support_graph_at`/
`support_flag_dissimilarity` the same way.
"""
function temporal_alignment_support(out, t::Int; align_cutoff::Real, window::Int=100)
    first_t = max(1, t - window + 1)
    tids = first_t:t
    N = size(swarm_snapshot(out, t).velocities, 1)
    counts = zeros(Int, N, N)
    for tt in tids
        D = alignment_dissimilarity(out, tt)
        @inbounds for i in 1:N, j in (i + 1):N
            if D[i, j] <= 1 - align_cutoff
                counts[i, j] += 1
                counts[j, i] += 1
            end
        end
    end
    return counts ./ length(tids)
end

"""
    support_graph_at(support, ε) -> SimpleGraph

Threshold a temporal support matrix (from `temporal_proximity_support` or
`temporal_alignment_support`, entries in `[0,1]` = fraction of the window an
edge was active) into an undirected graph: an edge `(i,j)` survives iff
`support[i,j] >= ε`. This graph *is* the carrier `K` -- unlike every
`filtration_fn` above (which fixes `K` = the complete graph and only varies
`f_t`), sweeping `ε` here changes which agent pairs are admissible
simplices at all.
"""
function support_graph_at(support::AbstractMatrix, ε::Real)
    N = size(support, 1)
    g = SimpleGraph(N)
    for i in 1:N, j in (i + 1):N
        support[i, j] >= ε && add_edge!(g, i, j)
    end
    return g
end

"""
    support_flag_dissimilarity(support, ε)

Return the hop-distance matrix of the graph obtained by thresholding
`support` at `ε`.
"""
function support_flag_dissimilarity(support::AbstractMatrix, ε::Real)
    return _graph_shortest_path_matrix(support_graph_at(support, ε))
end

"""
    forward_directed_graph_points(X, V; radius, half_angle=π/2)

Construct a directed proximity graph. Edge `i → j` exists when `j` is
within `radius` of `i` and inside the forward cone defined by agent
`i`'s velocity and `half_angle`.
"""
function forward_directed_graph_points(X::AbstractMatrix, V::AbstractMatrix; radius::Real, half_angle::Real=π / 2)
    N, D = size(X)
    g = SimpleDiGraph(N)
    cos_half_angle = cos(half_angle)
    @inbounds for i in 1:N
        vi = norm(view(V, i, :))
        vi == 0 && continue
        for j in 1:N
            j == i && continue
            d = view(X, j, :) .- view(X, i, :)
            r = norm(d)
            (r == 0 || r > radius) && continue
            cosang = dot(d, view(V, i, :)) / (r * vi)
            cosang >= cos_half_angle && add_edge!(g, i, j)
        end
    end
    return g
end

function forward_directed_graph(out, t::Int; radius::Real, half_angle::Real=π / 2)
    snap = swarm_snapshot(out, t)
    snap.velocities === nothing && throw(ArgumentError(
        "forward_directed_graph requires velocities."))
    return forward_directed_graph_points(snap.positions, snap.velocities; radius=radius, half_angle=half_angle)
end

"""
    _unit_rows(V::AbstractMatrix) -> Matrix{Float64}

Row-normalise `V` (each row a velocity/heading vector) to unit vectors;
zero rows stay zero rather than becoming `NaN`. Shared by
`alignment_dissimilarity`-style pairwise heading comparisons and
`lagged_heading_influence` below.
"""
function _unit_rows(V::AbstractMatrix)
    N, D = size(V)
    U = zeros(Float64, N, D)
    @inbounds for i in 1:N
        n = norm(view(V, i, :))
        if n > 0
            U[i, :] .= view(V, i, :) ./ n
        end
    end
    return U
end

"""
    lagged_heading_influence(out, t; τ, window=100) -> N×N Matrix{Float64}

Directional-influence counterpart to `alignment_dissimilarity`: instead of
comparing headings at the *same* instant, this asks whether agent `i`'s
heading at time `t′` predicts agent `j`'s heading `τ` frames later,
averaged over a window of `window` such `t′` ending `τ` frames before `t`.
Entry `(i,j)` is `mean_t′ [ û_i(t′)·û_j(t′+τ) ]` -- the lagged directional
correlation.

**Not symmetric** in `(i,j)` for `τ > 0`: writing `S(i,j,τ)` for entry
`(i,j)`, reindexing gives `S(j,i,τ) = S(i,j,-τ)`, a genuinely different
pairing (`j` predicting `i`'s future, vs. `i` predicting `j`'s) rather than
just a relabelling. `directional_influence_asymmetry` below subtracts the
two to build a real leadership signal; at `τ=0` the two coincide exactly
(`S(i,j,0) = S(j,i,0) = û_i(t)·û_j(t)`, ordinary instantaneous alignment),
so no lag carries no directional information, as it should.

Needs `window + τ` frames of history before `t`.
"""
function lagged_heading_influence(out, t::Int; τ::Int, window::Int=100)
    τ >= 0 || throw(ArgumentError("lagged_heading_influence: τ must be >= 0"))
    first_t = t - window - τ + 1
    first_t >= 1 || throw(ArgumentError(
        "lagged_heading_influence: need $(window + τ) frames of history before t=$t, only $(t - 1) available"))
    tids = first_t:(first_t + window - 1)
    N = size(swarm_snapshot(out, t).velocities, 1)
    S = zeros(Float64, N, N)
    for t′ in tids
        U0 = _unit_rows(swarm_snapshot(out, t′).velocities)
        U1 = _unit_rows(swarm_snapshot(out, t′ + τ).velocities)
        S .+= U0 * U1'
    end
    return S ./ length(tids)
end

"""
    directional_influence_asymmetry(out, t; τ, window=100) -> N×N Matrix{Float64}

Leadership signal: `S(i,j,τ) - S(j,i,τ)` from `lagged_heading_influence`,
i.e. how much better `i`'s heading at time `t′` predicts `j`'s heading `τ`
frames later than the reverse pairing does. Positive `(i,j)` means `i`
leads `j`; the matrix is antisymmetric by construction
(`asym[i,j] == -asym[j,i]`) and identically zero at `τ=0`.

Threshold with `directional_influence_graph_at` to build a directed
carrier `K` from genuine, non-instantaneous predictive influence --
distinct from `forward_directed_graph_points` (purely geometric, a single
snapshot, no time-lag) and from `alignment_dissimilarity` (instantaneous,
no lag, no direction).
"""
function directional_influence_asymmetry(out, t::Int; τ::Int, window::Int=100)
    S = lagged_heading_influence(out, t; τ=τ, window=window)
    return S .- S'
end

"""
    directional_influence_graph_at(asym, θ) -> SimpleDiGraph

Threshold a `directional_influence_asymmetry` matrix into a directed
graph: edge `i -> j` iff `asym[i,j] >= θ` (`i` leads `j` by at least `θ`).
Pair with `directed_flag_betti01`/`directed_short_cycle_counts` (both
defined earlier in this notebook, generic over any `SimpleDiGraph`)
exactly as Part C does for `forward_directed_graph_points`.
"""
function directional_influence_graph_at(asym::AbstractMatrix, θ::Real)
    N = size(asym, 1)
    g = SimpleDiGraph(N)
    for i in 1:N, j in 1:N
        i == j && continue
        asym[i, j] >= θ && add_edge!(g, i, j)
    end
    return g
end

"""
    interaction_graph_dissimilarity(out, t; P, displacement_fn=displacement)

Construct a **Vietoris--Rips filtration of the shortest-path metric** on
the Couzin active interaction network (`interaction_layers`). This does
not compute graph homology directly and it does not permanently restrict
the complex to the original interaction edges:

- at scale 1, its 1-skeleton is the symmetrised active graph and its
  simplices form that graph's clique (flag) complex;
- at scale `r > 1`, edges join vertices within `r` interaction hops, so
  the simplices form the clique complex of the corresponding graph power.

Thus this function changes the edge dissimilarity used by a Rips
filtration. A genuinely interaction-restricted complex would instead omit
non-interaction edges at every scale (for example with a sparse/custom
filtration and a fixed threshold).

The active graph is directed; it's symmetrised first (an undirected edge
wherever either direction is active) since Vietoris-Rips dissimilarity is
expected to be symmetric. Agent pairs with no path between them (different
connected components) are assigned one more than the largest finite graph
distance found, rather than `Inf`, so the matrix stays finite. Consequently
those components do merge at that final artificial scale. Set `thresh`
below the sentinel value if they should remain disconnected throughout
the reported filtration.

`displacement_fn` must match the geometry used to construct the network.
Its default, `displacement(p, q, P.L)`, is the minimum-image periodic
displacement used by periodic Couzin runs. For an unbounded run, pass
`(p, q, L) -> q - p`.

Returns a `(out, t)`-free `N×N` matrix; wrap in a closure over `P` to use
as a `ph_snapshot` `filtration_fn`, e.g. `(out, t) ->
interaction_graph_dissimilarity(out, t; P=P)`.
"""
function interaction_graph_dissimilarity(out, t::Int; P, displacement_fn=displacement, layer::Symbol=:active)
    snap = swarm_snapshot(out, t)
    snap.velocities === nothing && throw(ArgumentError(
        "interaction_graph_dissimilarity requires velocities."))
    D = size(snap.positions, 2)
    # interaction_layers dispatches on the ABM's own SVector2/SVector3 (see
    # HELPERS/my_geometry_helpers.jl), not StaticArrays.SVector -- the two are
    # distinct types even at matching dimension, so this conversion cannot use
    # StaticArrays.SVector the way to_points/to_points3 elsewhere in this file do.
    VecT = D == 2 ? SVector2 : D == 3 ? SVector3 : throw(ArgumentError(
        "interaction_graph_dissimilarity: unsupported dimension D=$D"))
    pos = [VecT(Tuple(snap.positions[i, :])...) for i in axes(snap.positions, 1)]
    vel = [VecT(Tuple(snap.velocities[i, :])...) for i in axes(snap.velocities, 1)]
    # layer picks which interaction_layers graph this K is built from --
    # :active (default, the union used everywhere else in this notebook),
    # or one of :rep/:ori/:att to restrict K to a single interaction zone.
    g_layer = getproperty(interaction_layers(pos, vel, P; displacement_fn=displacement_fn), layer)
    g_undirected = SimpleGraph(g_layer)

    return _graph_shortest_path_matrix(g_undirected)
end

"""
    alignment_dissimilarity(out, t)

Edge dissimilarity for the fixed agent-based Rips construction: heading
similarity rather than position, `1 - v̂ᵢ·v̂ⱼ` between normalised
velocity vectors (`0` for
identical heading, `2` for opposite headings), instead of raw Euclidean
distance. Two agents heading the same way count as topologically close
even if they're spatially far apart in this snapshot, and vice versa.

Dimension-agnostic: works on both 2D and 3D `out.vel_hist` unchanged (both
`SVector2`/`SVector3`, `HELPERS/my_geometry_helpers.jl`, support `unit`/`dot`
directly; neither is a broadcastable `AbstractVector`).
"""
function alignment_dissimilarity(out, t::Int)
    V = swarm_snapshot(out, t).velocities
    V === nothing && throw(ArgumentError("alignment_dissimilarity requires velocities."))
    N = size(V, 1)
    units = [norm(view(V, i, :)) > 0 ? collect(view(V, i, :)) ./ norm(view(V, i, :)) : zeros(size(V, 2)) for i in 1:N]

    Dmat = Matrix{Float64}(undef, N, N)
    for i in 1:N, j in 1:N
        Dmat[i, j] = i == j ? 0.0 : 1.0 - dot(units[i], units[j])
    end
    return Dmat
end

"""
    position_velocity_dissimilarity(out, t; standardize=true)

Edge dissimilarity for the fixed agent-based Rips construction: each
agent is embedded as one point in **combined position-velocity space**
(`(x, y, vx, vy)` in 2D,
`(x, y, z, vx, vy, vz)` in 3D), `f` = plain Euclidean distance in that
combined space. This is the filtration used by Topaz, Ziegelmeier &
Halverson (2015): not position alone, not
heading alone, but both together in one metric, so two agents only count
as topologically close if they're near each other *and* moving similarly.

`standardize=true` (default) z-scores each of the position and velocity
coordinates across the current snapshot's agents before combining, so
position (typically a much larger numeric range than velocity in this
codebase's units) doesn't simply dominate the distance by scale. This
specific normalisation is a reasonable default choice, not a verified
reproduction of the original paper's own preprocessing (their source
wasn't available to check); pass `standardize=false` for raw units.

Dimension-agnostic (2D or 3D `out.pos_hist`/`out.vel_hist`, matched by
`positions_at`'s own `SVector3`-detection convention).
"""
function position_velocity_dissimilarity(out, t::Int; standardize::Bool=true)
    combined = swarm_pointcloud(out, t;
        representation=:position_velocity, standardize=standardize)
    N = size(combined, 1)

    Dmat = Matrix{Float64}(undef, N, N)
    @inbounds for i in 1:N, j in 1:N
        Dmat[i, j] = i == j ? 0.0 : norm(view(combined, i, :) .- view(combined, j, :))
    end
    return Dmat
end

"""
    density_grid_ph(out, t; grid_res=40, bandwidth=nothing, maxdim=1,
                    margin=0.15, periodic=false, L=nothing)

Compute cubical persistence on a Gaussian density estimate of a 2D swarm
snapshot. With `periodic=true`, positions are unwrapped around the
periodic centre of mass and `L` is required.

Returns `(dgm, ρ, xs, ys)`.
"""
function density_grid_ph(out, t::Int; grid_res::Int=40, bandwidth::Union{Nothing,Float64}=nothing,
    maxdim::Int=1, margin::Float64=0.15, periodic::Bool=false, L=nothing)

    X = positions_at(out, t)
    size(X, 2) == 2 || throw(ArgumentError(
        "density_grid_ph: 2D snapshots only, got $(size(X, 2)) coordinate columns"))

    if periodic
        L === nothing && throw(ArgumentError("density_grid_ph: periodic=true requires L"))
        centre = [mod(atan(mean(sin.(2π .* X[:, d] ./ L)),
            mean(cos.(2π .* X[:, d] ./ L))) * L / (2π), L) for d in 1:2]
        X = [mod(X[i, d] - centre[d] + L / 2, L) - L / 2
            for i in axes(X, 1), d in 1:2]
    end

    return density_grid_ph_points(X; grid_res=grid_res, bandwidth=bandwidth, maxdim=maxdim, margin=margin)
end

"""
    density_grid_ph_points(X::AbstractMatrix; grid_res=40, bandwidth=nothing,
        maxdim=1, margin=0.15)

The generic core of `density_grid_ph`: KDE-smooth an arbitrary `N×2` point
cloud `X` onto a `grid_res×grid_res` grid and compute cubical persistent
homology of the resulting density field. No notion of an agent swarm, a
time index, or periodicity here -- just points in the plane. `X` can be
physical positions, but equally a heading embedding `(cos θ, sin θ)`, or
any other 2D coordinatisation. `density_grid_ph(out, t; ...)` above is a
thin wrapper over this that adds swarm-snapshot extraction and periodic
recentring.
"""
function density_grid_ph_points(X::AbstractMatrix; grid_res::Int=40,
    bandwidth::Union{Nothing,Float64}=nothing, maxdim::Int=1, margin::Float64=0.15)

    size(X, 2) == 2 || throw(ArgumentError(
        "density_grid_ph_points: 2D points only, got $(size(X, 2)) coordinate columns"))

    xmin, xmax = extrema(X[:, 1])
    ymin, ymax = extrema(X[:, 2])
    dx = (xmax - xmin) * margin + eps()
    dy = (ymax - ymin) * margin + eps()
    xs = collect(range(xmin - dx, xmax + dx; length=grid_res))
    ys = collect(range(ymin - dy, ymax + dy; length=grid_res))

    bw = bandwidth === nothing ? 2.5 * max(xs[2] - xs[1], ys[2] - ys[1]) : bandwidth

    ρ = zeros(Float64, grid_res, grid_res)
    @inbounds for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        s = 0.0
        for k in axes(X, 1)
            dxk = x - X[k, 1]
            dyk = y - X[k, 2]
            s += exp(-(dxk^2 + dyk^2) / (2 * bw^2))
        end
        ρ[i, j] = s
    end

    dgm = ripserer(Cubical(.-ρ); dim_max=maxdim)
    return (dgm=dgm, ρ=ρ, xs=xs, ys=ys)
end

# ------------------------------------------------------------
# 1. Full snapshots over time: diagrams + summaries
# ------------------------------------------------------------

"""
    _h0_h1_summary(dgm; maxdim) -> (tp0, tp1, ent1, n1, maxL1)

Shared scalar-summary extraction (total persistence in H0/H1, H1 persistence
entropy, H1 feature count, max H1 lifetime) used by both
`tda_snapshots_over_time` and `tda_summaries`.
"""
function _h0_h1_summary(dgm; maxdim::Int)
    tp0 = total_persistence(dgm[1]; p=1)

    tp1, ent1, n1, maxL1 = 0.0, 0.0, 0, 0.0
    if maxdim >= 1 && length(dgm) >= 2
        diag1 = dgm[2]
        _, _, L1 = bd_lifetime(diag1)
        tp1   = isempty(L1) ? 0.0 : sum(L1)
        ent1  = persistence_entropy(diag1)
        n1    = length(L1)
        maxL1 = isempty(L1) ? 0.0 : maximum(L1)
    end

    return tp0, tp1, ent1, n1, maxL1
end

"""
    tda_snapshots_over_time(out; t_idxs, maxdim=1, thresh=nothing)

Compute persistent homology on selected snapshots of the swarm.

Returns
-------
(dgms, feats)

where:
- `dgms[k]` is the persistence output for snapshot `t_idxs[k]`
- `feats` is a NamedTuple containing:
    - `t_idxs`: sampled frame indices
    - `time`: physical times supplied by the adapted trajectory (or frame index)
    - `tp0`: total persistence in H0
    - `tp1`: total persistence in H1
    - `ent1`: persistence entropy in H1
    - `n1`: number of finite H1 features
    - `maxL1`: maximum finite H1 lifetime

Holds one persistence-diagram bundle per timestep in memory; prefer
`tda_summaries` if you only need the scalar features (e.g. for long time
series where this would be memory-heavy).
"""
function tda_snapshots_over_time(out; t_idxs::AbstractVector{Int}, maxdim::Int=1, thresh=nothing)
    nT = length(t_idxs)

    dgms  = Vector{Any}(undef, nT)
    tp0   = zeros(nT)
    tp1   = zeros(nT)
    ent1  = zeros(nT)
    n1    = zeros(Int, nT)
    maxL1 = zeros(nT)

    prog = Progress(nT; desc="TDA snapshots", showspeed=true)

    for (k, t) in enumerate(t_idxs)
        dgm = ph_snapshot(out; t=t, maxdim=maxdim, thresh=thresh)
        dgms[k] = dgm
        tp0[k], tp1[k], ent1[k], n1[k], maxL1[k] = _h0_h1_summary(dgm; maxdim=maxdim)
        next!(prog)
    end

    time = [swarm_time(out, t) for t in t_idxs]

    feats = (
        t_idxs = collect(t_idxs),
        time   = time,
        tp0    = tp0,
        tp1    = tp1,
        ent1   = ent1,
        n1     = n1,
        maxL1  = maxL1
    )

    return dgms, feats
end

# ------------------------------------------------------------
# 2. Lightweight summaries only
# ------------------------------------------------------------

"""
    tda_summaries(out; t_idxs, maxdim=1, thresh=nothing)

Lightweight version of `tda_snapshots_over_time` that computes only the
scalar summary-statistics NamedTuple, without keeping the full persistence
diagrams in memory (each `ph_snapshot` result is discarded once its
summary is extracted).
"""
function tda_summaries(out; t_idxs::AbstractVector{Int}, maxdim::Int=1, thresh=nothing)
    nT = length(t_idxs)

    tp0   = zeros(nT)
    tp1   = zeros(nT)
    ent1  = zeros(nT)
    n1    = zeros(Int, nT)
    maxL1 = zeros(nT)

    prog = Progress(nT; desc="TDA summaries", showspeed=true)

    for (k, t) in enumerate(t_idxs)
        dgm = ph_snapshot(out; t=t, maxdim=maxdim, thresh=thresh)
        tp0[k], tp1[k], ent1[k], n1[k], maxL1[k] = _h0_h1_summary(dgm; maxdim=maxdim)
        next!(prog)
    end

    time = [swarm_time(out, t) for t in t_idxs]

    return (
        t_idxs = collect(t_idxs),
        time   = time,
        tp0    = tp0,
        tp1    = tp1,
        ent1   = ent1,
        n1     = n1,
        maxL1  = maxL1
    )
end


# ============================================================
# Diagram distances for comparing regimes / runs
# ============================================================

# Recommended:
# using PersistenceDiagrams
#
# Exact distances then use:
#   Bottleneck()(diagA, diagB)
#   Wasserstein(q=p)(diagA, diagB)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

"""
    finite_birth_death(diag) -> (B, D)

Extract finite birth/death pairs from a persistence diagram.
Intervals with non-finite death are ignored.
"""
function finite_birth_death(diag)
    B = Float64[]
    D = Float64[]
    for itv in diag
        b = Float64(getproperty(itv, :birth))
        d = getproperty(itv, :death)
        if isfinite(d) && d > b
            push!(B, b)
            push!(D, Float64(d))
        end
    end
    return B, D
end

"""
    birth_persistence(diag) -> (B, P)

Convert finite birth/death pairs to birth/persistence coordinates.
"""
function birth_persistence(diag)
    B, D = finite_birth_death(diag)
    P = D .- B
    return B, P
end

# ------------------------------------------------------------
# Fallback: simple persistence-image-style embedding
#
# Not used by default -- `diagram_distance` below (via PersistenceDiagrams.jl's
# exact Bottleneck()/Wasserstein()) is the one to reach for. Use
# `diagram_distance_fallback` only if PersistenceDiagrams.jl is unavailable,
# or you specifically want a persistence-image vectorisation rather than an
# exact diagram distance.
# ------------------------------------------------------------

"""
    persistence_image_vec(diag;
        xrange=nothing, prange=nothing,
        nx=40, ny=40,
        σx=0.15, σp=0.15,
        weight=:persistence)

Create a simple persistence-image-style vector from one diagram using
birth-persistence coordinates and Gaussian bumps.

This is a fallback approximation for diagram comparison when exact
Bottleneck/Wasserstein distance machinery is unavailable.
"""
function persistence_image_vec(diag;
    xrange=nothing,
    prange=nothing,
    nx::Int=40,
    ny::Int=40,
    σx::Real=0.15,
    σp::Real=0.15,
    weight::Symbol=:persistence
)
    B, P = birth_persistence(diag)

    # Empty diagram -> zero image
    if isempty(B)
        return zeros(Float64, nx * ny)
    end

    xmin, xmax = isnothing(xrange) ? (minimum(B), maximum(B)) : xrange
    pmin, pmax = isnothing(prange) ? (0.0, maximum(P)) : prange

    # Guard against degenerate ranges
    xmax = xmax == xmin ? xmin + 1.0 : xmax
    pmax = pmax == pmin ? pmin + 1.0 : pmax

    xs = collect(range(xmin, xmax; length=nx))
    ps = collect(range(pmin, pmax; length=ny))

    img = zeros(Float64, ny, nx)

    @inbounds for k in eachindex(B)
        bk = B[k]
        pk = P[k]

        wk = weight == :persistence ? pk :
             weight == :uniform      ? 1.0 :
             error("Unsupported weight = $(weight). Use :persistence or :uniform.")

        for ix in 1:nx, iy in 1:ny
            dx = (xs[ix] - bk)^2 / (2σx^2)
            dp = (ps[iy] - pk)^2 / (2σp^2)
            img[iy, ix] += wk * exp(-(dx + dp))
        end
    end

    return vec(img)
end

"""
    diagram_distance_fallback(diagA, diagB; p=2, kwargs...)

Approximate distance between diagrams by embedding them into the same
persistence-image-style vector space and taking an L^p norm.
"""
function diagram_distance_fallback(diagA, diagB;
    p::Real=2,
    nx::Int=40,
    ny::Int=40,
    σx::Real=0.15,
    σp::Real=0.15,
    weight::Symbol=:persistence
)
    BA, PA = birth_persistence(diagA)
    BB, PB = birth_persistence(diagB)

    # common ranges
    allB = vcat(BA, BB)
    allP = vcat(PA, PB)

    xrange = isempty(allB) ? (0.0, 1.0) : (minimum(allB), maximum(allB))
    prange = isempty(allP) ? (0.0, 1.0) : (0.0, maximum(allP))

    vA = persistence_image_vec(diagA;
        xrange=xrange, prange=prange,
        nx=nx, ny=ny, σx=σx, σp=σp, weight=weight)

    vB = persistence_image_vec(diagB;
        xrange=xrange, prange=prange,
        nx=nx, ny=ny, σx=σx, σp=σp, weight=weight)

    return norm(vA .- vB, p)
end

# ------------------------------------------------------------
# Main user-facing distance
# ------------------------------------------------------------

# ------------------------------------------------------------
# Diagram distance
# ------------------------------------------------------------

function diagram_distance(dgmA, dgmB;
    dim::Int=1,
    metric::Symbol=:wasserstein,
    p::Real=2
)

    # Extract diagrams if bundle (Ripser output)
    diagA = (dgmA isa AbstractVector) ? dgmA[dim + 1] : dgmA
    diagB = (dgmB isa AbstractVector) ? dgmB[dim + 1] : dgmB

    if metric == :wasserstein
        return PersistenceDiagrams.Wasserstein(p)(diagA, diagB)
    elseif metric == :bottleneck
        return PersistenceDiagrams.Bottleneck()(diagA, diagB)
    else
        error("metric must be :wasserstein or :bottleneck")
    end
end


# ------------------------------------------------------------
# Pairwise distance matrix with progress bar
# ------------------------------------------------------------

function distance_matrix_from_diagrams(dgms;
    dim::Int=1,
    metric::Symbol=:wasserstein,
    p::Real=2
)

    n = length(dgms)
    D = zeros(Float64, n, n)

    total = n*(n-1) ÷ 2
    prog = Progress(total; desc="Diagram distances", showspeed=true)

    for i in 1:n
        for j in i+1:n

            d = diagram_distance(
                dgms[i],
                dgms[j];
                dim=dim,
                metric=metric,
                p=p
            )

            D[i,j] = d
            D[j,i] = d

            next!(prog)
        end
    end

    return D
end

function plot_distance_matrix(D, tvals)

    fig = Figure(size=(650,550))

    ax = Axis(fig[1,1],
        xlabel="time",
        ylabel="time",
        title="Diagram distance matrix"
    )

    hm = heatmap!(ax, tvals, tvals, D; colormap=:viridis)

    Colorbar(fig[1,2], hm, label="distance")

    return fig
end
# ------------------------------------------------------------
# Distances between diagram bundles / runs
# ------------------------------------------------------------

"""
    diagram_distance_at_time(dgmsA, dgmsB, k; dim=1, metric=:wasserstein, p=2, kwargs...)

Compare two diagram bundles at the same sampled-time index `k`
for a chosen homology dimension `dim`.
"""
function diagram_distance_at_time(dgmsA, dgmsB, k::Int;
    dim::Int=1,
    metric::Symbol=:wasserstein,
    p::Real=2,
    kwargs...
)
    diagA = dgmsA[k][dim + 1]
    diagB = dgmsB[k][dim + 1]
    return diagram_distance(diagA, diagB; metric=metric, p=p, kwargs...)
end

"""
    distance_timeseries(dgmsA, dgmsB; dim=1, metric=:wasserstein, p=2, kwargs...)

Compute a distance time series between two runs, assuming the same sampled
time index set for both.
"""
function distance_timeseries(dgmsA, dgmsB;
    dim::Int=1,
    metric::Symbol=:wasserstein,
    p::Real=2,
    kwargs...
)
    n = min(length(dgmsA), length(dgmsB))
    d = zeros(Float64, n)

    for k in 1:n
        d[k] = diagram_distance_at_time(dgmsA, dgmsB, k;
            dim=dim, metric=metric, p=p, kwargs...)
    end

    return d
end



# ============================================================
# CROCKER utilities for swarm TDA
# ============================================================

# ------------------------------------------------------------
# Betti curve from a single persistence diagram
# ------------------------------------------------------------

"""
    betti_curve(diag, εs; min_lifetime=0.0)

Compute the Betti curve β(ε) for a persistence diagram on a grid `εs`.

Counts intervals satisfying:
    birth <= ε < death

Only counts finite intervals with lifetime >= min_lifetime,
unless death is infinite, in which case the interval is always kept.
"""
function betti_curve(diag, εs::AbstractVector{<:Real}; min_lifetime::Real=0.0)
    B = zeros(Int, length(εs))

    for itv in diag
        b = Float64(getproperty(itv, :birth))
        d_raw = getproperty(itv, :death)
        d = isfinite(d_raw) ? Float64(d_raw) : Inf

        if isfinite(d) && (d - b) < min_lifetime
            continue
        end

        i1 = searchsortedfirst(εs, b)
        i2 = searchsortedfirst(εs, d) - 1

        if i1 <= i2 && i1 <= length(εs) && i2 >= 1
            i1c = max(i1, 1)
            i2c = min(i2, length(εs))
            @inbounds B[i1c:i2c] .+= 1
        end
    end

    return B
end

"""
    betti_curves_snapshot(out; t, εs, maxdim=1, thresh=nothing, min_lifetime=0.0,
                           filtration_fn=nothing)

Compute Betti curves β_k(ε) for one snapshot `t`. `filtration_fn` is
forwarded to `ph_snapshot` unchanged (see there): `nothing` for the
default Euclidean-position filtration, or a `(out, t) -> N×N` dissimilarity
function for an alternative edge filtration on the same Rips construction,
e.g.
`interaction_graph_dissimilarity`/`alignment_dissimilarity`.

Returns a NamedTuple with:
- t_idx
- time
- εs
- curves   # Vector of curves, one for each dimension 0:maxdim
"""
function betti_curves_snapshot(out; t::Int, εs::AbstractVector{<:Real},
                               maxdim::Int=1, thresh=nothing, min_lifetime::Real=0.0,
                               filtration_fn=nothing)

    dgm = ph_snapshot(out; t=t, maxdim=maxdim, thresh=thresh, filtration_fn=filtration_fn)
    curves = [betti_curve(dgm[dim + 1], εs; min_lifetime=min_lifetime) for dim in 0:maxdim]

    return (
        t_idx = t,
        time  = swarm_time(out, t),
        εs    = collect(εs),
        curves = curves
    )
end

"""
    plot_betti_curves_snapshot(bc;
        dims=nothing,
        layout=:rows,
        figure_size=nothing,
        linewidth=2,
        titles=nothing)

Plot Betti curves from `betti_curves_snapshot`.
"""
function plot_betti_curves_snapshot(bc;
    dims=nothing,
    layout::Symbol=:rows,
    figure_size=nothing,
    linewidth::Real=2,
    titles=nothing
)

    εs = bc.εs
    curves = bc.curves
    ndims_total = length(curves)

    dims === nothing && (dims = collect(0:ndims_total-1))
    nd = length(dims)

    function dim_title(dim)
        if titles === nothing
            return "Betti curve β$(dim)(ϵ)"
        elseif titles isa AbstractDict
            return get(titles, dim, "Betti curve β$(dim)(ϵ)")
        else
            return "Betti curve β$(dim)(ϵ)"
        end
    end

    if layout == :overlay
        figsz = isnothing(figure_size) ? (900, 450) : figure_size
        fig = Figure(size=figsz)
        ax = Axis(fig[1,1],
            xlabel="Proximity Parameter ϵ",
            ylabel="Betti number",
            title="Betti curves at t = $(round(bc.time, digits=2))"
        )

        for dim in dims
            y = curves[dim + 1]
            stairs!(ax, εs, y; linewidth=linewidth, label="β$(dim)")
        end
        axislegend(ax)
        return fig
    end

    if layout == :rows
        figsz = isnothing(figure_size) ? (900, 260 * nd) : figure_size
        fig = Figure(size=figsz)
        axes = Axis[]

        for (i, dim) in enumerate(dims)
            ax = Axis(fig[i,1],
                xlabel = (i == nd ? "Proximity Parameter ϵ" : ""),
                ylabel = "β$(dim)",
                title  = dim_title(dim)
            )
            stairs!(ax, εs, curves[dim + 1]; linewidth=linewidth)
            push!(axes, ax)
        end

        return fig
    end

    error("layout must be :rows or :overlay")
end

"""
    plot_betti_curve_comparison(out; t_idxs, εs, dim=1,
                                thresh=nothing, min_lifetime=0.0,
                                figure_size=(900,500), linewidth=2)

Plot β_dim(ϵ) for several selected times on one axis.
"""
function plot_betti_curve_comparison(out; t_idxs::AbstractVector{Int},
    εs::AbstractVector{<:Real},
    dim::Int=1,
    thresh=nothing,
    min_lifetime::Real=0.0,
    figure_size=(900, 500),
    linewidth::Real=2
)

    fig = Figure(size=figure_size)
    ax = Axis(fig[1,1],
        xlabel="Proximity Parameter ϵ",
        ylabel="β$(dim)",
        title="Betti curve comparison"
    )

    for t in t_idxs
        dgm = ph_snapshot(out; t=t, maxdim=dim, thresh=thresh)
        y = betti_curve(dgm[dim + 1], εs; min_lifetime=min_lifetime)
        t_phys = swarm_time(out, t)
        stairs!(ax, εs, y; linewidth=linewidth, label="t = $(round(t_phys, digits=2))")
    end

    axislegend(ax)
    return fig
end
# ------------------------------------------------------------
# CROCKER computation
# ------------------------------------------------------------

"""
    crocker(out; t_idxs, εs, maxdim=1, thresh=nothing)

Compute CROCKER matrices for homology dimensions 0..maxdim on selected
snapshots of a simulation output.

Arguments
---------
- `out`: simulation output
- `t_idxs`: frame indices to analyse
- `εs`: ε-grid
- `maxdim`: maximum homology dimension
- `thresh`: optional threshold passed to `ph_snapshot`

Returns
-------
NamedTuple with fields:
- `t_idxs`
- `time`
- `εs`
- `crock`

where
- `crock[d+1]` is a matrix of size `(length(εs), length(t_idxs))`
- entry `(i,j)` is β_d(εs[i], t_idxs[j])
"""
function crocker(out; t_idxs::AbstractVector{Int},
                 εs::AbstractVector{<:Real},
                 maxdim::Int=1,
                 thresh=nothing,
                 τ=0.0)

    nt = length(t_idxs)
    ne = length(εs)

    crock = [zeros(Int, ne, nt) for _ in 0:maxdim]

    prog = Progress(nt; desc="CROCKER (PH per time)", showspeed=true)

    for (j, t) in enumerate(t_idxs)

        dgm = ph_snapshot(out; t=t, maxdim=maxdim, thresh=thresh)

        for dim in 0:maxdim
            diag = dgm[dim + 1]
            crock[dim + 1][:, j] .= betti_curve(diag, εs; min_lifetime=τ)
        end

        next!(prog)
    end

    time = [swarm_time(out, t) for t in t_idxs]

    return (
        t_idxs = collect(t_idxs),
        time   = time,
        εs     = collect(εs),
        crock  = crock
    )
end

function _plot_single_crocker_topaz!(ax, time, εs, β::AbstractMatrix;
    dim::Int=0,
    add_colorbar_to=nothing,
    label_text=nothing,
    filled::Bool=true,
    show_contours::Bool=true,
    contour_labels::Bool=false,
    linewidth::Real=2,
    max_display_level::Union{Nothing,Int}=nothing,
    contour_levels=nothing,
    palette=[:red, :blue, :green, :purple, :orange, :brown, :pink],
)

    # β is ε × time; Makie wants nt × ne
    Zraw = permutedims(Float32.(β), (2, 1))

    βmax_raw = maximum(β)

    # Clip displayed values if requested
    if isnothing(max_display_level)
        Z = copy(Zraw)
        βmax = βmax_raw
    else
        Z = clamp.(Zraw, 0, max_display_level)
        βmax = max_display_level
    end

    # contour levels to draw
    line_levels = contour_levels === nothing ? collect(0:βmax) : collect(contour_levels)

    # discrete filled bands centred on integers
    fill_levels = collect(-0.5:1:(βmax + 0.5))

    # categorical colour gradient
    ncols = βmax + 1
    cols = palette[1:min(ncols, length(palette))]
    if ncols > length(cols)
        # repeat if needed
        cols = [palette[mod1(i, length(palette))] for i in 1:ncols]
    end
    cmap = cgrad(cols; categorical=true)

    plt_fill = nothing

    if filled
        plt_fill = contourf!(
            ax, time, εs, Z;
            levels=fill_levels,
            mode=:normal,
            colormap=cmap
        )
    end

    if show_contours
        contour!(
            ax, time, εs, Z;
            levels=line_levels,
            linewidth=linewidth,
            color=:black,
            labels=contour_labels
        )
    end

    if label_text !== nothing
        text!(
            ax,
            label_text.position[1],
            label_text.position[2];
            text=label_text.text,
            fontsize=get(label_text, :fontsize, 28),
            align=get(label_text, :align, (:center, :center))
        )
    end

    if add_colorbar_to !== nothing && filled
        Colorbar(
            add_colorbar_to,
            plt_fill;
            label="level",
            ticks=0:βmax
        )
    end

    ax.xlabel = "Simulation Time t"
    ax.ylabel = "Proximity Parameter ϵ"
    ax.title  = "CROCKER: β$(dim)(ϵ,t)"

    return ax
end

function plot_crocker_topaz(crock_out;
    dim::Int=1,
    figure_size=(1100, 500),
    filled::Bool=true,
    show_contours::Bool=true,
    contour_labels::Bool=false,
    linewidth::Real=2,
    max_display_level::Union{Nothing,Int}=nothing,
    contour_levels=nothing,
    label_text=nothing,
    palette=[:red, :blue, :green, :purple, :orange, :brown, :pink],
)

    time = crock_out.time
    εs   = crock_out.εs
    β    = crock_out.crock[dim + 1]

    fig = Figure(size=figure_size)
    ax  = Axis(fig[1, 1])

    _plot_single_crocker_topaz!(
        ax, time, εs, β;
        dim=dim,
        add_colorbar_to=fig[1, 2],
        label_text=label_text,
        filled=filled,
        show_contours=show_contours,
        contour_labels=contour_labels,
        linewidth=linewidth,
        max_display_level=max_display_level,
        contour_levels=contour_levels,
        palette=palette
    )

    return fig
end

function plot_crocker_pair_topaz(crock_out;
    dims::Tuple{Int,Int}=(0,1),
    figure_size=(1200, 900),
    filled::Tuple{Bool,Bool}=(false, true),
    show_contours::Tuple{Bool,Bool}=(true, true),
    contour_labels::Bool=false,
    linewidth::Real=2,
    max_display_levels::Tuple{Union{Nothing,Int},Union{Nothing,Int}}=(10, 5),
    contour_levels=(nothing, nothing),
    panel_labels=nothing,
    palettes=(
        [:red, :blue, :green, :purple, :orange, :brown, :pink],
        [:red, :blue, :green, :purple, :orange, :brown, :pink]
    ),
    linkx::Bool=true
)

    time = crock_out.time
    εs   = crock_out.εs

    d1, d2 = dims
    βa = crock_out.crock[d1 + 1]
    βb = crock_out.crock[d2 + 1]

    fig = Figure(size=figure_size)

    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[2, 1])

    lbl1 = panel_labels === nothing ? nothing : panel_labels[1]
    lbl2 = panel_labels === nothing ? nothing : panel_labels[2]

    cbpos1 = filled[1] ? fig[1, 2] : nothing
    cbpos2 = filled[2] ? fig[2, 2] : nothing

    _plot_single_crocker_topaz!(
        ax1, time, εs, βa;
        dim=d1,
        add_colorbar_to=cbpos1,
        label_text=lbl1,
        filled=filled[1],
        show_contours=show_contours[1],
        contour_labels=contour_labels,
        linewidth=linewidth,
        max_display_level=max_display_levels[1],
        contour_levels=contour_levels[1],
        palette=palettes[1]
    )

    _plot_single_crocker_topaz!(
        ax2, time, εs, βb;
        dim=d2,
        add_colorbar_to=cbpos2,
        label_text=lbl2,
        filled=filled[2],
        show_contours=show_contours[2],
        contour_labels=contour_labels,
        linewidth=linewidth,
        max_display_level=max_display_levels[2],
        contour_levels=contour_levels[2],
        palette=palettes[2]
    )

    if linkx
        linkxaxes!(ax1, ax2)
    end

    return fig
end
# ------------------------------------------------------------
# 3. Plot selected summary series
# ------------------------------------------------------------

"""
    plot_tda_summaries(feats;
        series=[:tp0, :tp1, :ent1],
        x=:time,
        layout=:rows,
        figure_size=nothing,
        linewidth=2,
        markersize=8,
        show_markers=false,
        linkx=true,
        titles=nothing
    )

Plot selected TDA summary statistics from a NamedTuple returned by
`tda_summaries` or `tda_snapshots_over_time`.

Arguments
---------
- `feats`: NamedTuple containing summary arrays
- `series`: vector of Symbols naming fields to plot
- `x`: x-axis field, usually `:time` or `:t_idxs`
- `layout`: one of `:rows`, `:cols`, `:overlay`
- `figure_size`: optional `(width, height)`
- `linewidth`: line width
- `markersize`: scatter marker size
- `show_markers`: whether to overlay markers
- `linkx`: whether to link x axes in multi-panel layouts
- `titles`: optional Dict or NamedTuple of custom titles
"""
function plot_tda_summaries(feats;
    series::AbstractVector{Symbol}=[:tp0, :tp1, :ent1],
    x::Symbol=:time,
    layout::Symbol=:rows,
    figure_size=nothing,
    linewidth::Real=2,
    markersize::Real=8,
    show_markers::Bool=false,
    linkx::Bool=true,
    titles=nothing
)

    hasproperty(feats, x) || error("`feats` has no field $(x)")
    xvals = getproperty(feats, x)

    for s in series
        hasproperty(feats, s) || error("`feats` has no field $(s)")
    end

    n = length(series)

    default_titles = Dict(
        :tp0   => "Total persistence (H0)",
        :tp1   => "Total persistence (H1)",
        :ent1  => "Persistence entropy (H1)",
        :n1    => "Number of finite H1 features",
        :maxL1 => "Maximum H1 lifetime"
    )

    default_ylabels = Dict(
        :tp0   => "TP(H0)",
        :tp1   => "TP(H1)",
        :ent1  => "Entropy(H1)",
        :n1    => "Count(H1)",
        :maxL1 => "Max lifetime(H1)"
    )

    function get_title(s)
        if titles === nothing
            return get(default_titles, s, String(s))
        elseif titles isa AbstractDict
            return get(titles, s, get(default_titles, s, String(s)))
        elseif titles isa NamedTuple
            return hasproperty(titles, s) ? getproperty(titles, s) : get(default_titles, s, String(s))
        else
            return get(default_titles, s, String(s))
        end
    end

    xlabel_str = x == :time ? "time" : String(x)

    # ---------------- overlay ----------------
    if layout == :overlay
        figsz = isnothing(figure_size) ? (900, 450) : figure_size
        fig = Figure(size=figsz)
        ax = Axis(fig[1, 1], xlabel=xlabel_str, ylabel="value", title="TDA summaries")

        for s in series
            y = getproperty(feats, s)
            lines!(ax, xvals, y; linewidth=linewidth, label=get_title(s))
            if show_markers
                scatter!(ax, xvals, y; markersize=markersize)
            end
        end

        axislegend(ax)
        return fig
    end

    # ---------------- rows ----------------
    if layout == :rows
        figsz = isnothing(figure_size) ? (900, 240 * n) : figure_size
        fig = Figure(size=figsz)
        axes = Axis[]

        for (i, s) in enumerate(series)
            ax = Axis(
                fig[i, 1],
                xlabel = (i == n ? xlabel_str : ""),
                ylabel = get(default_ylabels, s, String(s)),
                title  = get_title(s)
            )

            y = getproperty(feats, s)
            lines!(ax, xvals, y; linewidth=linewidth)
            if show_markers
                scatter!(ax, xvals, y; markersize=markersize)
            end

            push!(axes, ax)
        end

        if linkx && length(axes) > 1
            linkxaxes!(axes...)
        end

        return fig
    end

    # ---------------- cols ----------------
    if layout == :cols
        figsz = isnothing(figure_size) ? (320 * n, 420) : figure_size
        fig = Figure(size=figsz)
        axes = Axis[]

        for (i, s) in enumerate(series)
            ax = Axis(
                fig[1, i],
                xlabel = xlabel_str,
                ylabel = get(default_ylabels, s, String(s)),
                title  = get_title(s)
            )

            y = getproperty(feats, s)
            lines!(ax, xvals, y; linewidth=linewidth)
            if show_markers
                scatter!(ax, xvals, y; markersize=markersize)
            end

            push!(axes, ax)
        end

        if linkx && length(axes) > 1
            linkxaxes!(axes...)
        end

        return fig
    end

    error("layout must be one of :rows, :cols, or :overlay")
end

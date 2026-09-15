# Couzin_network_time_series

# _unitvec
# _is_visible

# interaction_layers

# Couzin_node_data_snapshot_2d
# Couzin_active_edge_weights_2d
# Couzin_proximity_graph_2d

# Couzin_active_graph_series_2d
# Couzin_node_data_series_2d
# Couzin_active_weight_series_2d
# Couzin_proximity_graph_series_2d

# _strength_rep
# _strength_ori
# _strength_att

# Couzin_edge_metadata
# Couzin_edge_segments

using Graphs
using CairoMakie
using LinearAlgebra
using ProgressMeter

function Couzin_network_time_series(
    out,
    P::CouzinParams;
    layer::Symbol = :active,
    show_progress::Bool = true,
)
    idxs = eachindex(out.pos_hist)
    graph_series = Vector{SimpleDiGraph}(undef, length(idxs))
    prog = show_progress ? Progress(length(idxs); desc="Couzin_network_time_series", showspeed=true) : nothing

    for (j, k) in enumerate(idxs)
        graph_series[j] = getproperty(interaction_layers(out.pos_hist[k], out.vel_hist[k], P), layer)
        show_progress && next!(prog)
    end

    return network_time_series(graph_series; t = out.t)
end
# --------------------------------------------------
# Generic interaction layers
# --------------------------------------------------

function interaction_layers(
    pos::AbstractVector{V},
    vel::AbstractVector{V},
    P::CouzinParams;
    active_only::Bool = false,
    displacement_fn::Function = displacement,
) where {V<:Union{SVector2,SVector3}}

    N = length(pos)

    g_rep    = SimpleDiGraph(N)
    g_ori    = SimpleDiGraph(N)
    g_att    = SimpleDiGraph(N)
    g_active = SimpleDiGraph(N)

    n_rep = zeros(Int, N)
    n_ori = zeros(Int, N)
    n_att = zeros(Int, N)

    @inbounds for i in 1:N
        pi   = pos[i]
        vi_u = _unitvec(vel[i])

        rep_neigh = Int[]
        ori_neigh = Int[]
        att_neigh = Int[]

        for j in 1:N
            j == i && continue

            d = displacement_fn(pi, pos[j], P.L)
            r = norm(d)
            r == 0 && continue

            if r <= P.Zr
                add_edge!(g_rep, j, i)
                push!(rep_neigh, j)
                continue
            end

            d_u = d / r
            _is_visible(vi_u, d_u, P.blind_half_angle) || continue

            if r <= P.Zo
                add_edge!(g_ori, j, i)
                push!(ori_neigh, j)
            elseif r <= P.Za
                add_edge!(g_att, j, i)
                push!(att_neigh, j)
            end
        end

        n_rep[i] = length(rep_neigh)
        n_ori[i] = length(ori_neigh)
        n_att[i] = length(att_neigh)

        if !isempty(rep_neigh)
            for j in rep_neigh
                add_edge!(g_active, j, i)
            end
        else
            for j in ori_neigh
                add_edge!(g_active, j, i)
            end
            for j in att_neigh
                add_edge!(g_active, j, i)
            end
        end
    end

    return (
        rep = g_rep,
        ori = g_ori,
        att = g_att,
        active = g_active,
        n_rep = n_rep,
        n_ori = n_ori,
        n_att = n_att,
    )
end

"""Interaction graphs for either 2D or 3D Couzin output.

Pass the displacement used by the simulation. This matters for unbounded 3D
runs, where periodic minimum-image distances would create spurious edges.
"""
function Couzin_active_graph_series(
    out, P; displacement_fn::Function=displacement, show_progress::Bool=true,
)
    n = length(out.pos_hist)
    series = Vector{SimpleDiGraph}(undef, n)
    prog = show_progress ? Progress(n; desc="Couzin_active_graph_series", showspeed=true) : nothing
    for k in 1:n
        series[k] = interaction_layers(out.pos_hist[k], out.vel_hist[k], P;
            displacement_fn=displacement_fn).active
        show_progress && next!(prog)
    end
    return series
end

function Couzin_node_data_snapshot_2d(pos, vel, P)
    graphs = interaction_layers(pos, vel, P)
    return (
        n_rep = graphs.n_rep,
        n_ori = graphs.n_ori,
        n_att = graphs.n_att,
    )
end

function Couzin_active_edge_weights_2d(
    pos,
    vel,
    P;
    mode::Symbol = :zone_strength,
)
    graphs = interaction_layers(pos, vel, P)
    g = graphs.active

    weights = Dict{Tuple{Int,Int},Float64}()

    @inbounds for e in edges(g)
        j = src(e)
        i = dst(e)

        d = displacement(pos[i], pos[j], P.L)
        r = norm(d)
        r == 0 && continue

        w =
            if mode == :inverse_distance
                1.0 / r
            elseif mode == :zone_strength
                if r <= P.Zr
                    P.Zr > 0 ? 1.0 - r / P.Zr : 1.0
                elseif r <= P.Zo
                    denom = max(P.Zo - P.Zr, eps())
                    1.0 - (r - P.Zr) / denom
                elseif r <= P.Za
                    denom = max(P.Za - P.Zo, eps())
                    1.0 - (r - P.Zo) / denom
                else
                    0.0
                end
            elseif mode == :alignment
                vi = unit(vel[i])
                vj = unit(vel[j])
                max(0.0, dot(vi, vj))
            else
                error("Unknown mode = $mode")
            end

        weights[(j, i)] = w
    end

    return weights
end

function Couzin_proximity_graph_2d(pos, P; r_prox::Float64 = P.Za)
    N = length(pos)
    g = SimpleGraph(N)

    @inbounds for i in 1:N-1
        pi = pos[i]
        for j in i+1:N
            d = displacement(pi, pos[j], P.L)
            r = norm(d)
            r == 0 && continue
            r <= r_prox && add_edge!(g, i, j)
        end
    end

    return g
end

function Couzin_active_graph_series_2d(out, P; show_progress::Bool = true)
    n = length(out.pos_hist)
    series = Vector{SimpleDiGraph}(undef, n)
    prog = show_progress ? Progress(n; desc="Couzin_active_graph_series_2d", showspeed=true) : nothing

    for k in 1:n
        series[k] = interaction_layers(out.pos_hist[k], out.vel_hist[k], P).active
        show_progress && next!(prog)
    end

    return series
end

function Couzin_node_data_series_2d(out, P; show_progress::Bool = true)
    n = length(out.pos_hist)
    series = Vector{Any}(undef, n)
    prog = show_progress ? Progress(n; desc="Couzin_node_data_series_2d", showspeed=true) : nothing

    for k in 1:n
        graphs = interaction_layers(out.pos_hist[k], out.vel_hist[k], P)
        series[k] = (
            n_rep = graphs.n_rep,
            n_ori = graphs.n_ori,
            n_att = graphs.n_att,
        )
        show_progress && next!(prog)
    end

    return identity.(series)  # narrow Vector{Any} to a concrete Vector{<:NamedTuple}
end

function Couzin_active_weight_series_2d(
    out,
    P;
    mode::Symbol = :zone_strength,
    show_progress::Bool = true,
)
    n = length(out.pos_hist)
    series = Vector{Any}(undef, n)
    prog = show_progress ? Progress(n; desc="Couzin_active_weight_series_2d", showspeed=true) : nothing

    for k in 1:n
        series[k] = Couzin_active_edge_weights_2d(out.pos_hist[k], out.vel_hist[k], P; mode=mode)
        show_progress && next!(prog)
    end

    return identity.(series)  # narrow Vector{Any} to a concrete Vector{<:AbstractDict}
end

function Couzin_proximity_graph_series_2d(
    out,
    P;
    r_prox::Float64 = P.Za,
    show_progress::Bool = true,
)
    n = length(out.pos_hist)
    series = Vector{SimpleGraph}(undef, n)
    prog = show_progress ? Progress(n; desc="Couzin_proximity_graph_series_2d", showspeed=true) : nothing

    for k in 1:n
        series[k] = Couzin_proximity_graph_2d(out.pos_hist[k], P; r_prox=r_prox)
        show_progress && next!(prog)
    end

    return series
end


# --------------------------------------------------
# Couzin edge metadata helpers
# --------------------------------------------------

_strength_rep(r, P::CouzinParams) =
    P.Zr > 0 ? clamp(1.0 - r / P.Zr, 0.0, 1.0) : 1.0

function _strength_ori(r, P::CouzinParams)
    denom = max(P.Zo - P.Zr, eps())
    return clamp(1.0 - (r - P.Zr) / denom, 0.0, 1.0)
end

function _strength_att(r, P::CouzinParams)
    denom = max(P.Za - P.Zo, eps())
    return clamp(1.0 - (r - P.Zo) / denom, 0.0, 1.0)
end


# --------------------------------------------------
# Dimension-specific helpers
# --------------------------------------------------

_unitvec(v::SVector2) = unit(v)
_unitvec(v::SVector3) = unit3(v)

_is_visible(vi_u::SVector2, d_u::SVector2, blind_half_angle) =
    is_visible(vi_u, d_u, blind_half_angle)

_is_visible(vi_u::SVector3, d_u::SVector3, blind_half_angle) =
    is_visible_3d(vi_u, d_u, blind_half_angle)


# --------------------------------------------------
# Couzin edge metadata
# Returns edge list + layer labels + strengths
# --------------------------------------------------

function Couzin_edge_metadata(
    pos::AbstractVector{V},
    vel::AbstractVector{V},
    P::CouzinParams;
    layer::Symbol = :active,
) where {V<:Union{SVector2,SVector3}}

    edges_out = Tuple{Int,Int}[]
    strengths = Float64[]
    layers = Symbol[]

    N = length(pos)

    @inbounds for i in 1:N
        pi   = pos[i]
        vi_u = _unitvec(vel[i])

        rep_edges = Tuple{Int,Float64}[]
        ori_edges = Tuple{Int,Float64}[]
        att_edges = Tuple{Int,Float64}[]

        for j in 1:N
            j == i && continue

            d = displacement(pi, pos[j], P.L)
            r = norm(d)
            r == 0 && continue

            if r <= P.Zr
                push!(rep_edges, (j, _strength_rep(r, P)))
                continue
            end

            d_u = d / r
            _is_visible(vi_u, d_u, P.blind_half_angle) || continue

            if r <= P.Zo
                push!(ori_edges, (j, _strength_ori(r, P)))
            elseif r <= P.Za
                push!(att_edges, (j, _strength_att(r, P)))
            end
        end

        chosen =
            if layer == :rep
                [(j, s, :rep) for (j, s) in rep_edges]
            elseif layer == :ori
                [(j, s, :ori) for (j, s) in ori_edges]
            elseif layer == :att
                [(j, s, :att) for (j, s) in att_edges]
            elseif layer == :all
                vcat(
                    [(j, s, :rep) for (j, s) in rep_edges],
                    [(j, s, :ori) for (j, s) in ori_edges],
                    [(j, s, :att) for (j, s) in att_edges],
                )
            elseif layer == :active
                if !isempty(rep_edges)
                    [(j, s, :rep) for (j, s) in rep_edges]
                else
                    vcat(
                        [(j, s, :ori) for (j, s) in ori_edges],
                        [(j, s, :att) for (j, s) in att_edges],
                    )
                end
            else
                error("Unknown layer = $layer")
            end

        for (j, s, lay) in chosen
            push!(edges_out, (j, i))
            push!(strengths, s)
            push!(layers, lay)
        end
    end

    return (
        edges = edges_out,
        strengths = strengths,
        layers = layers,
    )
end

# --------------------------------------------------
# Convert metadata to plotted segments
# --------------------------------------------------

# ============================================================
# Couzin edge metadata and drawable edge segments
# ============================================================

function Couzin_edge_segments(meta, pos; displacement_fn)
    segs = Point2f[]

    for (j, i) in meta.edges
        p_j = pos[j]
        dji = displacement_fn(p_j, pos[i])
        p_i = p_j + dji

        push!(segs, Point2f(p_j[1], p_j[2]))
        push!(segs, Point2f(p_i[1], p_i[2]))
    end

    return segs
end

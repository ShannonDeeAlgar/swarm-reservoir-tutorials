using Graphs
using DataFrames
using Statistics
using LinearAlgebra
using ProgressMeter

# ============================================================
# Basic helpers
# ============================================================

edge_set(g::AbstractGraph) = Set((src(e), dst(e)) for e in edges(g))

function jaccard_sets(A::Set, B::Set)
    isempty(A) && isempty(B) && return 1.0
    return length(intersect(A, B)) / length(union(A, B))
end

function graph_density(g::SimpleDiGraph)
    n = nv(g)
    n <= 1 && return 0.0
    return ne(g) / (n * (n - 1))
end

function graph_density(g::SimpleGraph)
    n = nv(g)
    n <= 1 && return 0.0
    return 2ne(g) / (n * (n - 1))
end

function giant_component_size(g::SimpleGraph)
    comps = connected_components(g)
    isempty(comps) && return 0
    return maximum(length.(comps))
end

function component_size_per_node(g::SimpleGraph)
    comps = connected_components(g)
    out = zeros(Int, nv(g))
    for comp in comps
        s = length(comp)
        for v in comp
            out[v] = s
        end
    end
    return out
end

function incoming_neighbour_set(g::SimpleDiGraph, i::Int)
    Set(inneighbors(g, i))
end

function outgoing_neighbour_set(g::SimpleDiGraph, i::Int)
    Set(outneighbors(g, i))
end

# ============================================================
# Local graph statistics
# ============================================================

function reciprocity_per_node(g::SimpleDiGraph)
    N = nv(g)
    vals = fill(NaN, N)

    for i in 1:N
        ins  = Set(inneighbors(g, i))
        outs = Set(outneighbors(g, i))
        denom = length(union(ins, outs))
        vals[i] = denom == 0 ? NaN : length(intersect(ins, outs)) / denom
    end

    return vals
end

function weighted_indegree(g::SimpleDiGraph, weights::Dict{Tuple{Int,Int},<:Real})
    N = nv(g)
    vals = zeros(Float64, N)

    for e in edges(g)
        u = src(e)
        v = dst(e)
        vals[v] += get(weights, (u, v), 0.0)
    end

    return vals
end

function weighted_outdegree(g::SimpleDiGraph, weights::Dict{Tuple{Int,Int},<:Real})
    N = nv(g)
    vals = zeros(Float64, N)

    for e in edges(g)
        u = src(e)
        v = dst(e)
        vals[u] += get(weights, (u, v), 0.0)
    end

    return vals
end

function local_clustering_per_node(g::SimpleGraph)
    local_clustering_coefficient(g)
end

function local_stats_digraph(
    g::SimpleDiGraph;
    weights::Union{Nothing,Dict{Tuple{Int,Int},<:Real}} = nothing,
    undirected_for_components::Bool = true,
    proximity_graph::Union{Nothing,SimpleGraph} = nothing,
    node_data::NamedTuple = NamedTuple(),
)
    N = nv(g)
    gu = undirected_for_components ? SimpleGraph(g) : SimpleGraph(g)

    indeg = indegree(g)
    outdeg = outdegree(g)
    recip = reciprocity_per_node(g)
    comp_size = component_size_per_node(gu)

    df = DataFrame(
        node = 1:N,
        indegree = indeg,
        outdegree = outdeg,
        reciprocity = recip,
        component_size = comp_size,
    )

    if weights !== nothing
        df.weighted_indegree = weighted_indegree(g, weights)
        df.weighted_outdegree = weighted_outdegree(g, weights)
    end

    if proximity_graph !== nothing
        df.local_clustering = local_clustering_per_node(proximity_graph)
    end

    for (name, vals) in pairs(node_data)
        df[!, name] = vals
    end

    return df
end


# ============================================================
# Global graph statistics
# ============================================================

function global_reciprocity(g::SimpleDiGraph)
    m = ne(g)
    m == 0 && return NaN

    mutual_twice = 0
    for e in edges(g)
        u = src(e)
        v = dst(e)
        has_edge(g, v, u) && (mutual_twice += 1)
    end

    return mutual_twice / m
end

function algebraic_connectivity_safe(g::SimpleGraph)
    nv(g) <= 1 && return NaN
    is_connected(g) || return 0.0
    vals = eigvals(Matrix(laplacian_matrix(g)))
    vals = sort(real.(vals))
    return length(vals) >= 2 ? vals[2] : NaN
end

function adjacency_spectral_gap_safe(g::SimpleGraph)
    nv(g) <= 1 && return NaN
    vals = eigvals(Matrix(adjacency_matrix(g)))
    vals = sort(abs.(real.(vals)), rev=true)
    return length(vals) >= 2 ? vals[1] - vals[2] : NaN
end

function global_stats_digraph(
    g::SimpleDiGraph;
    weights::Union{Nothing,Dict{Tuple{Int,Int},<:Real}} = nothing,
    proximity_graph::Union{Nothing,SimpleGraph} = nothing,
)
    gu = SimpleGraph(g)

    scc = strongly_connected_components(g)
    scc_sizes = sort!(length.(scc), rev=true)

    stats = (
        mean_indegree = mean(indegree(g)),
        mean_outdegree = mean(outdegree(g)),
        max_indegree = maximum(indegree(g)),
        max_outdegree = maximum(outdegree(g)),
        density = graph_density(g),

        n_components = length(connected_components(gu)),
        giant_component = giant_component_size(gu),
        frac_isolated = mean(degree(gu) .== 0),

        n_strongly_connected_components = length(scc),
        giant_strong_component = isempty(scc_sizes) ? 0 : scc_sizes[1],

        mean_clustering = mean(local_clustering_coefficient(gu)),
        global_reciprocity = global_reciprocity(g),

        algebraic_connectivity = algebraic_connectivity_safe(gu),
        adjacency_spectral_gap = adjacency_spectral_gap_safe(gu),
    )

    if weights === nothing && proximity_graph === nothing
        return stats
    end

    extras = NamedTuple()

    if weights !== nothing
        extras = merge(extras, (
            mean_weighted_indegree = mean(weighted_indegree(g, weights)),
            mean_weighted_outdegree = mean(weighted_outdegree(g, weights)),
        ))
    end

    if proximity_graph !== nothing
        extras = merge(extras, (
            mean_clustering_proximity = mean(local_clustering_coefficient(proximity_graph)),
        ))
    end

    return merge(stats, extras)
end


# ============================================================
# Temporal graph statistics
# ============================================================

function global_edge_turnover(g_prev::SimpleDiGraph, g_curr::SimpleDiGraph)
    E_prev = edge_set(g_prev)
    E_curr = edge_set(g_curr)

    persist = jaccard_sets(E_prev, E_curr)
    turnover = 1.0 - persist

    lost   = isempty(E_prev) ? 0.0 : length(setdiff(E_prev, E_curr)) / length(E_prev)
    gained = isempty(E_curr) ? 0.0 : length(setdiff(E_curr, E_prev)) / length(E_curr)

    return (
        edge_persistence = persist,
        edge_turnover = turnover,
        frac_edges_lost = lost,
        frac_edges_gained = gained,
    )
end

function local_edge_persistence_in(g_prev::SimpleDiGraph, g_curr::SimpleDiGraph)
    N = nv(g_curr)
    vals = zeros(Float64, N)

    for i in 1:N
        A = incoming_neighbour_set(g_prev, i)
        B = incoming_neighbour_set(g_curr, i)
        vals[i] = jaccard_sets(A, B)
    end

    return vals
end

function local_edge_persistence_out(g_prev::SimpleDiGraph, g_curr::SimpleDiGraph)
    N = nv(g_curr)
    vals = zeros(Float64, N)

    for i in 1:N
        A = outgoing_neighbour_set(g_prev, i)
        B = outgoing_neighbour_set(g_curr, i)
        vals[i] = jaccard_sets(A, B)
    end

    return vals
end


# ============================================================
# Time series wrappers
# ============================================================

function local_stats_time_series(
    graphs::AbstractVector{<:SimpleDiGraph};
    t = nothing,
    weights_series::Union{Nothing,AbstractVector{<:AbstractDict}} = nothing,
    proximity_graphs::Union{Nothing,AbstractVector{<:SimpleGraph}} = nothing,
    node_data_series::Union{Nothing,AbstractVector{<:NamedTuple}} = nothing,
    show_progress::Bool = true,
)
    T = length(graphs)
    rows = DataFrame[]

    prog = show_progress ? Progress(T; desc="local_stats_time_series", showspeed=true) : nothing

    for k in 1:T
        weights_k = weights_series === nothing ? nothing : weights_series[k]
        prox_k    = proximity_graphs === nothing ? nothing : proximity_graphs[k]
        node_k    = node_data_series === nothing ? NamedTuple() : node_data_series[k]

        dfk = local_stats_digraph(
            graphs[k];
            weights = weights_k,
            proximity_graph = prox_k,
            node_data = node_k,
        )

        dfk.time_index .= k
        if t !== nothing
            dfk.t .= t[k]
        end

        if k == 1
            dfk.edge_persistence_in  = fill(NaN, nrow(dfk))
            dfk.edge_persistence_out = fill(NaN, nrow(dfk))
        else
            dfk.edge_persistence_in  = local_edge_persistence_in(graphs[k-1], graphs[k])
            dfk.edge_persistence_out = local_edge_persistence_out(graphs[k-1], graphs[k])
        end

        push!(rows, dfk)
        show_progress && next!(prog)
    end

    return vcat(rows...)
end

function global_stats_time_series(
    graphs::AbstractVector{<:SimpleDiGraph};
    t = nothing,
    weights_series::Union{Nothing,AbstractVector{<:AbstractDict}} = nothing,
    proximity_graphs::Union{Nothing,AbstractVector{<:SimpleGraph}} = nothing,
    show_progress::Bool = true,
)
    T = length(graphs)
    rows = NamedTuple[]

    prog = show_progress ? Progress(T; desc="global_stats_time_series", showspeed=true) : nothing

    for k in 1:T
        weights_k = weights_series === nothing ? nothing : weights_series[k]
        prox_k    = proximity_graphs === nothing ? nothing : proximity_graphs[k]

        snap = global_stats_digraph(
            graphs[k];
            weights = weights_k,
            proximity_graph = prox_k,
        )

        temporal =
            if k == 1
                (
                    edge_persistence = NaN,
                    edge_turnover = NaN,
                    frac_edges_lost = NaN,
                    frac_edges_gained = NaN,
                )
            else
                global_edge_turnover(graphs[k-1], graphs[k])
            end

        base = (time_index = k,)
        base = t === nothing ? base : merge(base, (t = t[k],))

        push!(rows, merge(base, snap, temporal))
        show_progress && next!(prog)
    end

    return DataFrame(rows)
end



function clusters_from_graph(g::SimpleDiGraph)
    connected_components(SimpleGraph(g)) #ignore direction
end

function cluster_sizes(g::SimpleDiGraph)
    length.(clusters_from_graph(g))
end

function cluster_labels(g::SimpleDiGraph)
    comps = clusters_from_graph(g)
    labels = zeros(Int, nv(g))

    for (k, comp) in enumerate(comps)
        for i in comp
            labels[i] = k
        end
    end

    return labels
end

function clusters_from_graph_series(graphs::AbstractVector{<:SimpleDiGraph})
    [clusters_from_graph(g) for g in graphs]
end

function cluster_sizes_series(graphs::AbstractVector{<:SimpleDiGraph})
    [cluster_sizes(g) for g in graphs]
end

function cluster_labels_series(graphs::AbstractVector{<:SimpleDiGraph})
    [cluster_labels(g) for g in graphs]
end

# --------------------------------------------------
# Subset helpers
# --------------------------------------------------

subset_state(pos, vel, idxs) = (pos[idxs], vel[idxs])

# --------------------------------------------------
# Order parameters for one specified cluster
# idxs is a vector of node indices
# --------------------------------------------------


function cluster_order_parameters(
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    idxs::AbstractVector{<:Integer},
    L::Float64;
    displacement,
)
    pos_sub, vel_sub = subset_state(pos, vel, idxs)
    dil, rot, pol, mabs = order_parameters_2d(pos_sub, vel_sub, L; displacement=displacement)

    return (
        size = length(idxs),
        nodes = collect(idxs),
        dilation = dil,
        rotation = rot,
        polarisation = pol,
        abs_angular_momentum = mabs,
    )
end

function cluster_order_parameters(
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    idxs::AbstractVector{<:Integer},
    L::Float64;
    displacement,
)
    pos_sub, vel_sub = subset_state(pos, vel, idxs)
    dil, rot, pol, mabs = order_parameters_3d(pos_sub, vel_sub, L; displacement=displacement)

    return (
        size = length(idxs),
        nodes = collect(idxs),
        dilation = dil,
        rotation = rot,
        polarisation = pol,
        abs_angular_momentum = mabs,
    )
end

# --------------------------------------------------
# Order parameters for all clusters in one graph snapshot
# --------------------------------------------------

function all_cluster_order_parameters(
    g::SimpleDiGraph,
    pos,
    vel,
    L::Float64;
    displacement,
    min_size::Int = 1,
)
    comps = clusters_from_graph(g)
    comps = [c for c in comps if length(c) >= min_size]

    rows = NamedTuple[]

    for (cluster_id, idxs) in enumerate(comps)
        stats = cluster_order_parameters(pos, vel, idxs, L; displacement=displacement)

        push!(rows, (
            cluster_id = cluster_id,
            cluster_size = stats.size,
            nodes = stats.nodes,
            dilation = stats.dilation,
            rotation = stats.rotation,
            polarisation = stats.polarisation,
            abs_angular_momentum = stats.abs_angular_momentum,
        ))
    end

    return DataFrame(rows)
end

# --------------------------------------------------
# Order parameters for all clusters through time
# graphs[k] should match pos_hist[k], vel_hist[k]
# --------------------------------------------------

function cluster_order_parameters_time_series(
    graphs::AbstractVector{<:SimpleDiGraph},
    pos_hist,
    vel_hist,
    L::Float64;
    displacement,
    t = nothing,
    min_size::Int = 1,
    show_progress::Bool = true,
)
    T = length(graphs)
    rows = NamedTuple[]

    prog = show_progress ? Progress(T; desc="cluster_order_parameters_time_series", showspeed=true) : nothing

    for k in 1:T
        dfk = all_cluster_order_parameters(
            graphs[k],
            pos_hist[k],
            vel_hist[k],
            L;
            displacement=displacement,
            min_size=min_size,
        )

        if nrow(dfk) > 0
            for row in eachrow(dfk)
                push!(rows, (
                    time_index = k,
                    t = t === nothing ? k : t[k],
                    cluster_id = row.cluster_id,
                    cluster_size = row.cluster_size,
                    nodes = row.nodes,
                    dilation = row.dilation,
                    rotation = row.rotation,
                    polarisation = row.polarisation,
                    abs_angular_momentum = row.abs_angular_momentum,
                ))
            end
        end

        show_progress && next!(prog)
    end

    return DataFrame(rows)
end




function summarise_cluster_order_parameters(
    cluster_order_df::DataFrame;
    tcol::Symbol = :t,
    time_index_col::Symbol = :time_index,
    sizecol::Symbol = :cluster_size,
)
    grouped = groupby(cluster_order_df, [time_index_col, tcol])

    combine(grouped) do sdf
        w = sdf[!, sizecol]
        wsum = sum(w)

        (
            n_clusters = nrow(sdf),

            dilation_mean = mean(sdf.dilation),
            rotation_mean = mean(sdf.rotation),
            polarisation_mean = mean(sdf.polarisation),
            abs_angular_momentum_mean = mean(sdf.abs_angular_momentum),

            dilation_weighted_mean = wsum > 0 ? sum(w .* sdf.dilation) / wsum : NaN,
            rotation_weighted_mean = wsum > 0 ? sum(w .* sdf.rotation) / wsum : NaN,
            polarisation_weighted_mean = wsum > 0 ? sum(w .* sdf.polarisation) / wsum : NaN,
            abs_angular_momentum_weighted_mean = wsum > 0 ? sum(w .* sdf.abs_angular_momentum) / wsum : NaN,

            dilation_std = std(sdf.dilation),
            rotation_std = std(sdf.rotation),
            polarisation_std = std(sdf.polarisation),
            abs_angular_momentum_std = std(sdf.abs_angular_momentum),

            mean_cluster_size = mean(w),
            max_cluster_size = maximum(w),
        )
    end
end


function edge_segments(
    g::SimpleDiGraph,
    pos;
    displacement_fn = (p1, p2) -> p2 - p1,
)
    segs = eltype(pos)[]

    for e in edges(g)
        j = src(e)
        i = dst(e)

        p1 = pos[j]
        d  = displacement_fn(pos[j], pos[i])
        p2 = p1 + d

        push!(segs, p1, p2)
    end

    return segs
end
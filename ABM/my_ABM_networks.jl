
using Graphs
using DataFrames
using Statistics




function edge_segments(g, pos; displacement_fn)
    segs = Point2f[]

    for e in edges(g)
        j = src(e)
        i = dst(e)

        p_j = pos[j]
        dji = displacement_fn(p_j, pos[i])
        p_i = p_j + dji

        push!(segs, Point2f(p_j[1], p_j[2]))
        push!(segs, Point2f(p_i[1], p_i[2]))
    end

    return segs
end


function network_time_series(graph_series; t = nothing)
    T = length(graph_series)
    tvals = isnothing(t) ? collect(1:T) : t

    rows = NamedTuple[]

    for k in 1:T
        g = graph_series[k]

        degs = degree(g)

        push!(rows, (
            time_index = k,
            t = tvals[k],
            n_vertices = nv(g),
            n_edges = ne(g),
            density = nv(g) > 1 ? ne(g) / (nv(g) * (nv(g) - 1)) : 0.0,
            mean_degree = isempty(degs) ? 0.0 : mean(degs),
            max_degree = isempty(degs) ? 0 : maximum(degs),
        ))
    end

    return DataFrame(rows)
end
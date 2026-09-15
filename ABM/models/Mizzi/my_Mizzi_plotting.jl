# Optional visualisation of fixed territories and current agent state.
using CairoMakie

function mizzi_voronoi_polygons(homes, xmin, xmax, ymin, ymax)
    clip_halfplane(poly, nx, ny, bound) = begin
        out = Point2f[]
        isempty(poly) && return out
        for k in eachindex(poly)
            a, b = poly[k], poly[mod1(k + 1, length(poly))]
            fa = nx * a[1] + ny * a[2] - bound
            fb = nx * b[1] + ny * b[2] - bound
            fa <= 1e-10 && push!(out, a)
            if (fa <= 0) != (fb <= 0)
                t = fa / (fa - fb)
                push!(out, Point2f(a[1] + t * (b[1] - a[1]),
                                   a[2] + t * (b[2] - a[2])))
            end
        end
        out
    end
    polygons = Vector{Vector{Point2f}}(undef, length(homes))
    for i in eachindex(homes)
        hi = homes[i]
        poly = Point2f[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for j in eachindex(homes)
            i == j && continue
            hj = homes[j]
            poly = clip_halfplane(poly, 2 * (hj[1] - hi[1]),
                2 * (hj[2] - hi[2]), hj[1]^2 + hj[2]^2 - hi[1]^2 - hi[2]^2)
        end
        polygons[i] = poly
    end
    return polygons
end

"Draw the fixed Mizzi Voronoi cells and their home generators on an existing axis."
function plot_Mizzi_territories!(ax, P::MizziParams;
    extent::Float64=P.plot_extent,
    show_homes::Bool=true,
)
    points = Point2f[(h.x, h.y) for h in P.homes]
    labelled_boundary = false
    for poly in mizzi_voronoi_polygons(P.homes, -extent, extent, -extent, extent)
        isempty(poly) && continue
        closed = copy(poly)
        push!(closed, first(poly))
        lines!(ax, closed; color=(:grey45, 0.7), linewidth=1.2,
            label=labelled_boundary ? nothing : "territory boundaries")
        labelled_boundary = true
    end
    show_homes && scatter!(ax, points;
        marker=:xcross, markersize=9, color=:black, label="homes")
    return ax
end

function plot_Mizzi_state(state::SwarmState{SVector2}, P::MizziParams;
    prey::Union{Nothing,SVector2}=nothing,
    extent::Float64=P.plot_extent,
    fig_size::Tuple{Int,Int}=(650, 600),
)
    fig = Figure(size=fig_size)
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y",
        title="Mizzi agents and fixed Voronoi territories", aspect=DataAspect())
    plot_Mizzi_territories!(ax, P; extent=extent)
    scatter!(ax, getfield.(state.pos, :x), getfield.(state.pos, :y);
        markersize=8, color=:firebrick, label="agents")
    prey !== nothing && scatter!(ax, [prey.x], [prey.y];
        markersize=13, color=:dodgerblue, label="prey")
    xlims!(ax, -extent, extent); ylims!(ax, -extent, extent)
    axislegend(ax, position=:rt)
    return fig
end

function plot_Mizzi_state(res; kwargs...)
    hasproperty(res, :state) && hasproperty(res, :P) ||
        error("Expected a Mizzi reservoir with `state` and `P` fields.")
    return plot_Mizzi_state(res.state, res.P; kwargs...)
end

# layer_basecolor
# layer_alpha
# layer_bin_alpha

# plot_interaction_snapshot_2d!
# plot_interaction_snapshot_2d
# plot_interaction_snapshot_grid_2d

# plot_layer_zone_overlay_2d!

# animate_active_network_alpha_2d
# animate_active_network_alpha_with_zones_2d
# animate_active_network_alpha_with_zones_3d

# classify_neighbours_2d
# plot_focal_interactions_2d


function layer_basecolor(lay::Symbol)
    lay == :rep    && return :crimson
    lay == :ori    && return :darkorange
    lay == :att    && return :dodgerblue
    lay == :active && return :black
    error("Unknown layer = $lay")
end

function layer_alpha(lay::Symbol, s::Real)
    s = clamp(float(s), 0.0, 1.0)

    if lay == :rep
        # Rare and important: make visible even when weak
        return 0.55 + 0.40s

    elseif lay == :ori
        # Often rarer than attraction: emphasise
        return 0.45 + 0.40s

    elseif lay == :att
        # Usually dense: keep faint to avoid washing out the plot
        return 0.05 + 0.18s

    elseif lay == :active
        return 0.25 + 0.35s

    else
        error("Unknown layer = $lay")
    end
end

function layer_bin_alpha(lay::Symbol)
    lay == :rep    && return 0.85
    lay == :ori    && return 0.70
    lay == :att    && return 0.18
    lay == :active && return 0.35
    error("Unknown layer = $lay")
end

function layer_linewidth(lay::Symbol, base::Real = 1.5)
    lay == :rep    && return 2.2base
    lay == :ori    && return 1.7base
    lay == :att    && return 0.7base
    lay == :active && return base
    error("Unknown layer = $lay")
end

function plot_interaction_snapshot_2d!(
    ax,
    pos,
    vel,
    P;
    layer::Symbol = :active,
    edge_style::Symbol = :alpha,      # :alpha, :weighted
    node_color_by::Symbol = :none,    # :none, :indegree
    node_markersize::Real = 10,
    edge_linewidth::Real = 1.5,
    show_colorbar::Bool = false,
    colorbar_slot = nothing,
)
    graphs = interaction_layers(pos, vel, P)
    pts = Point2f[(p[1], p[2]) for p in pos]

    # ------------------------------------------------------------
    # Draw nodes first so short repulsion/orientation edges can sit on top
    # ------------------------------------------------------------
    if node_color_by == :none
        scatter!(
            ax,
            pts;
            color = :black,
            markersize = node_markersize,
        )

    elseif node_color_by == :indegree
        g_nodes = layer == :active ? graphs.active : getproperty(graphs, layer)
        degs = indegree(g_nodes)

        sc = scatter!(
            ax,
            pts;
            color = degs,
            colormap = :viridis,
            markersize = node_markersize,
        )

        if show_colorbar
            colorbar_slot === nothing && error("show_colorbar=true requires colorbar_slot")
            Colorbar(colorbar_slot, sc, label = "in-degree")
        end

    else
        error("Unknown node_color_by = $node_color_by. Use :none or :indegree.")
    end

    if edge_style == :alpha
        meta = Couzin_edge_metadata(pos, vel, P; layer = layer)
        segs = Couzin_edge_segments(
            meta,
            pos;
            displacement_fn = (p1, p2) -> displacement(p1, p2, P.L),
        )
        segs_plot = Point2f[(p[1], p[2]) for p in segs]

        for lay in (:att, :ori, :rep)
            idx = findall(==(lay), meta.layers)
            isempty(idx) && continue

            segs_lay = Point2f[]
            colors_lay = Any[]

            for k in idx
                push!(segs_lay, segs_plot[2k - 1], segs_plot[2k])

                s = meta.strengths[k]
                α = layer_alpha(lay, s)
                base = layer_basecolor(lay)

                push!(colors_lay, (base, α))
                push!(colors_lay, (base, α))
            end

            linesegments!(
                ax,
                segs_lay,
                color = colors_lay,
                linewidth = layer_linewidth(lay, edge_linewidth),
            )
        end

    elseif edge_style == :weighted
        meta = Couzin_edge_metadata(pos, vel, P; layer = layer)
        segs = Couzin_edge_segments(
            meta,
            pos;
            displacement_fn = (p1, p2) -> displacement(p1, p2, P.L),
        )
        segs_plot = Point2f[(p[1], p[2]) for p in segs]

        bins = [0.0, 0.33, 0.66, 1.01]
        widths_for_layer(lay) =
            lay == :rep ? [1.6, 2.4, 3.2] :
            lay == :ori ? [1.2, 1.8, 2.6] :
            lay == :att ? [0.25, 0.45, 0.75] :
            [0.5, 1.0, 2.0]

        for lay in (:att, :ori, :rep)
            idx_layer = findall(==(lay), meta.layers)
            isempty(idx_layer) && continue

            α = layer_bin_alpha(lay)
            base = layer_basecolor(lay)

            for b in 1:3
                lo, hi = bins[b], bins[b + 1]
                idx = [k for k in idx_layer if lo <= meta.strengths[k] < hi]
                isempty(idx) && continue

                segs_bin = Point2f[]
                for k in idx
                    push!(segs_bin, segs_plot[2k - 1], segs_plot[2k])
                end

                linesegments!(
                    ax,
                    segs_bin,
                    color = (base, α),
                    linewidth = widths_for_layer(lay)[b],
                )
            end
        end

    else
        error("Unknown edge_style = $edge_style. Use :alpha, or :weighted.")
    end

    xlims!(ax, 0, P.L)
    ylims!(ax, 0, P.L)

    return ax
end


function plot_interaction_snapshot_2d(
    pos,
    vel,
    P;
    layer::Symbol = :active,
    edge_style::Symbol = :alpha,
    node_color_by::Symbol = :none,
    fig_size::Tuple{Int,Int} = (300, 300),
    node_markersize::Real = 10,
    edge_linewidth::Real = 1.5,
)
    fig = Figure(size = fig_size)
    ax = Axis(fig[1, 1], aspect = DataAspect(), title = "Layer = $(layer)")

    show_cb = node_color_by == :indegree

    plot_interaction_snapshot_2d!(
        ax,
        pos,
        vel,
        P;
        layer = layer,
        edge_style = edge_style,
        node_color_by = node_color_by,
        node_markersize = node_markersize,
        edge_linewidth = edge_linewidth,
        show_colorbar = show_cb,
        colorbar_slot = show_cb ? fig[1, 2] : nothing,
    )

    return fig
end

function plot_interaction_snapshot_grid_2d(
    pos,
    vel,
    P;
    layers = (:active, :rep, :ori, :att),
    edge_style::Symbol = :alpha,
    node_color_by::Symbol = :indegree,
    fig_size::Tuple{Int,Int} = (600, 600),
    node_markersize::Real = 10,
    edge_linewidth::Real = 1.5,
    focal::Union{Nothing,Int} = nothing,
    show_zones::Bool = true,
    zone_alpha::Float64 = 0.12,
    zone_stroke::Float64 = 1.0,
    zone_npoints::Int = 120,
    highlight_focal::Bool = true,
)
    fig = Figure(size = fig_size)

    for (k, lay) in enumerate(layers)
        i = div(k - 1, 2) + 1
        j = mod(k - 1, 2) + 1

        ax = Axis(fig[i, j], aspect = DataAspect(), title = "Layer = $(lay)")

        plot_interaction_snapshot_2d!(
            ax,
            pos,
            vel,
            P;
            layer = lay,
            edge_style = edge_style,
            node_color_by = node_color_by,
            node_markersize = node_markersize,
            edge_linewidth = edge_linewidth,
            show_colorbar = false,
        )

        if show_zones && !isnothing(focal)
            plot_layer_zone_overlay_2d!(
                ax,
                pos,
                vel,
                P,
                focal;
                layer = lay,
                alpha = zone_alpha,
                strokewidth = zone_stroke,
                npoints = zone_npoints,
                highlight_focal = highlight_focal,
                focal_markersize = node_markersize * 1.5,
            )
        end
    end

    return fig
end


function plot_layer_zone_overlay_2d!(
    ax,
    pos,
    vel,
    P,
    focal::Int;
    layer::Symbol = :active,
    alpha::Float64 = 0.12,
    strokewidth::Float64 = 1.0,
    npoints::Int = 120,
    highlight_focal::Bool = true,
    focal_markersize::Real = 14,
)
    c = Point2f(pos[focal][1], pos[focal][2])

    β = hasproperty(P, :blind_half_angle) ? P.blind_half_angle : 0.0
    v̂ = unit(vel[focal])
    ϕ = atan(v̂[2], v̂[1])

    θ1 = ϕ - (π - β)
    θ2 = ϕ + (π - β)

    # focal highlight
    if highlight_focal
        scatter!(ax, [c], color = :black, markersize = focal_markersize)
    end

    # no blind zone -> full discs/annuli
    if β == 0
        if layer == :rep
            zr = circle_points(c, P.Zr; n = npoints)
            poly!(ax, zr; color = (:crimson, alpha), strokecolor = (:crimson, 0.8), strokewidth = strokewidth)

        elseif layer == :ori
            zo = annulus_points(c, P.Zr, P.Zo; n = npoints)
            poly!(ax, zo; color = (:darkorange, alpha), strokecolor = (:darkorange, 0.75), strokewidth = strokewidth)

        elseif layer == :att
            za = annulus_points(c, P.Zo, P.Za; n = npoints)
            poly!(ax, za; color = (:dodgerblue, alpha), strokecolor = (:dodgerblue, 0.75), strokewidth = strokewidth)

        elseif layer == :active
            zr = circle_points(c, P.Zr; n = npoints)
            zo = annulus_points(c, P.Zr, P.Zo; n = npoints)
            za = annulus_points(c, P.Zo, P.Za; n = npoints)

            poly!(ax, za; color = (:dodgerblue, 0.6alpha),  strokecolor = (:dodgerblue, 0.65), strokewidth = strokewidth)
            poly!(ax, zo; color = (:darkorange, 0.8alpha), strokecolor = (:darkorange, 0.7), strokewidth = strokewidth)
            poly!(ax, zr; color = (:crimson, alpha),       strokecolor = (:crimson, 0.8), strokewidth = strokewidth)
        end

        return ax
    end

    # blind zone present -> sectors / annular sectors
    if layer == :rep
        zr = circle_points(c, P.Zr; n = npoints)
        poly!(ax, zr; color = (:crimson, alpha), strokecolor = (:crimson, 0.8), strokewidth = strokewidth)

    elseif layer == :ori
        zo = annulus_sector_points(c, P.Zr, P.Zo, θ1, θ2; n = max(60, npoints))
        poly!(ax, zo; color = (:darkorange, alpha), strokecolor = (:darkorange, 0.75), strokewidth = strokewidth)

    elseif layer == :att
        za = annulus_sector_points(c, P.Zo, P.Za, θ1, θ2; n = max(60, npoints))
        poly!(ax, za; color = (:dodgerblue, alpha), strokecolor = (:dodgerblue, 0.75), strokewidth = strokewidth)

    elseif layer == :active
        zr = circle_points(c, P.Zr; n = npoints)
        zo = annulus_sector_points(c, P.Zr, P.Zo, θ1, θ2; n = max(60, npoints))
        za = annulus_sector_points(c, P.Zo, P.Za, θ1, θ2; n = max(60, npoints))

        poly!(ax, za; color = (:dodgerblue, 0.6alpha),  strokecolor = (:dodgerblue, 0.65), strokewidth = strokewidth)
        poly!(ax, zo; color = (:darkorange, 0.8alpha), strokecolor = (:darkorange, 0.7), strokewidth = strokewidth)
        poly!(ax, zr; color = (:crimson, alpha),       strokecolor = (:crimson, 0.8), strokewidth = strokewidth)
    end

    return ax
end


function animate_active_network_alpha_2d(
    out,
    P;
    stride::Int = 10,
    filename::AbstractString = "active_network_alpha.mp4",
)

    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="Layer = active")

    edgeplot = linesegments!(
        ax,
        Point2f[],
        color = RGBAf[],
        linewidth = 1.5,
    )

    nodeplot = scatter!(
        ax,
        Point2f[];
        color = Float64[],
        colormap = :viridis,
        colorrange = (0, P.N - 1),
        markersize = 10,
    )
    Colorbar(fig[1, 2], nodeplot, label="in-degree")

    xlims!(ax, 0, P.L)
    ylims!(ax, 0, P.L)

    frames = collect(1:stride:length(out.pos_hist))
    prog = Progress(length(frames); desc="Saving network animation", dt=0.5)

    record(fig, joinpath(ANIM_NET_DIR, filename), frames) do k
        pos = out.pos_hist[k]
        vel = out.vel_hist[k]

        pts = Point2f[(p[1], p[2]) for p in pos]
        edges = interaction_edges_2d(pos, vel, P; layer=:active)
        graphs = interaction_layers(pos, vel, P)
        degs = Float64.(indegree(graphs.active))

        segs_lay = Point2f[]
        colors_lay = Any[]

        for edge_idx in eachindex(edges.layers)
            lay = edges.layers[edge_idx]
            s   = edges.strengths[edge_idx]

            α =
                if lay == :rep
                    0.25 + 0.55 * s
                elseif lay == :ori
                    0.12 + 0.35 * s
                else
                    0.03 + 0.12 * s
                end

            basecolor =
                lay == :rep ? :red :
                lay == :ori ? :yellow :
                              :blue

            push!(colors_lay, (basecolor, edge_alpha))
            push!(colors_lay, (basecolor, edge_alpha))

            push!(segs_lay, edges.segs[2edge_idx - 1], edges.segs[2edge_idx])
        end

        edgeplot[1] = segs_lay
        edgeplot.color = colors_lay

        nodeplot[1] = pts
        nodeplot.color = degs

        ax.title = "Layer = active,  t = $(round(out.t[k], digits=2))"

        next!(prog)
    end

    return fig
end




function animate_active_network_alpha_with_zones_3d(
    out,
    P;
    stride::Int = 10,
    filename::AbstractString = "active_network_3d.mp4",
    focal::Int = out.focal,
    fig_size::Tuple{Int,Int} = (1200, 900),
    agent_ms::Float64 = 0.35,
    edge_width::Float64 = 1.0,
    show_zones::Bool = true,
    zone_linewidth::Float64 = 1.0,
    fps::Int = 20,

    arrow_shaft_len::Float64 = 0.35,
    arrow_linewidth::Float64 = 0.8,
    arrow_size::Float64 = 2.0,
    arrow_alpha::Float64 = 0.35,
    show_nodes::Bool = true,

    arrow_colour_by::Union{Nothing,Symbol} = nothing,   # nothing, :indegree, :outdegree, :degree
    arrow_colormap = :viridis,
)

    fig = Figure(size = fig_size)
    ax = Axis3(
        fig[1, 1],
        title = "Layer = active",
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        aspect = (1, 1, 1),
    )

    xlims!(ax, 0, P.L)
    ylims!(ax, 0, P.L)
    zlims!(ax, 0, P.L)

    # --------------------------------------------------
    # Initial frame data
    # --------------------------------------------------
    k0 = 1
    pos0 = out.pos_hist[k0]
    vel0 = out.vel_hist[k0]

    pts0 = Point3f[Point3f(p[1], p[2], p[3]) for p in pos0]
    apos0, adir0 = arrows_from_3d(pos0, vel0; shaft_len = arrow_shaft_len)

    graphs0 = interaction_layers(pos0, vel0, P)
    g0 = graphs0.active
    indegs0 = Float64.(indegree(g0))

    # --------------------------------------------------
    # Edge plot
    # --------------------------------------------------
    edgeplot = linesegments!(
        ax,
        Point3f[],
        color = Any[],
        linewidth = edge_width,
    )

    # --------------------------------------------------
    # Node observables + plot
    # --------------------------------------------------
    node_pos_obs = Observable(pts0)
    node_col_obs = Observable(indegs0)

    nodeplot = nothing
    if show_nodes
        nodeplot = meshscatter!(
            ax,
            node_pos_obs;
            color = node_col_obs,
            colormap = :viridis,
            colorrange = (0, P.N - 1),
            markersize = agent_ms,
        )
        Colorbar(fig[1, 2], nodeplot, label = "in-degree")
    end

    # --------------------------------------------------
    # Arrow observables + plot
    # --------------------------------------------------
    apos_obs = Observable(apos0)
    adir_obs = Observable(adir0)

    function arrow_values(g::SimpleDiGraph, mode)
        if mode === nothing
            return nothing
        elseif mode == :indegree
            return Float64.(indegree(g))
        elseif mode == :outdegree
            return Float64.(outdegree(g))
        elseif mode == :degree
            return Float64.(indegree(g) .+ outdegree(g))
        else
            error("Unknown arrow_colour_by = $mode")
        end
    end

    function values_to_rgba(vals, cmap_symbol, α)
        if vals === nothing
            return fill(RGBAf(0, 0, 0, α), P.N)
        end

        vmax = max(maximum(vals), 1.0)
        cmap = to_colormap(cmap_symbol)

        return [
            begin
                u = clamp(v / vmax, 0.0, 1.0)
                idx = clamp(1 + round(Int, u * (length(cmap) - 1)), 1, length(cmap))
                c = cmap[idx]
                RGBAf(red(c), green(c), blue(c), α)
            end
            for v in vals
        ]
    end

    acol0 = values_to_rgba(arrow_values(g0, arrow_colour_by), arrow_colormap, arrow_alpha)
    acol_obs = Observable(acol0)

    arrows!(
        ax,
        apos_obs,
        adir_obs;
        linewidth = arrow_linewidth,
        arrowsize = arrow_size,
        color = acol_obs,
    )

    # --------------------------------------------------
    # Zones
    # --------------------------------------------------
    zone_state = nothing
    if show_zones
        c0 = Point3f(pos0[focal][1], pos0[focal][2], pos0[focal][3])

        zr_curves = sphere_wireframe_curves(c0, P.Zr; nθ = 30, nϕ = 6, nλ = 6)
        zo_curves = sphere_wireframe_curves(c0, P.Zo; nθ = 30, nϕ = 6, nλ = 6)
        za_curves = sphere_wireframe_curves(c0, P.Za; nθ = 30, nϕ = 6, nλ = 6)

        zr_obs = [Observable(curve) for curve in zr_curves]
        zo_obs = [Observable(curve) for curve in zo_curves]
        za_obs = [Observable(curve) for curve in za_curves]

        [lines!(ax, obs; color = (:red, 0.25), linewidth = zone_linewidth) for obs in zr_obs]
        [lines!(ax, obs; color = (:yellow, 0.25), linewidth = zone_linewidth) for obs in zo_obs]
        [lines!(ax, obs; color = (:blue, 0.25), linewidth = zone_linewidth) for obs in za_obs]

        zone_state = (zr = zr_obs, zo = zo_obs, za = za_obs)
    end

    # --------------------------------------------------
    # Animation loop
    # --------------------------------------------------
    frames = collect(1:stride:length(out.pos_hist))
    prog = Progress(length(frames); desc = "Saving 3D network animation", dt = 0.5)

    record(fig, joinpath(ANIM_NET_DIR, filename), frames; framerate = fps) do k
        pos = out.pos_hist[k]
        vel = out.vel_hist[k]

        pts = Point3f[Point3f(p[1], p[2], p[3]) for p in pos]
        apos, adir = arrows_from_3d(pos, vel; shaft_len = arrow_shaft_len)

        graphs = interaction_layers(pos, vel, P)
        g = graphs.active
        indegs = Float64.(indegree(g))

        meta = Couzin_edge_metadata(pos, vel, P; layer = :active)
        segs = Couzin_edge_segments(
            meta,
            pos;
            displacement_fn = (p1, p2) -> displacement(p1, p2, P.L),
        )
        segs_plot = Point3f[(p[1], p[2], p[3]) for p in segs]

        segs_lay = Point3f[]
        colors_lay = Any[]

        for edge_idx in eachindex(meta.layers)
            lay = meta.layers[edge_idx]
            s   = meta.strengths[edge_idx]

            α =
                if lay == :rep
                    0.25 + 0.55 * s
                elseif lay == :ori
                    0.12 + 0.35 * s
                else
                    0.03 + 0.12 * s
                end

            basecolor =
                lay == :rep ? :crimson :
                lay == :ori ? :darkorange :
                              :dodgerblue

            push!(segs_lay, segs_plot[2edge_idx - 1], segs_plot[2edge_idx])
            push!(colors_lay, (basecolor, α))
            push!(colors_lay, (basecolor, α))
        end

        edgeplot[1] = segs_lay
        edgeplot.color = colors_lay

        if show_nodes
            node_pos_obs.val = pts
            node_col_obs.val = indegs
            notify(node_pos_obs)
        end

        arrow_cols = values_to_rgba(arrow_values(g, arrow_colour_by), arrow_colormap, arrow_alpha)

        apos_obs.val = apos
        adir_obs.val = adir
        acol_obs.val = arrow_cols
        notify(apos_obs)

        if zone_state !== nothing
            cf = pos[focal]
            ck = Point3f(cf[1], cf[2], cf[3])

            zr_curves = sphere_wireframe_curves(ck, P.Zr; nθ = 60, nϕ = 10, nλ = 10)
            zo_curves = sphere_wireframe_curves(ck, P.Zo; nθ = 60, nϕ = 10, nλ = 10)
            za_curves = sphere_wireframe_curves(ck, P.Za; nθ = 60, nϕ = 10, nλ = 10)

            for i in eachindex(zone_state.zr)
                zone_state.zr[i][] = zr_curves[i]
            end
            for i in eachindex(zone_state.zo)
                zone_state.zo[i][] = zo_curves[i]
            end
            for i in eachindex(zone_state.za)
                zone_state.za[i][] = za_curves[i]
            end
        end

        ax.title = "Layer = active, focal = $(focal), t = $(round(out.t[k], digits = 2))"
        next!(prog)
    end

    return fig
end


function animate_active_network_alpha_with_zones_2d(
    out,
    P;
    stride::Int = 10,
    filename::AbstractString = "active_network_alpha_zones.mp4",
    focal::Int = out.focal,
    show_zones::Bool = true,
    highlight_focal::Bool = true,
    zone_npoints::Int = 200,
    zone_alpha::Float64 = 0.12,
    zone_stroke::Float64 = 2.0,
)

    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="Layer = active")

    edgeplot = linesegments!(
        ax,
        Point2f[],
        color = Any[],
        linewidth = 1.5,
    )

    nodeplot = scatter!(
        ax,
        Point2f[];
        color = Float64[],
        colormap = :viridis,
        colorrange = (0, P.N - 1),
        markersize = 10,
    )
    Colorbar(fig[1, 2], nodeplot, label="in-degree")

    xlims!(ax, 0, P.L)
    ylims!(ax, 0, P.L)

    pos0 = out.pos_hist[1]
    vel0 = out.vel_hist[1]

    zone_state = nothing
    if show_zones
        c0 = Point2f(pos0[focal].x, pos0[focal].y)

        Zr = P.Zr
        Zo = P.Zo
        Za = P.Za

        β = P.blind_half_angle
        v0 = vel0[focal]
        ϕ0 = angle_of(unit(v0))

        θ1 = ϕ0 - (π - β)
        θ2 = ϕ0 + (π - β)

        zr_poly = Observable(circle_points(c0, Zr; n=zone_npoints))
        zo_poly = β > 0 ?
            Observable(annulus_sector_points(c0, Zr, Zo, θ1, θ2; n=max(60, zone_npoints))) :
            Observable(annulus_points(c0, Zr, Zo; n=zone_npoints))
        za_poly = β > 0 ?
            Observable(annulus_sector_points(c0, Zo, Za, θ1, θ2; n=max(60, zone_npoints))) :
            Observable(annulus_points(c0, Zo, Za; n=zone_npoints))

        poly!(ax, za_poly; color = (:blue, zone_alpha * 0.7),     strokecolor = (:blue, 0.6),      strokewidth = zone_stroke)
        poly!(ax, zo_poly; color = (:yellow, zone_alpha),         strokecolor = (:goldenrod, 0.9), strokewidth = zone_stroke)
        poly!(ax, zr_poly; color = (:red, zone_alpha),            strokecolor = (:red, 0.85),      strokewidth = zone_stroke)

        focal_obs = nothing
        if highlight_focal
            focal_obs = Observable(Point2f[Point2f(c0[1], c0[2])])
            scatter!(ax, focal_obs; markersize = 16, color = :black)
        end

        zone_state = (zr = zr_poly, zo = zo_poly, za = za_poly, focal_obs = focal_obs, β = β)
    end

    frames = collect(1:stride:length(out.pos_hist))
    prog = Progress(length(frames); desc="Saving network animation", dt=0.5)

    record(fig, joinpath(ANIM_NET_DIR, filename), frames) do k
        pos = out.pos_hist[k]
        vel = out.vel_hist[k]

        pts = Point2f[(p[1], p[2]) for p in pos]
        graphs = interaction_layers(pos, vel, P)
        degs = Float64.(indegree(graphs.active))

        meta = Couzin_edge_metadata(pos, vel, P; layer = :active)
        segs = Couzin_edge_segments(
            meta,
            pos;
            displacement_fn = (p1, p2) -> displacement(p1, p2, P.L),
        )
        segs_plot = Point2f[(p[1], p[2]) for p in segs]

        segs_lay = Point2f[]
        colors_lay = Any[]

        for edge_idx in eachindex(meta.layers)
            lay = meta.layers[edge_idx]
            s   = meta.strengths[edge_idx]

            α =
                if lay == :rep
                    0.25 + 0.55 * s
                elseif lay == :ori
                    0.12 + 0.35 * s
                else
                    0.03 + 0.12 * s
                end

            basecolor =
                lay == :rep ? :crimson :
                lay == :ori ? :darkorange :
                              :dodgerblue

            push!(segs_lay, segs_plot[2edge_idx - 1], segs_plot[2edge_idx])
            push!(colors_lay, (basecolor, α))
            push!(colors_lay, (basecolor, α))
        end

        edgeplot[1] = segs_lay
        edgeplot.color = colors_lay

        nodeplot[1] = pts
        nodeplot.color = degs

        if zone_state !== nothing
            ck = Point2f(pos[focal].x, pos[focal].y)

            Zr = P.Zr
            Zo = P.Zo
            Za = P.Za

            β = zone_state.β
            zone_state.zr[] = circle_points(ck, Zr; n=zone_npoints)

            if β > 0
                vk = vel[focal]
                ϕk = angle_of(unit(vk))
                θ1 = ϕk - (π - β)
                θ2 = ϕk + (π - β)

                zone_state.zo[] = annulus_sector_points(ck, Zr, Zo, θ1, θ2; n=max(60, zone_npoints))
                zone_state.za[] = annulus_sector_points(ck, Zo, Za, θ1, θ2; n=max(60, zone_npoints))
            else
                zone_state.zo[] = annulus_points(ck, Zr, Zo; n=zone_npoints)
                zone_state.za[] = annulus_points(ck, Zo, Za; n=zone_npoints)
            end

            if zone_state.focal_obs !== nothing
                zone_state.focal_obs[] = Point2f[Point2f(ck[1], ck[2])]
            end
        end

        ax.title = "Layer = active, focal = $(focal), t = $(round(out.t[k], digits=2))"

        next!(prog)
    end

    return fig
end


function classify_neighbours_2d(
    i::Int,
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::CouzinParams,
)
    pi   = pos[i]
    vi_u = unit(vel[i])

    rep = Int[]
    ori = Int[]
    att = Int[]
    other = Int[]

    for j in eachindex(pos)
        j == i && continue

        d = displacement(pi, pos[j], P.L)
        r = norm(d)
        r == 0 && continue

        if r <= P.Zr
            push!(rep, j)
            continue
        end

        d_u = d / r
        if !is_visible(vi_u, d_u, P.blind_half_angle)
            push!(other, j)
            continue
        end

        if r <= P.Zo
            push!(ori, j)
        elseif r <= P.Za
            push!(att, j)
        else
            push!(other, j)
        end
    end

    return (rep=rep, ori=ori, att=att, other=other)
end

function plot_focal_interactions_2d(
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::CouzinParams;
    focal::Int = 1,
    show_edges::Bool = true,
    title::AbstractString = "Focal agent interaction zones",
)
    cls = classify_neighbours_2d(focal, pos, vel, P)

    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)

    xlims!(ax, 0, P.L)
    ylims!(ax, 0, P.L)

    # --- all agents in background
    pts_all = Point2f[(p[1], p[2]) for p in pos]
    scatter!(ax, pts_all, color=(:grey, 0.35), markersize=8)

    # --- focal agent
    pf = pos[focal]
    vf = vel[focal]
    pf_pt = Point2f(pf[1], pf[2])

    scatter!(ax, [pf_pt], color=:black, markersize=16)

    # focal heading arrow
    apos_obs = Observable(Point3f[])
    adir_obs = Observable(Vec3f[])
    arrowcolor_obs = Observable(Float64[])

    arrows!(
        ax,
        apos_obs,
        adir_obs;
        color = arrowcolor_obs,
        colormap = :viridis,
        colorrange = (0, P.N - 1),
        arrowsize = 0.25,
        lengthscale = 1.0,
    )

    # --- zones
    β = P.blind_half_angle
    ϕ = atan(vf[2], vf[1])

    if β > 0
        θ1 = ϕ - (π - β)
        θ2 = ϕ + (π - β)

        zr_poly = circle_points(pf_pt, P.Zr; n=200)
        zo_poly = annulus_sector_points(pf_pt, P.Zr, P.Zo, θ1, θ2; n=200)
        za_poly = annulus_sector_points(pf_pt, P.Zo, P.Za, θ1, θ2; n=200)
    else
        zr_poly = circle_points(pf_pt, P.Zr; n=200)
        zo_poly = annulus_points(pf_pt, P.Zr, P.Zo; n=200)
        za_poly = annulus_points(pf_pt, P.Zo, P.Za; n=200)
    end

    poly!(ax, zr_poly; color=(:red,    0.18), strokecolor=(:red,    0.9), strokewidth=2)
    poly!(ax, zo_poly; color=(:yellow, 0.16), strokecolor=(:goldenrod, 0.9), strokewidth=2)
    poly!(ax, za_poly; color=(:blue,   0.08), strokecolor=(:blue,   0.7), strokewidth=2)

    # --- neighbours by class
    function pts(idxs)
        Point2f[(pos[j][1], pos[j][2]) for j in idxs]
    end

    if !isempty(cls.other)
        scatter!(ax, pts(cls.other), color=(:grey, 0.5), markersize=8)
    end
    if !isempty(cls.att)
        scatter!(ax, pts(cls.att), color=:blue, markersize=11)
    end
    if !isempty(cls.ori)
        scatter!(ax, pts(cls.ori), color=:yellow, markersize=11)
    end
    if !isempty(cls.rep)
        scatter!(ax, pts(cls.rep), color=:red, markersize=11)
    end

    # --- optional edges from neighbour to focal
    if show_edges
        # function edge_segments(idxs)
        #     segs = Point2f[]
        #     for j in idxs
        #         p1 = Point2f(pos[j][1], pos[j][2])
        #         dji = displacement(pos[j], pos[focal], P.L)
        #         p2 = p1 + Point2f(dji[1], dji[2])
        #         push!(segs, p1, p2)
        #     end
        #     segs
        # end

        if !isempty(cls.att)
            linesegments!(ax, edge_segments(cls.att), color=(:blue, 0.25), linewidth=1.5)
        end
        if !isempty(cls.ori)
            linesegments!(ax, edge_segments(cls.ori), color=(:yellow, 0.45), linewidth=2)
        end
        if !isempty(cls.rep)
            linesegments!(ax, edge_segments(cls.rep), color=(:red, 0.6), linewidth=2.5)
        end
    end

    Label(
        fig[1, 2],
        "focal = $(focal)\nrep = $(length(cls.rep))\nori = $(length(cls.ori))\natt = $(length(cls.att))\nother = $(length(cls.other))",
        tellheight=false
    )

    return fig
end
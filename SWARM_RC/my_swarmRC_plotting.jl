# ============================================================
# my_swarmRC_plotting.jl
# Optional visualisation helpers for the swarm-reservoir pipeline
# (my_swarmRC.jl). Not required to run or train a swarm-reservoir; load
# this only if you want animations / kernel diagnostics plots.
# ============================================================
#
# Load after ABM/load_ABM.jl (for circle_points/_agent_trail_segments/
# plot_swarm_frame!, defined in ABM/my_ABM_plotting.jl) and my_swarmRC.jl.
# Requires CairoMakie and ProgressMeter.

using CairoMakie, ProgressMeter

@inline kernel_sigma(inv2w::Float64) = sqrt(1 / (2 * inv2w))

"""
    animate_from_states(S, L; pred_hist=nothing, rp, center=(0.0,0.0), savepath="swarm_from_states.mp4", ...)

Render an mp4 from a vector of logged reservoir states `S` (each element as
returned by `raw_state`, i.e. a NamedTuple with fields `P` (2×N positions)
and `V` (2×N velocities)).

If `pred_hist` is given (a vector of predator positions with the same
length as `S`, e.g. `run_offline_predator_drive!(...).pred_hist`), the
predator is drawn as a red dot with a fading trail of its recent positions,
and -- if `show_influence_radius=true` -- a ring of radius `rp` around it
(pass the same value used in your `InputCoupling`, e.g.
`PredatorCoupling.rp`, if you want the ring to mean something). Without
`pred_hist`, only the swarm is drawn.

Every agent also gets its own short trailing path (`show_agent_trails=true`
by default, length `agent_trail_len`) -- this is what makes swirling/milling
motion visible at a glance (vs. a cloud of dots that only shows
instantaneous positions); set `show_agent_trails=false` for the old
dots-only look.

The plotted window is `[center[1], center[1]+L] x [center[2], center[2]+L]`.
The default `center=(0.0, 0.0)` matches `CouzinReservoir`'s periodic
`[0,L]x[0,L]` domain unchanged. For a `LymburnReservoir` (no periodic
domain -- the swarm roams around its home point, the origin, by default),
pass `L=2*P.plot_extent, center=(-P.plot_extent, -P.plot_extent)` to centre
the view on the origin instead.
"""
function animate_from_states(
    S, L;
    pred_hist = nothing,
    predator_color = :red,
    every::Int = 2,
    agent_ms::Float64 = 5.0,
    agent_color = :black,
    color_by_heading::Bool = false,
    heading_colormap = :vikO,
    agent_state_hist = nothing,
    savepath::String = "swarm_from_states.mp4",
    show_influence_radius::Bool = true,
    rp::Float64 = 0.0,
    ring_pts::Int = 200,
    trail_len::Int = 60,
    show_agent_dirs::Bool = true,
    dir_stride::Int = 1,
    dir_len::Union{Nothing,Float64} = nothing,
    dir_linewidth::Float64 = 1.6,
    center::Tuple{<:Real,<:Real} = (0.0, 0.0),
    show_agent_trails::Bool = true,
    agent_trail_len::Int = 25,
    agent_trail_color = (:grey40, 0.3),
    show_grid::Bool = false,
    territory_homes = nothing,
)
    @assert !isempty(S)
    @assert hasproperty(S[1], :P) "State must contain :P (2×N positions)."
    if show_agent_dirs || color_by_heading
        @assert hasproperty(S[1], :V) "State must contain :V (2×N velocities)."
    end

    T = length(S)
    has_pred = pred_hist !== nothing
    has_agent_states = agent_state_hist !== nothing
    if has_pred
        @assert length(pred_hist) == T "pred_hist must have the same length as S."
    end
    has_agent_states && @assert length(agent_state_hist) == T

    P0 = S[1].P
    V0 = (show_agent_dirs || color_by_heading) ? S[1].V : nothing
    dir_len_val = isnothing(dir_len) ? 0.04 * L : dir_len

    fig = Figure(size=color_by_heading ? (850, 910) : (850, 850))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Swarm-reservoir evolution (logged states)")
    ax.xgridvisible = show_grid
    ax.ygridvisible = show_grid
    xlims!(ax, center[1], center[1] + L); ylims!(ax, center[2], center[2] + L)
    ax.aspect = DataAspect()

    if territory_homes !== nothing
        points = Point2f[(h[1], h[2]) for h in territory_homes]
        labelled_boundary = false
        for poly in mizzi_voronoi_polygons(territory_homes,
            center[1], center[1] + L, center[2], center[2] + L)
            isempty(poly) && continue
            closed = copy(poly)
            push!(closed, first(poly))
            lines!(ax, closed; color=(:grey45, 0.65), linewidth=1.1,
                label=labelled_boundary ? nothing : "territory boundaries")
            labelled_boundary = true
        end
        scatter!(ax, points;
            marker=:xcross, markersize=8, color=:black, label="homes")
    end

    agent_trail_hist = Matrix{Float64}[]
    agent_trail_obs = nothing
    if show_agent_trails
        push!(agent_trail_hist, P0)
        agent_trail_obs = Observable(_agent_trail_segments(agent_trail_hist))
        lines!(ax, agent_trail_obs; color=agent_trail_color, linewidth=1.0)
    end

    agent_pos = Observable(P0)
    agent_vel = V0 === nothing ? nothing : Observable(V0)
    all_points = lift(agent_pos) do P
        Point2f.(P[1, :], P[2, :])
    end
    agent_colors = has_agent_states ?
        Observable([s == 0 ? :dodgerblue : :firebrick for s in agent_state_hist[1]]) :
        (color_by_heading ? lift(V -> atan.(V[2, :], V[1, :]), agent_vel) : agent_color)
    scatter!(ax, all_points; markersize=agent_ms, color=agent_colors,
        colormap=heading_colormap, colorrange=(-pi, pi),
        label=territory_homes === nothing ? nothing : "agents")

    if color_by_heading
        Colorbar(fig[2, 1]; limits=(-pi, pi), colormap=heading_colormap,
            vertical=false, ticks=([-pi, -pi/2, 0, pi/2, pi],
            ["−π", "−π/2", "0", "π/2", "π"]), label="heading angle")
    end

    if show_agent_dirs
        dir_bases = lift(agent_pos) do P
            idx = 1:dir_stride:size(P, 2)
            Point2f.(P[1, idx], P[2, idx])
        end

        dir_vecs = lift(agent_vel) do V
            idx = 1:dir_stride:size(V, 2)
            vx = @view V[1, idx]; vy = @view V[2, idx]
            nrm = sqrt.(vx .^ 2 .+ vy .^ 2) .+ 1e-12
            Vec2f.(dir_len_val .* vx ./ nrm, dir_len_val .* vy ./ nrm)
        end

        dir_colors = color_by_heading ? lift(agent_vel) do V
            idx = 1:dir_stride:size(V, 2)
            atan.(V[2, idx], V[1, idx])
        end : (has_agent_states ? lift(agent_colors) do colors
            colors[1:dir_stride:end]
        end : agent_color)
        arrows2d!(ax, dir_bases, dir_vecs; color=dir_colors,
            colormap=heading_colormap, colorrange=(-pi, pi),
            tiplength=5, shaftwidth=dir_linewidth)
    end

    if has_pred
        pred_point = Point2f(pred_hist[1][1], pred_hist[1][2])
        pred_obs  = Observable(pred_point)
        trail_obs = Observable(Point2f[pred_point])

        if show_influence_radius && rp > 0
            ring_obs = Observable(circle_points(pred_point, rp; n=ring_pts))
            lines!(ax, ring_obs; color=(predator_color, 0.5), linewidth=1.5)
        else
            ring_obs = nothing
        end

        lines!(ax, trail_obs; linewidth=2, color=(predator_color, 0.6))
        scatter!(ax, pred_obs; markersize=agent_ms * 2.2, color=predator_color,
            label=territory_homes === nothing ? "predator" : "prey")
    end

    territory_homes !== nothing && axislegend(ax; position=:rt)

    frames = collect(1:every:T)
    prog = Progress(length(frames); desc="Rendering mp4", dt=0.5)

    record(fig, savepath, 1:length(frames)) do idx
        k = frames[idx]
        agent_pos[] = S[k].P
        has_agent_states && (agent_colors[] =
            [s == 0 ? :dodgerblue : :firebrick for s in agent_state_hist[k]])
        (show_agent_dirs || color_by_heading) && (agent_vel[] = S[k].V)

        if show_agent_trails
            push!(agent_trail_hist, S[k].P)
            if length(agent_trail_hist) > agent_trail_len
                popfirst!(agent_trail_hist)
            end
            agent_trail_obs[] = _agent_trail_segments(agent_trail_hist)
        end

        if has_pred
            pk = Point2f(pred_hist[k][1], pred_hist[k][2])
            pred_obs[] = pk

            old = trail_obs[]
            push!(old, pk)
            if length(old) > trail_len
                old = old[(end - trail_len + 1):end]
            end
            trail_obs[] = old

            ring_obs !== nothing && (ring_obs[] = circle_points(pk, rp; n=ring_pts))
        end

        next!(prog)
    end

    println("Saved animation to $savepath")
    return savepath
end

"""
    animate_swarm_comparison(states_a, states_b, half_extent; pred_hist=nothing, rp=0.0,
                              labels=("undriven","driven"), center=(0.0,0.0), savepath="comparison.mp4")

Two-panel side-by-side `.mp4` comparing two logged state trajectories
(each a vector of `raw_state`-shaped NamedTuples, as in `animate_from_states`)
frame by frame on a shared, synchronised playhead -- e.g. an undriven swarm
next to a predator-driven one, to see directly what the coupling does.
`states_a`/`states_b` must have the same length `T`.

If `pred_hist` (length `T`) is given, the predator is drawn in *both*
panels at the same position each frame -- useful for comparing a driven
swarm that responds to it against an undriven one shown the identical
trajectory but structurally unable to feel it (`Kp=0`), or for comparing
two different driven parameter regimes against the same drive. Each panel
also draws its own agents' trailing paths (`show_agent_trails=true` by
default), same as `animate_from_states`.
`half_extent` and `center` set an identical `[center, center+2half_extent]`
window on both panels (see `animate_from_states`'s `center` docs for the
`LymburnReservoir` convention).
"""
function animate_swarm_comparison(
    states_a, states_b, half_extent::Real;
    pred_hist = nothing,
    labels::Tuple{String,String} = ("undriven", "driven"),
    center::Tuple{<:Real,<:Real} = (0.0, 0.0),
    every::Int = 2,
    agent_ms::Float64 = 5.0,
    savepath::String = "swarm_comparison.mp4",
    show_influence_radius::Bool = true,
    rp::Float64 = 0.0,
    ring_pts::Int = 200,
    trail_len::Int = 60,
    show_agent_trails::Bool = true,
    agent_trail_len::Int = 25,
    agent_trail_color = (:grey40, 0.3),
)
    T = length(states_a)
    @assert length(states_b) == T "states_a and states_b must have the same length"
    has_pred = pred_hist !== nothing
    has_pred && @assert length(pred_hist) == T "pred_hist must have the same length as states_a/states_b"

    fig = Figure(size = (1400, 750))
    agent_obs = Observable[]
    agent_trail_hists = Vector{Matrix{Float64}}[]
    agent_trail_obs_list = Any[]
    pred_obs_list = Any[]
    trail_obs_list = Any[]
    ring_obs_list = Any[]
    axes_list = Axis[]

    for (col, (S, label)) in enumerate(zip((states_a, states_b), labels))
        ax = Axis(fig[1, col], xlabel = "x", ylabel = "y", title = label, aspect = DataAspect())
        xlims!(ax, center[1], center[1] + 2half_extent)
        ylims!(ax, center[2], center[2] + 2half_extent)
        push!(axes_list, ax)

        if show_agent_trails
            hist = Matrix{Float64}[S[1].P]
            push!(agent_trail_hists, hist)
            tobs = Observable(_agent_trail_segments(hist))
            push!(agent_trail_obs_list, tobs)
            lines!(ax, tobs; color = agent_trail_color, linewidth = 1.0)
        end

        pos0 = Observable(S[1].P)
        push!(agent_obs, pos0)
        all_points = lift(pos0) do P
            Point2f.(P[1, :], P[2, :])
        end
        scatter!(ax, all_points; markersize = agent_ms, color = :black)

        if has_pred
            pred_point = Point2f(pred_hist[1][1], pred_hist[1][2])
            pobs = Observable(pred_point)
            tobs = Observable(Point2f[pred_point])
            push!(pred_obs_list, pobs)
            push!(trail_obs_list, tobs)

            if show_influence_radius && rp > 0
                robs = Observable(circle_points(pred_point, rp; n = ring_pts))
                lines!(ax, robs; color = (:red, 0.5), linewidth = 1.5)
                push!(ring_obs_list, robs)
            else
                push!(ring_obs_list, nothing)
            end
            lines!(ax, tobs; linewidth = 2, color = (:red, 0.6))
            scatter!(ax, pobs; markersize = agent_ms * 2.2, color = :red)
        end
    end

    linkaxes!(axes_list...)

    frames = collect(1:every:T)
    prog = Progress(length(frames); desc = "Rendering comparison mp4", dt = 0.5)

    record(fig, savepath, 1:length(frames)) do idx
        k = frames[idx]
        agent_obs[1][] = states_a[k].P
        agent_obs[2][] = states_b[k].P

        if show_agent_trails
            for (j, S) in enumerate((states_a, states_b))
                hist = agent_trail_hists[j]
                push!(hist, S[k].P)
                if length(hist) > agent_trail_len
                    popfirst!(hist)
                end
                agent_trail_obs_list[j][] = _agent_trail_segments(hist)
            end
        end

        if has_pred
            pk = Point2f(pred_hist[k][1], pred_hist[k][2])
            for j in 1:2
                pred_obs_list[j][] = pk
                old = trail_obs_list[j][]
                push!(old, pk)
                if length(old) > trail_len
                    old = old[(end - trail_len + 1):end]
                end
                trail_obs_list[j][] = old
                ring_obs_list[j] !== nothing && (ring_obs_list[j][] = circle_points(pk, rp; n = ring_pts))
            end
        end

        next!(prog)
    end

    println("Saved comparison animation to $savepath")
    return savepath
end

"""
    plot_kernels(K::KernelLayer; res=nothing, L=nothing, show_agents=true, domain=:auto)

Visualise the Gaussian observation kernels (centres + receptive-field radii)
built by `build_observation_layer_spatial_gaussian!` (or the
coverage/k-means placement rule), optionally overlaid on the current Couzin
or Lymburn swarm state in `res`. For a non-periodic Lymburn reservoir,
`domain=:auto` tightly frames the kernels and current agents; use
`domain=:model` to show the full nominal `plot_extent`. A periodic Couzin
reservoir always shows its full physical domain.
"""
function plot_kernels(K::KernelLayer;
    res = nothing,
    importance = nothing,
    L::Union{Nothing,Float64} = nothing,
    show_agents::Bool = true,
    kernel_rmult::Float64 = 1.0,
    n_circle::Int = 80,
    fig_size::Tuple{Int,Int} = (400, 400),
    domain::Symbol = :auto,
    padding_fraction::Float64 = 0.08,
)
    domain in (:auto, :model) || error("domain must be :auto or :model")
    padding_fraction >= 0 || error("padding_fraction must be non-negative")

    C, inv2w = K.C, K.inv2w
    M = size(C, 2)
    σ = kernel_sigma.(inv2w)
    σmin, σmax = extrema(σ)

    if isnothing(res)
        Lval = something(L, error("Provide L=... if res is not given"))
        xbounds = (0.0, Lval)
        ybounds = (0.0, Lval)
    elseif nameof(typeof(res)) == :CouzinReservoir
        Lval = res.P.L
        xbounds = (0.0, Lval)
        ybounds = (0.0, Lval)
    elseif nameof(typeof(res)) == :LymburnReservoir && domain == :model
        h = res.P.plot_extent
        xbounds = (res.P.xh.x - h, res.P.xh.x + h)
        ybounds = (res.P.xh.y - h, res.P.xh.y + h)
    elseif nameof(typeof(res)) == :LymburnReservoir
        radii = kernel_rmult .* σ
        xs = vcat(C[1, :] .- radii, C[1, :] .+ radii)
        ys = vcat(C[2, :] .- radii, C[2, :] .+ radii)
        if show_agents
            append!(xs, getfield.(res.state.pos, :x))
            append!(ys, getfield.(res.state.pos, :y))
        end

        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)
        span = max(xmax - xmin, ymax - ymin, eps(Float64))
        padding = padding_fraction * span
        halfwidth = 0.5span + padding
        xmid, ymid = 0.5(xmin + xmax), 0.5(ymin + ymax)
        xbounds = (xmid - halfwidth, xmid + halfwidth)
        ybounds = (ymid - halfwidth, ymid + halfwidth)
    else
        error("plot_kernels does not know the plotting domain for $(typeof(res)); provide a supported swarm reservoir.")
    end

    fig = Figure(size=fig_size)
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Observation kernels over swarm state")
    xlims!(ax, xbounds...); ylims!(ax, ybounds...)
    ax.aspect = DataAspect()

    if show_agents && !isnothing(res)
        pts = Point2f.(getfield.(res.state.pos, :x), getfield.(res.state.pos, :y))
        scatter!(ax, pts; markersize=5, color=(:black, 0.18))
    end

    centres = Point2f.(C[1, :], C[2, :])
    scatter!(ax, centres; markersize=4, color=:dodgerblue)

    for m in 1:M
        c = centres[m]
        r = kernel_rmult * σ[m]
        if importance === nothing
            t = (σ[m] - σmin) / (σmax - σmin + 1e-12)
            α = 0.25 * (1 - t) + 0.05
            col = RGBf(t, 0.2, 1 - t)
        else
            α = 0.05 + 0.95 * importance[m]
            col = (:red, α)
        end
        lines!(ax, circle_points(c, r; n=n_circle), color=(col, α), linewidth=1.2)
    end

    Colorbar(fig[1, 2], limits=(σmin, σmax), colormap=:coolwarm, label="kernel σ (spatial sensitivity)")
    return fig
end

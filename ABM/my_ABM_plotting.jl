using CairoMakie
using FFMPEG
using LinearAlgebra
using Statistics

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
    domain_axis_box(P) -> (xlo, xhi, ylo, yhi)

Fixed (non-comoving) display box for a model's params `P`, used by
`animate_simulation_2d`/`3d`. Two conventions exist across models and are
not interchangeable: Couzin/Helbing use `P.L` and Lymburn/Mizzi use the
home-centred half-range `P.plot_extent`.
"""
function domain_axis_box(P)
    if hasproperty(P, :L)
        L = getproperty(P, :L)
        return (0.0, L, 0.0, L)
    elseif hasproperty(P, :plot_extent)
        e = getproperty(P, :plot_extent)
        cx, cy = hasproperty(P, :xh) ? (P.xh.x, P.xh.y) : (0.0, 0.0)
        return (cx - e, cx + e, cy - e, cy + e)
    else
        throw(ArgumentError(
            "animate_simulation expects params to have field L or plot_extent"))
    end
end

# positions_points
# positions_points_3d

# arrows_from

# AnimationConfig

# circle_points
# annulus_points
# wrap_pi
# sector_points
# annulus_sector_points
# sphere_wireframe_curves

# _setup_order_axis!
# _setup_order_traces!
# _run_animation!

# plot_order_parameters

# animate_simulation_2d
# animate_simulation_3d

# Zone overlays are Couzin-specific and are enabled only when requested.

positions_points(pos::Vector{SVector2}) = Point2f[Point2f(p.x, p.y) for p in pos]

positions_points_3d(pos::Vector{SVector3}) = Point3f[
    Point3f(p.x, p.y, p.z) for p in pos
]

function _fill_positions!(buf::Vector{Point2f}, pos::Vector{SVector2})
    @inbounds for i in eachindex(pos)
        buf[i] = Point2f(pos[i].x, pos[i].y)
    end
    return buf
end

function _fill_positions_3d!(buf::Vector{Point3f}, pos::Vector{SVector3})
    @inbounds for i in eachindex(pos)
        buf[i] = Point3f(pos[i].x, pos[i].y, pos[i].z)
    end
    return buf
end


function arrows_from(pos::Vector{SVector2}, vel::Vector{SVector2}; shaft_len=0.9)
    ap = Point2f[]
    ad = Vec2f[]
    sizehint!(ap, length(pos))
    sizehint!(ad, length(pos))
    @inbounds for i in eachindex(pos)
        d = unit(vel[i])
        push!(ap, Point2f(pos[i].x, pos[i].y))
        push!(ad, Vec2f(shaft_len * d.x, shaft_len * d.y))
    end
    return ap, ad
end

function _fill_arrows!(ap::Vector{Point2f}, ad::Vector{Vec2f},
                       pos::Vector{SVector2}, vel::Vector{SVector2};
                       shaft_len::Float64 = 0.9)
    @inbounds for i in eachindex(pos)
        d = unit(vel[i])
        ap[i] = Point2f(pos[i].x, pos[i].y)
        ad[i] = Vec2f(shaft_len * d.x, shaft_len * d.y)
    end
    return ap, ad
end

function _fill_arrows_3d!(ap::Vector{Point3f}, ad::Vector{Vec3f},
                          pos::Vector{SVector3}, vel::Vector{SVector3};
                          shaft_len::Float64 = 1.0)
    @inbounds for i in eachindex(pos)
        d = unit(vel[i])
        ap[i] = Point3f(pos[i].x, pos[i].y, pos[i].z)
        ad[i] = Vec3f(shaft_len * d.x, shaft_len * d.y, shaft_len * d.z)
    end
    return ap, ad
end


Base.@kwdef struct AnimationConfig
    fps::Int           = 60
    stride::Int        = 1       # render every Nth stored frame (1 = all frames)
    shaft_len::Float64 = 1.0
    agent_ms::Float64  = 6.0
    save_mp4::Bool     = false
    filename::String   = "abm.mp4"
    live::Bool         = false
    fig_size::Tuple{Int,Int} = (1300, 600)

    show_zones::Bool = false

    zone_alpha::Float64   = 0.12
    zone_stroke::Float64  = 1.0
    zone_npoints::Int     = 120

    highlight_focal::Bool = true
    focal_scale::Float64  = 1.6

    # false (default): grow traces incrementally each frame — faster for CairoMakie
    #                   because it draws fewer points on early frames
    # true:  draw all traces upfront + a moving cursor — better for GPU renderers
    #                   (GLMakie/WGLMakie) where per-frame upload cost dominates
    static_order_traces::Bool = false

    # false (default): fixed [0,L]^dim axis limits — correct for periodic/bounded runs.
    # true: animate_simulation_2d/3d re-centre every frame on the instantaneous centre of
    #       mass (axis limits become symmetric about the origin, sized from the initial
    #       frame's spread). Needed for unbounded (periodic=false) runs — e.g. the
    #       Couzin (2002) publication reproduction — where the group drifts outside a
    #       fixed box over a long run.
    comoving::Bool = false
end

function update_AnimationConfig(
    cfg :: AnimationConfig;
    fps                 = cfg.fps,
    stride              = cfg.stride,
    shaft_len           = cfg.shaft_len,
    agent_ms            = cfg.agent_ms,
    save_mp4            = cfg.save_mp4,
    filename            = cfg.filename,
    live                = cfg.live,
    fig_size            = cfg.fig_size,
    show_zones          = cfg.show_zones,
    zone_alpha          = cfg.zone_alpha,
    zone_stroke         = cfg.zone_stroke,
    zone_npoints        = cfg.zone_npoints,
    highlight_focal     = cfg.highlight_focal,
    focal_scale         = cfg.focal_scale,
    static_order_traces = cfg.static_order_traces,
    comoving            = cfg.comoving,
)
    return AnimationConfig(
        fps                 = fps,
        stride              = stride,
        shaft_len           = shaft_len,
        agent_ms            = agent_ms,
        save_mp4            = save_mp4,
        filename            = filename,
        live                = live,
        fig_size            = fig_size,
        show_zones          = show_zones,
        zone_alpha          = zone_alpha,
        zone_stroke         = zone_stroke,
        zone_npoints        = zone_npoints,
        highlight_focal     = highlight_focal,
        focal_scale         = focal_scale,
        static_order_traces = static_order_traces,
        comoving            = comoving,
    )
end

function circle_points(c::Point2f, r::Float64; n::Int = 120)
    θ = range(0, 2π; length=n+1)[1:end-1]
    return Point2f[(c[1] + r*cos(t), c[2] + r*sin(t)) for t in θ]
end

"""
    _agent_trail_segments(hist) -> Vector{Point2f}

Builds one NaN-separated polyline per agent from `hist` (a `Vector` of 2×N
position matrices, oldest first) so all N agent trails can be drawn with a
single `lines!` call instead of N separate ones. Used by `plot_swarm_frame!`
below and by `SWARM_RC/my_swarmRC_plotting.jl`'s animation functions.
"""
function _agent_trail_segments(hist::Vector{<:AbstractMatrix})
    isempty(hist) && return Point2f[]
    N = size(hist[1], 2)
    nframes = length(hist)
    segs = Vector{Point2f}(undef, N * (nframes + 1))
    idx = 1
    @inbounds for i in 1:N
        for f in 1:nframes
            P = hist[f]
            segs[idx] = Point2f(P[1, i], P[2, i])
            idx += 1
        end
        segs[idx] = Point2f(NaN, NaN)
        idx += 1
    end
    return segs
end

"""
    plot_swarm_frame!(ax, pos_hist, vel_hist, k; trail_len=25, agent_ms=5.0,
                       color_by_heading=true, trail_color=(:grey40,0.3), colormap=:hsv,
                       periodic_L=nothing)

Draw a single static "snapshot" (frame `k`) from a `pos_hist`/`vel_hist`
time series (as returned by `simulate`/`simulate_Lymburn_2d`, i.e.
`Vector{Vector{SVector2}}`), with each agent's recent trailing path (the
last `trail_len` frames, fewer near the start of the run) drawn behind its
current position -- makes swirling/milling motion visible in a single
frame instead of only in a full animation. Colours agents by heading
(`atan(vy,vx)`) unless `color_by_heading=false`.

Pass `periodic_L` (the domain side length, e.g. `P.L` for a `Couzin`
model) for a periodic `[0,L]x[0,L]` domain -- this breaks the trail
wherever an agent wraps around the boundary between frames (jump >
`L/2` in either coordinate), instead of drawing a spurious line straight
across the box. Leave `periodic_L=nothing` (default) for non-periodic
models like Lymburn.
"""
function plot_swarm_frame!(ax, pos_hist, vel_hist, k::Int;
    trail_len::Int = 25, agent_ms::Float64 = 5.0,
    color_by_heading::Bool = true, trail_color = (:grey40, 0.3),
    colormap = :hsv, agent_color=:black,
    periodic_L::Union{Nothing,Real} = nothing,
)
    k0 = max(1, k - trail_len + 1)
    window = pos_hist[k0:k]
    N = length(pos_hist[k])
    nframes = length(window)

    segs = Point2f[]
    sizehint!(segs, N * (nframes + 1))
    @inbounds for i in 1:N
        prev = nothing
        for f in 1:nframes
            p = window[f][i]
            cur = Point2f(p.x, p.y)
            if periodic_L !== nothing && prev !== nothing &&
               (abs(cur[1] - prev[1]) > periodic_L / 2 || abs(cur[2] - prev[2]) > periodic_L / 2)
                push!(segs, Point2f(NaN, NaN))
            end
            push!(segs, cur)
            prev = cur
        end
        push!(segs, Point2f(NaN, NaN))
    end
    lines!(ax, segs; color=trail_color, linewidth=1.0)

    pos = pos_hist[k]
    xs = [p.x for p in pos]
    ys = [p.y for p in pos]
    if color_by_heading
        vel = vel_hist[k]
        headings = [atan(v.y, v.x) for v in vel]
        scatter!(ax, xs, ys; color=headings, colormap=colormap, colorrange=(-pi, pi), markersize=agent_ms)
    else
        scatter!(ax, xs, ys; color=agent_color, markersize=agent_ms)
    end
    return ax
end

function annulus_points(c::Point2f, r_in::Float64, r_out::Float64; n::Int = 120)
    outer = circle_points(c, r_out; n=n)
    inner = reverse(circle_points(c, r_in; n=n))
    return vcat(outer, inner)
end

@inline function wrap_pi(θ::Float64)
    ϕ = mod(θ + π, 2π) - π
    return ϕ
end

function sector_points(c::Point2f, r::Float64, θ1::Float64, θ2::Float64; n::Int=120)
    Δ = wrap_pi(θ2 - θ1)
    Δ ≤ 0 && (Δ += 2π)
    θ = range(θ1, θ1 + Δ; length=n)
    pts = Point2f[Point2f(c[1], c[2])]
    append!(pts, Point2f[(c[1] + r*cos(t), c[2] + r*sin(t)) for t in θ])
    push!(pts, Point2f(c[1], c[2]))
    return pts
end



function annulus_sector_points(c::Point2f, r_in::Float64, r_out::Float64,
                               θ1::Float64, θ2::Float64; n::Int=180)
    Δ = wrap_pi(θ2 - θ1)
    Δ ≤ 0 && (Δ += 2π)
    θ = range(θ1, θ1 + Δ; length=n)

    outer = Point2f[(c[1] + r_out*cos(t), c[2] + r_out*sin(t)) for t in θ]
    inner = Point2f[(c[1] + r_in*cos(t),  c[2] + r_in*sin(t)) for t in reverse(θ)]
    return vcat(outer, inner)
end



function sphere_wireframe_curves(c::Point3f, r::Float64; nθ::Int = 60, nϕ::Int = 12, nλ::Int = 12)
    θs = range(0, 2π; length=nθ+1)

    curves = Vector{Vector{Point3f}}()

    # latitude circles
    ϕs = range(0, π; length=nϕ+2)[2:end-1]   # exclude poles
    for ϕ in ϕs
        curve = Point3f[]
        for θ in θs
            x = c[1] + r * sin(ϕ) * cos(θ)
            y = c[2] + r * sin(ϕ) * sin(θ)
            z = c[3] + r * cos(ϕ)
            push!(curve, Point3f(x, y, z))
        end
        push!(curves, curve)
    end

    # longitude circles
    λs = range(0, 2π; length=nλ+1)[1:end-1]
    ϕcurve = range(0, π; length=nθ)
    for λ in λs
        curve = Point3f[]
        for ϕ in ϕcurve
            x = c[1] + r * sin(ϕ) * cos(λ)
            y = c[2] + r * sin(ϕ) * sin(λ)
            z = c[3] + r * cos(ϕ)
            push!(curve, Point3f(x, y, z))
        end
        push!(curves, curve)
    end

    return curves
end


function _setup_order_axis!(gl, simcfg)
    ax_ts = Axis(gl[1, 2], title="Order parameters", xlabel=L"$t$", ylabel="value")
    ylims!(ax_ts, -1.05, 1.05)
    xlims!(ax_ts, 0, simcfg.steps * simcfg.dt)
    return ax_ts
end

function _setup_order_traces!(ax_ts, out; static::Bool = true)
    k_obs = Observable(0)

    have_order = length(out.dilation) > 0
    have_mabs  = hasproperty(out, :abs_angular_momentum) && !isempty(out.abs_angular_momentum)

    if have_order
        if static
            # Draw complete traces upfront + moving cursor.
            # Draws T points every frame — better for GPU renderers (GLMakie/WGLMakie).
            lines!(ax_ts, out.t, out.dilation,     label = "dilation")
            lines!(ax_ts, out.t, out.rotation,     label = "rotation")
            lines!(ax_ts, out.t, out.polarisation, label = "polarisation")
            if have_mabs
                lines!(ax_ts, out.t, out.abs_angular_momentum, label = "abs angular momentum")
            end
            cursor_t = lift(k -> k > 0 ? Float64(out.t[k]) : Float64(out.t[1]), k_obs)
            vlines!(ax_ts, cursor_t; color = (:black, 0.45), linewidth = 1, linestyle = :dash)
        else
            # Grow traces incrementally (default for CairoMakie).
            # Draws k points on frame k, so total draw work is O(T²/2).
            dil_pts = lift(k -> Point2f.(out.t[1:k], out.dilation[1:k]),     k_obs)
            rot_pts = lift(k -> Point2f.(out.t[1:k], out.rotation[1:k]),     k_obs)
            pol_pts = lift(k -> Point2f.(out.t[1:k], out.polarisation[1:k]), k_obs)
            lines!(ax_ts, dil_pts, label = "dilation")
            lines!(ax_ts, rot_pts, label = "rotation")
            lines!(ax_ts, pol_pts, label = "polarisation")
            if have_mabs
                mabs_pts = lift(k -> Point2f.(out.t[1:k], out.abs_angular_momentum[1:k]), k_obs)
                lines!(ax_ts, mabs_pts, label = "abs angular momentum")
            end
        end
        axislegend(ax_ts; position = :rt)
    end

    return (k_obs=k_obs, have_order=have_order, have_mabs=have_mabs)
end

function _run_animation!(fig, update_frame!::Function, out, animcfg::AnimationConfig)
    update_frame!(1)
    T      = length(out.pos_hist)
    frames = collect(1:animcfg.stride:T)

    if animcfg.save_mp4
        n    = length(frames)
        prog = Progress(n; desc = "Rendering mp4", dt = 0.5)
        record(fig, animcfg.filename, 1:n; framerate = animcfg.fps) do idx
            update_frame!(frames[idx])
            next!(prog)
        end
        println("Saved movie: $(animcfg.filename)")
    elseif animcfg.live
        for k in frames[2:end]
            update_frame!(k)
            sleep(1 / animcfg.fps)
        end
    end

    return fig
end


function plot_order_parameters(out; title::String="Order parameters")
    @assert has_order(out) "Order parameter time series not available. Run simulation with collect_order=true."

    fig = Figure(size=(900, 400))
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = L"$t$",
        ylabel = "value",
    )

    xlims!(ax, minimum(out.t), maximum(out.t))
    ylims!(ax, -1.05, 1.05)

    lines!(ax, out.t, out.dilation, label = "dilation")
    lines!(ax, out.t, out.rotation, label = "rotation")
    lines!(ax, out.t, out.polarisation, label = "polarisation")

    if hasproperty(out, :abs_angular_momentum) && !isempty(out.abs_angular_momentum)
        lines!(ax, out.t, out.abs_angular_momentum, label = "abs angular momentum")
    end

    axislegend(ax; position = :rt)

    return fig
end


# ============================================================
# 2D/3D animation
# ============================================================

function animate_simulation_2d(out, animcfg::AnimationConfig; title::String="ABM simulation")
    @assert has_history(out) "Animation requires stored position/velocity history. Run simulation with collect_history=true."

    simcfg = out.simcfg
    P      = out.params
    focal  = out.focal

    xlo, xhi, ylo, yhi = domain_axis_box(P)

    pos0 = out.pos_hist[1]
    vel0 = out.vel_hist[1]

    # comoving: re-centre on the instantaneous centre of mass every frame, instead of a
    # fixed [0,L]^2 box. Needed for unbounded (periodic=false) runs where the group drifts
    # over a long simulation -- see animate_simulation_3d for the same trick. off0/offk
    # below are zero vectors (no-ops) when comoving=false, so the non-comoving path is
    # numerically identical to the original fixed-box behaviour.
    off0 = animcfg.comoving ? sum(pos0) / length(pos0) : (pos0[1] - pos0[1])

    fig = Figure(size=animcfg.fig_size)
    gl  = fig[1, 1] = GridLayout()

    ax = Axis(gl[1, 1], title=title, xlabel=L"$x$", ylabel=L"$y$")

    if animcfg.comoving
        R0 = maximum(norm(p - off0) for p in pos0)
        R0 = max(R0 * 1.6, 1.0)   # padding + guard against a degenerate (R0≈0) initial frame
        xlims!(ax, -R0, R0)
        ylims!(ax, -R0, R0)
    else
        xlims!(ax, xlo, xhi)
        ylims!(ax, ylo, yhi)
    end
    ax.aspect = DataAspect()

    ax_ts = _setup_order_axis!(gl, simcfg)

    colsize!(gl, 1, Relative(0.55))
    colsize!(gl, 2, Relative(0.45))
    colgap!(gl, 12)

    pos0_c = pos0 .- Ref(off0)

    _pos_buf = positions_points(pos0_c)
    _ap_buf, _ad_buf = arrows_from(pos0_c, vel0; shaft_len=animcfg.shaft_len)
    pos_obs   = Observable(_pos_buf)
    arrow_pos = Observable(_ap_buf)
    arrow_dir = Observable(_ad_buf)

    scatter!(ax, pos_obs; markersize=animcfg.agent_ms)
    arrows2d!(ax, arrow_pos, arrow_dir; tiplength=5, shaftwidth=1.5)

    has_pred = hasproperty(out, :pred_hist) && !isempty(getproperty(out, :pred_hist))

    pred_obs  = Observable(Point2f(0, 0))
    trail_obs = Observable(Point2f[])
    pred_rp   = hasproperty(out, :pred_rp) ? getproperty(out, :pred_rp) : 0.0

    if has_pred
        p0 = getproperty(out, :pred_hist)[1]
        pred_obs[]  = Point2f(p0.x - off0.x, p0.y - off0.y)
        trail_obs[] = Point2f[pred_obs[]]

        scatter!(ax, pred_obs; markersize=14, color=:red)
        lines!(ax, trail_obs; linewidth=2, color=:red)

        if pred_rp > 0
            θ = range(0, 2π; length=animcfg.zone_npoints)
            pred_ring = lift(pred_obs) do p
                Point2f.(p[1] .+ pred_rp .* cos.(θ), p[2] .+ pred_rp .* sin.(θ))
            end
            lines!(ax, pred_ring; linewidth=2, color=:red)
        end
    end

    zone_state = nothing
    if animcfg.show_zones && hasproperty(P, :Zr) && hasproperty(P, :Zo) && hasproperty(P, :Za)
        c0 = Point2f(pos0_c[focal].x, pos0_c[focal].y)

        Zr = getproperty(P, :Zr)
        Zo = getproperty(P, :Zo)
        Za = getproperty(P, :Za)

        β = hasproperty(P, :blind_half_angle) ? getproperty(P, :blind_half_angle) : 0.0
        v0 = vel0[focal]
        ϕ0 = angle_of(unit(v0))

        θ1 = ϕ0 - (π - β)
        θ2 = ϕ0 + (π - β)

        npts = animcfg.zone_npoints
        α  = animcfg.zone_alpha
        sw = animcfg.zone_stroke

        zr_poly = Observable(circle_points(c0, Zr; n=npts))
        zo_poly = β > 0 ?
            Observable(annulus_sector_points(c0, Zr, Zo, θ1, θ2; n=max(60, npts))) :
            Observable(annulus_points(c0, Zr, Zo; n=npts))
        za_poly = β > 0 ?
            Observable(annulus_sector_points(c0, Zo, Za, θ1, θ2; n=max(60, npts))) :
            Observable(annulus_points(c0, Zo, Za; n=npts))

        poly!(ax, zr_poly; color = (:red, α),    strokecolor = (:red, 0.75),    strokewidth = sw)
        poly!(ax, zo_poly; color = (:blue, α),   strokecolor = (:blue, 0.65),   strokewidth = sw)
        poly!(ax, za_poly; color = (:orange, α), strokecolor = (:orange, 0.65), strokewidth = sw)

        focal_obs = nothing
        if animcfg.highlight_focal
            focal_obs = Observable(Point2f[Point2f(c0[1], c0[2])])
            scatter!(ax, focal_obs; markersize = animcfg.agent_ms * animcfg.focal_scale)
        end

        zone_state = (zr=zr_poly, zo=zo_poly, za=za_poly, focal_obs=focal_obs, β=β)
    end

    order_state = _setup_order_traces!(ax_ts, out; static = animcfg.static_order_traces)

    function update_frame!(k::Int)
        posk = out.pos_hist[k]
        velk = out.vel_hist[k]

        offk = animcfg.comoving ? sum(posk) / length(posk) : (posk[1] - posk[1])
        posk_c = posk .- Ref(offk)

        _fill_positions!(_pos_buf, posk_c)
        pos_obs[] = _pos_buf
        _fill_arrows!(_ap_buf, _ad_buf, posk_c, velk; shaft_len = Float64(animcfg.shaft_len))
        arrow_pos[] = _ap_buf
        arrow_dir[] = _ad_buf

        if has_pred
            pk = getproperty(out, :pred_hist)[k]
            pred_obs[] = Point2f(pk.x - offk.x, pk.y - offk.y)

            old = trail_obs[]
            push!(old, pred_obs[])
            maxtrail = 150
            if length(old) > maxtrail
                old = old[(end - maxtrail + 1):end]
            end
            trail_obs[] = old
        end

        if zone_state !== nothing
            ck = Point2f(posk_c[focal].x, posk_c[focal].y)

            Zr = getproperty(P, :Zr)
            Zo = getproperty(P, :Zo)
            Za = getproperty(P, :Za)

            β = zone_state.β
            zone_state.zr[] = circle_points(ck, Zr; n=animcfg.zone_npoints)

            if β > 0
                vk = velk[focal]
                ϕk = angle_of(unit(vk))
                θ1 = ϕk - (π - β)
                θ2 = ϕk + (π - β)

                zone_state.zo[] = annulus_sector_points(ck, Zr, Zo, θ1, θ2; n=max(60, animcfg.zone_npoints))
                zone_state.za[] = annulus_sector_points(ck, Zo, Za, θ1, θ2; n=max(60, animcfg.zone_npoints))
            else
                zone_state.zo[] = annulus_points(ck, Zr, Zo; n=animcfg.zone_npoints)
                zone_state.za[] = annulus_points(ck, Zo, Za; n=animcfg.zone_npoints)
            end

            if zone_state.focal_obs !== nothing
                zone_state.focal_obs[] = Point2f[Point2f(ck[1], ck[2])]
            end
        end

        if order_state.have_order
            order_state.k_obs[] = k
        end

        return nothing
    end

    return _run_animation!(fig, update_frame!, out, animcfg)
end


function animate_simulation_3d(out, animcfg::AnimationConfig; title::String="ABM simulation (3D)")
    @assert has_history(out) "Animation requires stored position/velocity history. Run simulation with collect_history=true."

    simcfg = out.simcfg
    P      = out.params
    focal  = out.focal

    @assert hasproperty(P, :L) "animate_simulation_3d expects params to have field L"
    L = getproperty(P, :L)

    pos0 = out.pos_hist[1]
    vel0 = out.vel_hist[1]

    # comoving: re-centre on the instantaneous centre of mass every frame, instead of a
    # fixed [0,L]^3 box. Needed for unbounded (periodic=false) runs where the group drifts
    # over a long simulation -- see the co-moving snapshot trick used for Couzin (2002) in
    # the tutorial. off0/offk below are zero vectors (no-ops) when comoving=false, so the
    # non-comoving path is numerically identical to the original fixed-box behaviour.
    off0 = animcfg.comoving ? sum(pos0) / length(pos0) : (pos0[1] - pos0[1])

    fig = Figure(size=animcfg.fig_size)
    gl  = fig[1, 1] = GridLayout()

    ax = Axis3(
        gl[1, 1],
        title=title,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        aspect=(1, 1, 1),
    )

    if animcfg.comoving
        R0 = maximum(norm(p - off0) for p in pos0)
        R0 = max(R0 * 1.6, 1.0)   # padding + guard against a degenerate (R0≈0) initial frame
        xlims!(ax, -R0, R0)
        ylims!(ax, -R0, R0)
        zlims!(ax, -R0, R0)
    else
        xlims!(ax, 0, L)
        ylims!(ax, 0, L)
        zlims!(ax, 0, L)
    end

    ax_ts = _setup_order_axis!(gl, simcfg)

    colsize!(gl, 1, Relative(0.6))
    colsize!(gl, 2, Relative(0.4))
    colgap!(gl, 12)

    pos0_c = pos0 .- Ref(off0)

    _pos_buf_3d = positions_points_3d(pos0_c)
    _ap_buf_3d, _ad_buf_3d = arrows_from_3d(pos0_c, vel0; shaft_len=animcfg.shaft_len)
    pos_obs   = Observable(_pos_buf_3d)
    arrow_pos = Observable(_ap_buf_3d)
    arrow_dir = Observable(_ad_buf_3d)

    meshscatter!(ax, pos_obs;
        markersize = 0.4,
        color = (:dodgerblue, 0.85)
    )

    arrows3d!(ax, arrow_pos, arrow_dir;
        color       = (:black, 0.8),
        tiplength   = 0.30 * animcfg.shaft_len,
        tipradius   = 0.12 * animcfg.shaft_len,
        shaftradius = 0.04 * animcfg.shaft_len,
    )

    focal_obs = Observable(Point3f[])
    if animcfg.highlight_focal
        c0 = pos0_c[focal]
        focal_obs[] = Point3f[Point3f(c0.x, c0.y, c0.z)]
        scatter!(ax, focal_obs; markersize=animcfg.agent_ms * animcfg.focal_scale)
    end

        zone_state = nothing
    if animcfg.show_zones && hasproperty(P, :Zr) && hasproperty(P, :Zo) && hasproperty(P, :Za)
        Zr = getproperty(P, :Zr)
        Zo = getproperty(P, :Zo)
        Za = getproperty(P, :Za)

        c0 = Point3f(pos0_c[focal].x, pos0_c[focal].y, pos0_c[focal].z)

        nθ = max(12, animcfg.zone_npoints ÷ 3)
        nϕ = max(8,  animcfg.zone_npoints ÷ 6)
        nλ = max(8,  animcfg.zone_npoints ÷ 6)

        zr_curves0 = sphere_wireframe_curves(c0, Zr; nθ=nθ, nϕ=nϕ, nλ=nλ)
        zo_curves0 = sphere_wireframe_curves(c0, Zo; nθ=nθ, nϕ=nϕ, nλ=nλ)
        za_curves0 = sphere_wireframe_curves(c0, Za; nθ=nθ, nϕ=nϕ, nλ=nλ)

        zr_obs = [Observable(curve) for curve in zr_curves0]
        zo_obs = [Observable(curve) for curve in zo_curves0]
        za_obs = [Observable(curve) for curve in za_curves0]

        [lines!(ax, obs; color = (:red,    0.25), linewidth = animcfg.zone_stroke) for obs in zr_obs]
        [lines!(ax, obs; color = (:blue,   0.20), linewidth = animcfg.zone_stroke) for obs in zo_obs]
        [lines!(ax, obs; color = (:orange, 0.20), linewidth = animcfg.zone_stroke) for obs in za_obs]

        zone_state = (
            zr = zr_obs,
            zo = zo_obs,
            za = za_obs,
            nθ = nθ,
            nϕ = nϕ,
            nλ = nλ,
        )
    end

    order_state = _setup_order_traces!(ax_ts, out; static = animcfg.static_order_traces)

    function update_frame!(k::Int)
        posk = out.pos_hist[k]
        velk = out.vel_hist[k]

        offk = animcfg.comoving ? sum(posk) / length(posk) : (posk[1] - posk[1])
        posk_c = posk .- Ref(offk)

        _fill_positions_3d!(_pos_buf_3d, posk_c)
        pos_obs[] = _pos_buf_3d
        _fill_arrows_3d!(_ap_buf_3d, _ad_buf_3d, posk_c, velk; shaft_len = Float64(animcfg.shaft_len))
        arrow_pos[] = _ap_buf_3d
        arrow_dir[] = _ad_buf_3d

        if animcfg.highlight_focal
            ck = posk_c[focal]
            focal_obs[] = Point3f[Point3f(ck.x, ck.y, ck.z)]
        end

        if zone_state !== nothing
            ck = Point3f(posk_c[focal].x, posk_c[focal].y, posk_c[focal].z)

            Zr = getproperty(P, :Zr)
            Zo = getproperty(P, :Zo)
            Za = getproperty(P, :Za)

            zr_curves = sphere_wireframe_curves(ck, Zr; nθ=zone_state.nθ, nϕ=zone_state.nϕ, nλ=zone_state.nλ)
            zo_curves = sphere_wireframe_curves(ck, Zo; nθ=zone_state.nθ, nϕ=zone_state.nϕ, nλ=zone_state.nλ)
            za_curves = sphere_wireframe_curves(ck, Za; nθ=zone_state.nθ, nϕ=zone_state.nϕ, nλ=zone_state.nλ)

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

        if order_state.have_order
            order_state.k_obs[] = k
        end

        return nothing
    end

    return _run_animation!(fig, update_frame!, out, animcfg)
end


function animate_model_output(
    out;
    model::Symbol,
    dim::Int = 2,
    save_mp4::Bool = false,
    filename::String = "abm.mp4",
    show_zones::Bool = model in (:Couzin, :couzin),
    animcfg::Union{Nothing, AnimationConfig} = nothing,
)
    cfg_base = animcfg !== nothing ? animcfg : AnimationConfig(
        fps        = 30,
        shaft_len  = 0.8,
        agent_ms   = 6.0,
        save_mp4   = save_mp4,
        filename   = filename,
        live       = false,
    )
    # show_zones has a smart default (true for Couzin) — always apply it so that
    # passing animcfg without explicitly setting show_zones doesn't silently hide zones.
    cfg = update_AnimationConfig(cfg_base; show_zones = show_zones)

    if dim == 2
        return animate_simulation_2d(out, cfg; title = "$(model) simulation")
    elseif dim == 3
        return animate_simulation_3d(out, cfg; title = "$(model) simulation")
    else
        error("dim must be 2 or 3.")
    end
end


# ============================================================
# Auto-labelling helpers  (plot_title / plot_filename)
# ============================================================
#
# Build a human-readable title and a filesystem-safe filename
# from a scenario + SimulationConfig, so no manual string editing
# is required between runs.
#
# Usage:
#   title    = plot_title(scenario, simcfg)
#   filename = plot_filename(scenario, simcfg)            # → "couzin_milling_N100_seed42.png"
#   filename = plot_filename(scenario, simcfg; tag="order", suffix="png")
#   filename = plot_filename(scenario, simcfg; suffix="mp4")

function _scale_tag(::Nothing)
    return ""
end
function _scale_tag(si::NamedTuple)
    si.type == :space    && return "_s$(si.s)"
    si.type == :time     && return "_f$(si.f)"
    si.type == :dynamics && return "_g$(si.g)"
    return "_$(si.type)"
end

function plot_title(scenario, simcfg)
    T = nameof(typeof(scenario))
    if T === :CouzinScenario
        P      = scenario.P
        preset = replace(titlecase(string(scenario.preset)), "_" => " ")
        nd     = scenario.nondim ? " (nondim)" : ""
        return "Couzin $(preset)$(nd)  N=$(P.N)  seed=$(simcfg.seed)"
    elseif T === :HelbingScenario
        P      = scenario.P
        preset = replace(titlecase(string(scenario.preset)), "_" => " ")
        return "Helbing $(preset)  N=$(P.N)  seed=$(simcfg.seed)"
    else
        return "Simulation  seed=$(simcfg.seed)"
    end
end

function plot_filename(scenario, simcfg;
                       tag::String    = "",
                       suffix::String = "png")
    tag_str = isempty(tag) ? "" : "_$(tag)"
    T = nameof(typeof(scenario))
    if T === :CouzinScenario
        P  = scenario.P
        nd = scenario.nondim ? "_nd" : ""
        sc = _scale_tag(scenario.scale_info)
        return "couzin_$(scenario.preset)_N$(P.N)$(nd)$(sc)$(tag_str)_seed$(simcfg.seed).$(suffix)"
    elseif T === :HelbingScenario
        P = scenario.P
        return "helbing_$(scenario.preset)_N$(P.N)$(tag_str)_seed$(simcfg.seed).$(suffix)"
    else
        return "sim$(tag_str)_seed$(simcfg.seed).$(suffix)"
    end
end


# ============================================================
# make_pretty_plot — universal figure-save wrapper
# ============================================================
#
# Calls any function that returns a Figure, then optionally saves
# the result to disk.  Works with all plot_* and animate_* functions.
#
# Examples:
#
#   fig = make_pretty_plot(plot_order_parameters, out;
#             title    = plot_title(scenario, simcfg),
#             save_fig = true, fig_dir = FIG_DIR,
#             filename = plot_filename(scenario, simcfg; tag = "order"))
#
#   fig = make_pretty_plot(animate_model_output, out;
#             model    = :Couzin,
#             animcfg  = AnimationConfig(fps=30, stride=2),
#             save_fig = true, fig_dir = FIG_DIR,
#             filename = plot_filename(scenario, simcfg; tag = "frame"))
#
#   fig = make_pretty_plot(plot_phase_heatmaps, xvals, yvals, Zs;
#             metrics  = metrics, xlabel = "Zo", ylabel = "Za",
#             save_fig = true, fig_dir = FIG_DIR,
#             filename = plot_filename(scenario, simcfg; tag = "phase"))
#
# Notes:
#   - px_per_unit controls output resolution (2 = 2× screen density,
#     good for publication; 1 = screen resolution).
#   - The directory fig_dir is created automatically if it does not exist.
#   - For animations, the saved PNG is the first frame (frame 1).
#     Pass save_mp4=true inside AnimationConfig to save the full video.

function make_pretty_plot(
    plot_fn  :: Function,
    args...;
    save_fig    :: Bool   = false,
    fig_dir     :: String = "",
    filename    :: String = "figure.png",
    px_per_unit :: Int    = 2,
    kwargs...,
)
    fig = plot_fn(args...; kwargs...)

    if save_fig
        isempty(fig_dir) && error(
            "make_pretty_plot: `fig_dir` must be provided when `save_fig = true`.\n" *
            "  Pass fig_dir = FIG_DIR  (or any path string)."
        )
        path = joinpath(ensure_dir(fig_dir), filename)
        save(path, fig; px_per_unit = px_per_unit)
        println("Saved: $path")
    end

    return fig
end

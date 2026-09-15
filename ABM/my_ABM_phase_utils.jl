# summary_to_matrix

# gaussian_kernel1d
# smooth_matrix_nan
# interpolate_matrix_bilinear
# process_phase_matrix


# ============================================================
# 4. Convert long summary dataframe -> matrix
# ------------------------------------------------------------
# rows = Δra, cols = Δro
# ============================================================
function summary_to_matrix(
    df::DataFrame,
    xcol::Symbol,
    ycol::Symbol,
    zcol::Symbol;
    xs_expected = nothing,
    ys_expected = nothing,
)
    xs = isnothing(xs_expected) ? sort(unique(df[!, xcol])) : collect(xs_expected)
    ys = isnothing(ys_expected) ? sort(unique(df[!, ycol])) : collect(ys_expected)

    Z = fill(NaN, length(ys), length(xs))  # rows = y, cols = x

    xmap = Dict(x => i for (i, x) in enumerate(xs))
    ymap = Dict(y => i for (i, y) in enumerate(ys))

    for row in eachrow(df)
        ix = xmap[row[xcol]]
        iy = ymap[row[ycol]]
        Z[iy, ix] = row[zcol]
    end

    return xs, ys, Z
end


# ============================================================
# Optional smoothing / interpolation for phase plots
# ============================================================

function gaussian_kernel1d(σ::Real; radius::Int = ceil(Int, 3σ))
    xs = collect(-radius:radius)
    k = exp.(-(xs .^ 2) ./ (2σ^2))
    return k ./ sum(k)
end

function smooth_matrix_nan(Z::AbstractMatrix; σ::Real = 1.0)
    k = gaussian_kernel1d(σ)
    r = length(k) ÷ 2

    Zf = Float64.(Z)
    out = similar(Zf)

    ny, nx = size(Zf)

    for i in 1:ny, j in 1:nx
        num = 0.0
        den = 0.0

        for di in -r:r, dj in -r:r
            ii = i + di
            jj = j + dj

            if 1 ≤ ii ≤ ny && 1 ≤ jj ≤ nx
                z = Zf[ii, jj]
                if !isnan(z)
                    w = k[di + r + 1] * k[dj + r + 1]
                    num += w * z
                    den += w
                end
            end
        end

        out[i, j] = den > 0 ? num / den : NaN
    end

    return out
end

function interpolate_matrix_bilinear(xs, ys, Z; factor::Int = 4)
    factor ≤ 1 && return xs, ys, Z

    xs_new = collect(range(first(xs), last(xs), length = (length(xs)-1)*factor + 1))
    ys_new = collect(range(first(ys), last(ys), length = (length(ys)-1)*factor + 1))

    Znew = fill(NaN, length(ys_new), length(xs_new))

    for (jj, x) in enumerate(xs_new)
        ix = searchsortedlast(xs, x)
        ix = clamp(ix, 1, length(xs)-1)

        x1, x2 = xs[ix], xs[ix+1]
        tx = (x - x1) / (x2 - x1)

        for (ii, y) in enumerate(ys_new)
            iy = searchsortedlast(ys, y)
            iy = clamp(iy, 1, length(ys)-1)

            y1, y2 = ys[iy], ys[iy+1]
            ty = (y - y1) / (y2 - y1)

            z11 = Z[iy, ix]
            z21 = Z[iy, ix+1]
            z12 = Z[iy+1, ix]
            z22 = Z[iy+1, ix+1]

            if any(isnan, (z11, z21, z12, z22))
                Znew[ii, jj] = NaN
            else
                Znew[ii, jj] =
                    (1-tx)*(1-ty)*z11 +
                    tx*(1-ty)*z21 +
                    (1-tx)*ty*z12 +
                    tx*ty*z22
            end
        end
    end

    return xs_new, ys_new, Znew
end

function process_phase_matrix(xs, ys, Z;
    smooth::Bool = false,
    σ::Real = 1.0,
    interpolate::Bool = false,
    interpolation_factor::Int = 4,
)
    Zp = Float64.(Z)

    if smooth
        Zp = smooth_matrix_nan(Zp; σ=σ)
    end

    if interpolate
        return interpolate_matrix_bilinear(xs, ys, Zp; factor=interpolation_factor)
    else
        return xs, ys, Zp
    end
end


# ============================================================
# Generic phase metric specification
# ============================================================

function phase_metric(source::Symbol;
    name::Symbol = source,
    title::AbstractString = String(source),
    label::AbstractString = String(source),
    zlabel = String(source),
    colormap = :viridis,
)
    return (
        source = source,
        name = name,
        mean_col = Symbol(name, :_mean),
        sd_col = Symbol(name, :_sd),
        title = title,
        label = label,
        zlabel = zlabel,
        colormap = colormap,
    )
end

# ============================================================
# Generic model phase specification
# ============================================================

function phase_specification(model::Symbol, P_grid)
    return phase_specification(Val(model), P_grid)
end

function phase_specification(::Val{M}, P_grid) where {M}
    error("No phase specification defined for model = $(M).")
end


# ============================================================
# Generic ABM phase-grid plotting utilities
# ============================================================

# ------------------------------------------------------------
# Metric specification helper
# ------------------------------------------------------------
function phase_metric(source::Symbol;
    name::Symbol = source,
    title::AbstractString = String(source),
    label::AbstractString = String(source),
    zlabel = String(source),
    colormap = :viridis,
)
    return (
        source = source,
        name = name,
        mean_col = Symbol(name, :_mean),
        sd_col = Symbol(name, :_sd),
        title = title,
        label = label,
        zlabel = zlabel,
        colormap = colormap,
    )
end


# ============================================================
# Summarise phase-grid results
# ============================================================

function summarise_phase(df::DataFrame;
    xcol::Symbol,
    ycol::Symbol,
    metrics,
)
    combine_args = Pair[]

    for m in metrics
        push!(combine_args, m.source => mean => m.mean_col)
        push!(combine_args, m.source => std  => m.sd_col)
    end

    push!(combine_args, nrow => :n_runs)

    summary = combine(
        groupby(df, [xcol, ycol]),
        combine_args...,
    )

    sort!(summary, [ycol, xcol])

    return summary
end


# ============================================================
# Heatmaps
# ============================================================

function plot_phase_heatmaps(
    xvals,
    yvals,
    Zs;
    metrics,
    xlabel = "x",
    ylabel = "y",
    size_per_panel = 520,
)
    nmetrics = length(metrics)

    fig = Figure(size = (size_per_panel * nmetrics, 420))

    for (j, m) in enumerate(metrics)
        ax = Axis(fig[1, 2j - 1],
            xlabel = xlabel,
            ylabel = ylabel,
            title = m.title,
        )

        hm = heatmap!(
            ax,
            xvals,
            yvals,
            Zs[m.name];
            colormap = m.colormap,
        )

        Colorbar(fig[1, 2j], hm, label = m.label)
    end

    return fig
end


# ============================================================
# Surface plots
# ------------------------------------------------------------
# For Makie.surface!, matrix dimensions should match:
#   size(Z) == (length(x), length(y))
#
# summary_to_matrix stores heatmap matrices as:
#   size(Z) == (length(y), length(x))
#
# Therefore, transpose before surface plotting.
# ============================================================

function plot_phase_surfaces(
    xvals,
    yvals,
    Zs;
    metrics,
    xlabel = "x",
    ylabel = "y",
    size_per_panel = 520,
)
    nmetrics = length(metrics)

    fig = Figure(size = (size_per_panel * nmetrics, 480))

    for (j, m) in enumerate(metrics)
        ax = Axis3(fig[1, j],
            xlabel = xlabel,
            ylabel = ylabel,
            zlabel = m.zlabel,
            title = m.title,
            azimuth = 1.0,
            elevation = 0.45,
        )

        surface!(ax, xvals, yvals, Zs[m.name]'; colormap = m.colormap)
    end

    return fig
end


# ============================================================
# Generic phase plotting wrapper
# ============================================================

function phase_plots(results::DataFrame;
    xcol::Symbol,
    ycol::Symbol,
    xvals,
    yvals,
    metrics,
    xlabel = String(xcol),
    ylabel = String(ycol),
    smooth::Bool = false,
    σ::Real = 1.0,
    interpolate::Bool = false,
    interpolation_factor::Int = 4,
)
    summary = summarise_phase(
        results;
        xcol = xcol,
        ycol = ycol,
        metrics = metrics,
    )

    raw_Zs = Dict{Symbol, Matrix}()
    plot_Zs = Dict{Symbol, Matrix}()

    x_raw_ref = nothing
    y_raw_ref = nothing
    x_plot_ref = nothing
    y_plot_ref = nothing

    for m in metrics
        xs, ys, Z = summary_to_matrix(
            summary,
            xcol,
            ycol,
            m.mean_col;
            xs_expected = xvals,
            ys_expected = yvals,
        )

        xs_plot, ys_plot, Z_plot = process_phase_matrix(
            xs,
            ys,
            Z;
            smooth = smooth,
            σ = σ,
            interpolate = interpolate,
            interpolation_factor = interpolation_factor,
        )

        raw_Zs[m.name] = Z
        plot_Zs[m.name] = Z_plot

        x_raw_ref = xs
        y_raw_ref = ys
        x_plot_ref = xs_plot
        y_plot_ref = ys_plot
    end

    fig_heatmap = plot_phase_heatmaps(
        x_plot_ref,
        y_plot_ref,
        plot_Zs;
        metrics = metrics,
        xlabel = xlabel,
        ylabel = ylabel,
    )

    fig_surface = plot_phase_surfaces(
        x_plot_ref,
        y_plot_ref,
        plot_Zs;
        metrics = metrics,
        xlabel = xlabel,
        ylabel = ylabel,
    )

    return (
        summary = summary,

        xvals_raw = x_raw_ref,
        yvals_raw = y_raw_ref,
        Zs_raw = raw_Zs,

        xvals = x_plot_ref,
        yvals = y_plot_ref,
        Zs = plot_Zs,

        smooth = smooth,
        σ = σ,
        interpolate = interpolate,
        interpolation_factor = interpolation_factor,

        fig_heatmap = fig_heatmap,
        fig_surface = fig_surface,
    )
end


# ============================================================
# Generic model phase specification dispatcher
# ============================================================

function phase_specification(model::Symbol, P_grid)
    return phase_specification(Val(model), P_grid)
end

function phase_specification(::Val{M}, P_grid) where {M}
    error("No phase specification defined for model = $(M).")
end

# ============================================================
# Grid value helpers
# ============================================================

function grid_values(grid::NamedTuple, name::Symbol)
    return collect(getproperty(grid, name))
end

function grid_values(grid::AbstractVector{<:NamedTuple}, name::Symbol)
    isempty(grid) && return []
    return sort(unique(getproperty.(grid, name)))
end

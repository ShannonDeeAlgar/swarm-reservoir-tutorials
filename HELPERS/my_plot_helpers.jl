# my_plot_helpers.jl

include("my_geometry_helpers.jl")

using CairoMakie
using DataFrames
using Statistics
using LinearAlgebra

# --------------------------------------------------
# Helper: validate requested columns
# --------------------------------------------------

function _check_columns(df::DataFrame, cols)
    missing = [c for c in cols if c ∉ names(df)]
    isempty(missing) || error("Missing columns in dataframe: $(missing)")
end

# --------------------------------------------------
# Generic time series plot
#
# Works for:
# - global dataframes: one row per time
# - local dataframes: one row per (time, node), with optional grouping
#
# groupby:
#   nothing        -> plot columns directly
#   :node          -> plot one line per node for each property
#
# reduce:
#   nothing        -> no aggregation
#   mean, median, maximum, minimum, std, etc
#                   -> aggregate over groupby dimension at each time
# --------------------------------------------------

using CairoMakie
using DataFrames
using Statistics

# --------------------------------------------------
# Helpers
# --------------------------------------------------

_as_symbol(x::Symbol) = x
_as_symbol(x::AbstractString) = Symbol(x)

function _df_symbol_map(df::DataFrame)
    Dict(Symbol(n) => n for n in names(df))
end

function _resolve_df_name(df::DataFrame, col)
    sym = _as_symbol(col)
    name_map = _df_symbol_map(df)
    haskey(name_map, sym) || error("Column $(repr(sym)) not found. Available columns: $(names(df))")
    return name_map[sym]
end

function _check_columns(df::DataFrame, cols)
    name_map = _df_symbol_map(df)
    missing = [_as_symbol(c) for c in cols if !haskey(name_map, _as_symbol(c))]
    isempty(missing) || error("Missing columns in dataframe: $(missing)")
end

# --------------------------------------------------
# Generic time series plot
# --------------------------------------------------

function plot_timeseries(
    df::DataFrame,
    properties;
    tcol = :t,
    groupby = nothing,
    reduce::Union{Nothing,Function} = nothing,
    filters::NamedTuple = NamedTuple(),
    fig_size::Tuple{Int,Int} = (800, 400),
    linewidth::Real = 2,
    title::AbstractString = "Time series",
    xlabel::AbstractString = "t",
    ylabel::Union{Nothing,AbstractString} = nothing,
    legend::Bool = true,
)

    props_in = properties isa Union{Symbol,AbstractString} ? [properties] : collect(properties)

    _check_columns(df, [tcol; props_in])

    tname = _resolve_df_name(df, tcol)
    pnames = [_resolve_df_name(df, p) for p in props_in]

    dfp = copy(df)

    # ----------------------------------------------
    # Optional filtering
    # ----------------------------------------------
    for (k, v) in pairs(filters)
        kname = _resolve_df_name(dfp, k)
        dfp = dfp[dfp[!, kname] .== v, :]
    end

    fig = Figure(size = fig_size)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = isnothing(ylabel) ? "" : ylabel,
    )

    # ----------------------------------------------
    # No grouping
    # ----------------------------------------------
    if isnothing(groupby)
        for (p_in, p) in zip(props_in, pnames)
            lines!(ax, dfp[!, tname], dfp[!, p], linewidth = linewidth, label = String(_as_symbol(p_in)))
        end

    # ----------------------------------------------
    # Grouped data
    # ----------------------------------------------
    else
        gname = _resolve_df_name(dfp, groupby)

        if isnothing(reduce)
            gdf = groupby(dfp, gname)

            for (p_in, p) in zip(props_in, pnames)
                for subdf in gdf
                    gval = first(subdf[!, gname])
                    lines!(
                        ax,
                        subdf[!, tname],
                        subdf[!, p],
                        linewidth = linewidth,
                        label = "$(String(_as_symbol(p_in))), $(String(_as_symbol(groupby)))=$(gval)",
                    )
                end
            end

        else
            grouped = groupby(dfp, tname)

            agg_df = combine(
                grouped,
                [p => reduce => p for p in pnames]...
            )

            for (p_in, p) in zip(props_in, pnames)
                lines!(
                    ax,
                    agg_df[!, tname],
                    agg_df[!, p],
                    linewidth = linewidth,
                    label = "$(String(_as_symbol(p_in))) ($(nameof(reduce)))",
                )
            end
        end
    end

    if legend
        axislegend(ax, position = :rb)
    end

    return fig
end


function arrows_from_3d(pos::Vector{SVector3}, vel::Vector{SVector3}; shaft_len::Float64 = 1.0)
    ap = Point3f[]
    ad = Vec3f[]
    sizehint!(ap, length(pos))
    sizehint!(ad, length(pos))

    @inbounds for i in eachindex(pos)
        d = unit(vel[i])
        push!(ap, Point3f(pos[i].x, pos[i].y, pos[i].z))
        push!(ad, Vec3f(shaft_len * d.x, shaft_len * d.y, shaft_len * d.z))
    end

    return ap, ad
end


function sphere_wireframe_curves(c::Point3f, r::Float64; nθ::Int = 60, nϕ::Int = 10, nλ::Int = 10)
    θs = range(0, 2π; length = nθ + 1)

    curves = Vector{Vector{Point3f}}()

    # Latitude circles
    ϕs = range(0, π; length = nϕ + 2)[2:end-1]
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

    # Longitude circles
    λs = range(0, 2π; length = nλ + 1)[1:end-1]
    ϕcurve = range(0, π; length = nθ)
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
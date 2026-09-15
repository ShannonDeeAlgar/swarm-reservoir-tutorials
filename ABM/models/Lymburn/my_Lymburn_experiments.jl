# ============================================================
# my_Lymburn_experiments.jl
# Parameter-sweep infrastructure for reproducing Figs 6-9 of Lymburn et al.
# (2021): a grid over (Kr, Ka) for a given Kp, recording time-averaged
# order parameters (via the generic run_model_experiment, extended in
# ABM/my_ABM_experiments.jl to dispatch on model=:Lymburn) plus, for each
# grid point, RC performance R and consistency capacity Theta.
# ============================================================
#
# PAPER'S ACTUAL PARAMETERS (Sec III), for a full-scale reproduction:
#   N              = 200
#   Kr, Ka         swept in [0.001, 100] ("pseudo-logarithmic"; the paper
#                   doesn't give the exact grid points/count -- a dense
#                   `10.0 .^ range(-3, 2, length=...)` is the best-effort
#                   reconstruction)
#   Kp             0 ("safe world", undriven) and 100 ("risky world",
#                   Lorenz-driven) -- run the sweep once for each
#   Kh, Kf         2, 20 (held fixed across the whole sweep)
#   rr, ra, rp     1, 1, 2 (held fixed)
#   s, alpha, beta 10, 200, 0.1 (held fixed -- friction target speed,
#                   tanh force-cap amplitude/steepness)
#   dt             0.02
#   sim length     5x10^4 steps per grid point, discarding a 1000-step
#                   transient before time-averaging order parameters
#   RC task        predict the (uniformly dt=0.02-sampled, std=2-rescaled)
#                   Lorenz x-coordinate 0.5 time units (~half a Lyapunov
#                   time) ahead; kernel observation layer with M=200
#
# All of these are `LymburnParams`/`Lymburn_params_from_preset` fields or
# `run_Lymburn_rc_sweep` keyword arguments here, so a paper-scale run is a
# matter of setting them to the values above -- see `Tutorial_swarmRC.ipynb`
# for a worked (small-scale) example of the exact task construction
# (`horizon`, the Lorenz rescaling, etc.).
#
# COST WARNING: the paper's own scale above is genuinely expensive --
# plausibly hours of compute (200^2-scaling dynamics x 5x10^4 steps x a
# dense 2D grid x 2 Kp scenarios x however many ensemble seeds -- see
# below). Default any exploratory/demo run to a small grid, small N/step
# count, and nensemble=1 (as `Tutorial_swarmRC.ipynb`'s demo does); scale up
# deliberately, and expect to wait, for anything resembling the paper's
# actual figures.
#
# ENSEMBLES: `run_Lymburn_rc_sweep`'s `nensemble` kwarg (default 1) repeats
# each grid point with `nensemble` different seeds instead of one -- a
# single run per grid point is noisy (see `RC/my_consistency.jl`'s notes on
# Theta's run-to-run variance), so a single-seed sweep is a mechanism demo,
# not a replication. Pair `nensemble>1` with `summarise_experiment` (from
# `ABM/my_ABM_experiments.jl`) to collapse the resulting one-row-per-run
# DataFrame down to one row per grid point with `_ens_mean`/`_ens_std`
# columns -- see `Tutorial_swarmRC.ipynb` for a worked example.
#
# Requires: ABM/models/Lymburn/load_Lymburn.jl, SWARM_RC/my_swarmRC_lymburn.jl,
# RC/my_consistency.jl, ABM/my_ABM_experiments.jl (for summarise_experiment)
# already loaded.

using DataFrames
using ProgressMeter
using Random
using Statistics
using LinearAlgebra
using CairoMakie
using LaTeXStrings

"""
    run_Lymburn_rc_sweep(base_simcfg, base_params, U, target; grid, shift, train_len,
                          predict_len, washout, ridge_grid=10.0 .^ (-3:1:5), cv_folds=5,
                          cv_gap=25, M=100, kneigh=5, n_repeats_consistency=3, nensemble=1,
                          seed0=1, show_progress=true)

For each `(Kr, Ka)` combination in `grid` (a NamedTuple of vectors, e.g.
`(Kr=[0.1,1,10], Ka=[0.01,0.1,1])`), overriding `base_params`:

1. Runs an undriven simulation (`base_simcfg`) and records time-averaged
   order parameters (`polarisation_mean`, `rotation_mean`) via the generic
   `run_model_experiment`/`mean_order_parameters` machinery -- this is the
   "safe world" behavioural characterisation (Fig 6a/9a style).
2. Builds a driven `LymburnReservoir` (`base_params`'s `Kp`/`rp` control the
   driven regime) and trains it on `U` to predict `target` (a pre-shifted
   scalar target the same length as `U`'s columns -- e.g. the Lorenz
   x-coordinate `horizon` steps ahead; see `Tutorial_swarmRC.ipynb` for how
   to build this), recording the Gaussian-kernel-observation-layer
   correlation `R` (Fig 8c/9b style) and consistency capacity `Theta`
   (Fig 9b-adjacent).

**Ridge selection (`ridge_grid`/`cv_folds`/`cv_gap`):** `R` is fit with
`train_readout_cv` (blocked, MSE-selected cross-validation over
`ridge_grid`, see `RC/my_reservoir_core.jl`) rather than one fixed
`ridgeλ` -- exactly how `Tutorial_swarmRC.ipynb` fits `R_kernel`. A single
fixed λ across the whole `(Kr, Ka)` grid can silently overfit or
underfit at grid points whose feature conditioning differs a lot from
where that λ happened to be tuned (see that function's docstring for a
worked example of a fixed λ producing a badly overfit readout). Defaults
(`ridge_grid=10.0 .^ (-3:1:5)`, `cv_folds=5`, `cv_gap=25`) match the
kernel-feature settings used for `R_kernel` in the tutorial, since this
sweep always uses the kernel observation layer.

**Ensembles (`nensemble`):** a single run per grid point is noisy -- both
the swarm's own dynamics and, per the diagnostic work in
`RC/my_consistency.jl`, `Theta` specifically can vary a lot run-to-run at
short horizons. `nensemble` (default `1`, i.e. the old single-run
behaviour) repeats each grid point `nensemble` times with different seeds
(`run_id`/`ensemble_id`/`seed` columns identify which is which, matching
the convention `ABM/my_ABM_experiments.jl`'s `run_model_experiment` already
uses) and returns one row per `(combo, ensemble)` pair -- *not*
pre-averaged. Call the generic `summarise_experiment(df, [:Kr, :Ka];
statcols=[:polarisation_mean, :rotation_mean, :R, :Theta])` on the result
to collapse to one row per grid point with `_ens_mean`/`_ens_std` columns
before plotting. For anything resembling the paper's actual Fig 6-9
phase diagrams, `nensemble` in the 5-10+ range (not the default 1) is what
"properly replicating" the paper means here -- a single seed per grid
point, even at the paper's own `N=200`/`5x10⁴` steps, is not what the
paper's own figures are (those are themselves already time/ensemble
averages).

**`measure_phiP_driven` (default `false`):** `polarisation_mean` above is
always measured from an *undriven* pre-run (`Kp=0`), matching Fig 6a/9a's
own "safe world" methodology -- but for a Fig 8(c)-style performance-vs-
polarisation plot, this can be misleading: a swarm that looks fully rigid
undriven can still respond normally once the predator is active (the
predator's forcing is often strong enough to break a lock that only
exists in the swarm's own undriven dynamics). Set `measure_phiP_driven=true`
to also record `polarisation_mean_driven`, computed from the *driven* run
already being simulated for `R` (via `log_raw=true` on that same pass --
no extra simulation needed), averaged over the same post-washout window
used for training/testing. Use `polarisation_mean_driven`, not
`polarisation_mean`, as the x-axis for a Fig 8(c) reproduction.

Returns a `DataFrame`, one row per `(combo, ensemble)` pair.
"""
function _lymburn_polarisation_from_raw_state(rs)
    V = rs.V
    N = size(V, 2)
    mx = 0.0
    my = 0.0
    @inbounds for i in 1:N
        vx, vy = V[1, i], V[2, i]
        nrm = sqrt(vx^2 + vy^2)
        if nrm > 1e-12
            mx += vx / nrm
            my += vy / nrm
        end
    end
    return sqrt((mx / N)^2 + (my / N)^2)
end

function run_Lymburn_rc_sweep(
    base_simcfg::SimulationConfig,
    base_params::LymburnParams,
    U::AbstractMatrix,
    target::AbstractVector;
    grid::NamedTuple,
    shift::Int,
    train_len::Int,
    predict_len::Int,
    washout::Int,
    ridge_grid = 10.0 .^ (-3:1:5),
    cv_folds::Int = 5,
    cv_gap::Int = 25,
    M::Int = 100,
    kneigh::Int = 5,
    n_repeats_consistency::Int = 3,
    transient_steps::Int = 0,
    nensemble::Int = 1,
    seed0::Int = 1,
    measure_phiP_driven::Bool = false,
    show_progress::Bool = true,
)
    @assert size(U, 2) == length(target) "U and target must have the same number of timesteps."
    Tdata = size(U, 2)
    @assert shift + train_len + predict_len <= Tdata "Not enough driven data for shift+train_len+predict_len."

    combos = parameter_combinations(grid)
    rows = NamedTuple[]

    ntotal = length(combos) * nensemble
    prog = show_progress ? Progress(ntotal; desc = "Lymburn RC sweep", dt = 0.5, showspeed = false) : nothing

    run_id = 0
    for (combo_id, pars) in enumerate(combos)
        P = update_LymburnParams(base_params; pars...)

        for ensemble_id in 1:nensemble
            run_id += 1
            seed = seed0 + run_id - 1

            # --- 1. Undriven behaviour: order parameters ---
            simcfg = SimulationConfig(steps = base_simcfg.steps, dt = base_simcfg.dt, seed = seed)
            P_undriven = update_LymburnParams(P; Kp = 0.0)
            out = simulate_Lymburn_2d(simcfg, P_undriven; show_progress = false)
            order_stats = mean_order_parameters(out; transient_steps = transient_steps)

            # --- 2. Driven RC performance + consistency capacity ---
            res = build_Lymburn_reservoir(P, base_simcfg.dt; rng = MersenneTwister(seed))

            Utrain_full = @view U[:, 1:(shift + train_len)]
            K = build_observation_layer_spatial_gaussian!(res, Utrain_full;
                washout = washout, rng = MersenneTwister(seed + 1), M = M, kneigh = kneigh, show_progress = false)

            reset!(res; rng = MersenneTwister(seed))
            Xfeat, S = collect_features(res, @view(U[:, 1:(shift + train_len + predict_len)]);
                rng = MersenneTwister(seed + 2), log_raw = measure_phiP_driven, reset_res = true,
                feature_fn = feature_map, obs = K, show_progress = false)

            Y = reshape(target[1:(shift + train_len + predict_len)], 1, :)
            Xtr = Xfeat[:, (shift + washout + 1):(shift + train_len)]
            Ytr = Y[:, (shift + washout + 1):(shift + train_len)]
            Xte = Xfeat[:, (shift + train_len + 1):(shift + train_len + predict_len)]
            Yte = Y[:, (shift + train_len + 1):(shift + train_len + predict_len)]

            Wout, μx, σx, _, _ = train_readout_cv(Xtr, Ytr; ridge_grid = ridge_grid, n_folds = cv_folds, gap = cv_gap)
            Yhat = apply_readout(Wout, Xte, μx, σx)
            R = cor(vec(Yhat), vec(Yte))

            reset!(res; rng = MersenneTwister(seed))
            cc = consistency_capacity(res, Utrain_full;
                n_repeats = n_repeats_consistency, washout = washout,
                rng = MersenneTwister(seed + 3), obs = K, show_progress = false)

            driven_extra = measure_phiP_driven ?
                (polarisation_mean_driven = mean(_lymburn_polarisation_from_raw_state(S[t]) for t in (shift + washout + 1):length(S)),) :
                NamedTuple()

            push!(rows, merge(
                (run_id = run_id, combo_id = combo_id, ensemble_id = ensemble_id, seed = seed),
                pars,
                order_stats,
                (R = R, Theta = cc.Theta),
                driven_extra,
            ))

            show_progress && next!(prog)
        end
    end

    return DataFrame(rows)
end

# ============================================================
# Visualisation: heatmaps over the (Kr, Ka) grid, matching the style of
# the paper's Fig 6/8 parameter-space plots.
# ============================================================

function _sweep_grid_matrix(df::DataFrame, value::Symbol, xcol::Symbol, ycol::Symbol)
    xs = sort(unique(df[!, xcol]))
    ys = sort(unique(df[!, ycol]))
    Z = fill(NaN, length(xs), length(ys))
    for row in eachrow(df)
        i = findfirst(==(row[xcol]), xs)
        j = findfirst(==(row[ycol]), ys)
        Z[i, j] = row[value]
    end
    return xs, ys, Z
end

"""
    plot_Lymburn_sweep_heatmap(df; value=:R, xcol=:Kr, ycol=:Ka, log_scale=true,
                                colormap=:viridis, title=string(value))

Visualise one metric from a `run_Lymburn_rc_sweep` result as a heatmap over
the swept `(xcol, ycol)` grid (default `Kr`/`Ka`) -- e.g.
`value=:polarisation_mean`, `:rotation_mean`, `:R`, or `:Theta`. `log_scale`
plots `log10` of the axis values, matching the paper's own "pseudo-logarithmic"
sweep (Sec III) and its log-scaled Fig 6/8 axes.
"""
function plot_Lymburn_sweep_heatmap(df::DataFrame;
    value::Symbol = :R,
    xcol::Symbol = :Kr,
    ycol::Symbol = :Ka,
    log_scale::Bool = true,
    colormap = :viridis,
    title::String = String(value),
    fig_size::Tuple{Int,Int} = (480, 400),
)
    xs, ys, Z = _sweep_grid_matrix(df, value, xcol, ycol)
    xplot = log_scale ? log10.(xs) : xs
    yplot = log_scale ? log10.(ys) : ys

    fig = Figure(size = fig_size)
    ax = Axis(fig[1, 1],
        xlabel = log_scale ? "log10($xcol)" : String(xcol),
        ylabel = log_scale ? "log10($ycol)" : String(ycol),
        title = title,
    )
    hm = heatmap!(ax, xplot, yplot, Z; colormap = colormap)
    Colorbar(fig[1, 2], hm)
    return fig
end

"""
    plot_Lymburn_sweep_grid(df; values=[:polarisation_mean, :rotation_mean, :R, :Theta],
                             xcol=:Kr, ycol=:Ka, log_scale=true)

Convenience wrapper: one `plot_Lymburn_sweep_heatmap` panel per entry in
`values`, laid out in a single row -- a quick multi-metric overview of a
sweep (order parameters alongside RC performance/consistency), matching how
the paper shows polarisation, rotation, and performance side by side
(Figs 6, 8).
"""
function plot_Lymburn_sweep_grid(df::DataFrame;
    values::Vector{Symbol} = [:polarisation_mean, :rotation_mean, :R, :Theta],
    xcol::Symbol = :Kr,
    ycol::Symbol = :Ka,
    log_scale::Bool = true,
    colormap = :viridis,
    panel_size::Tuple{Int,Int} = (330, 320),
)
    n = length(values)
    fig = Figure(size = (panel_size[1] * n, panel_size[2]))

    for (k, val) in enumerate(values)
        xs, ys, Z = _sweep_grid_matrix(df, val, xcol, ycol)
        xplot = log_scale ? log10.(xs) : xs
        yplot = log_scale ? log10.(ys) : ys

        ax = Axis(fig[1, 2k - 1],
            xlabel = log_scale ? "log10($xcol)" : String(xcol),
            ylabel = log_scale ? "log10($ycol)" : String(ycol),
            title = String(val),
        )
        hm = heatmap!(ax, xplot, yplot, Z; colormap = colormap)
        Colorbar(fig[1, 2k], hm)
    end

    return fig
end

"""
    _bivariate_KrKa_color(kr_norm, ka_norm)

Two-channel colour for a `(Kr, Ka)` point, `kr_norm`/`ka_norm` each in
`[0, 1]` (normalised `log10(Kr)`/`log10(Ka)`). Matching the orientation of
the paper's Fig. 8(c) key, `Ka` controls hue (blue -> cyan -> green ->
yellow -> red from bottom to top) and `Kr` lightens that hue toward white
from left to right. This is an approximation of the paper's inset
convention (independently varying colour with both axes, legible as a 2D
key), not a pixel-exact reproduction of its (undocumented) colormap.
"""
function _bivariate_KrKa_color(kr_norm::Real, ka_norm::Real)
    hue_stops = (
        RGBf(0.06, 0.06, 0.45),
        RGBf(0.10, 0.55, 0.95),
        RGBf(0.25, 0.80, 0.35),
        RGBf(0.95, 0.85, 0.15),
        RGBf(0.85, 0.15, 0.05),
    )
    n = length(hue_stops)
    pos = clamp(ka_norm, 0.0, 1.0) * (n - 1) + 1
    i0 = clamp(floor(Int, pos), 1, n - 1)
    frac = clamp(pos - i0, 0.0, 1.0)
    base = hue_stops[i0] * (1 - frac) .+ hue_stops[i0 + 1] * frac

    lighten = 0.7 * clamp(kr_norm, 0.0, 1.0)
    lit = base * (1 - lighten) .+ RGBf(1, 1, 1) * lighten
    return RGBAf(lit.r, lit.g, lit.b, 1.0)
end

"""
    plot_Lymburn_performance_vs_polarisation(df; phiP_col=:polarisation_mean_driven,
                                              Rcol=:R, Kr_col=:Kr, Ka_col=:Ka)

Reproduces the style of Lymburn et al.'s Fig 8(c): performance (`Rcol`)
against polarisation order (`phiP_col`), one point per row of `df`,
coloured by a genuine 2D `(Kr_col, Ka_col)` encoding (see
`_bivariate_KrKa_color`) with a matching 2D legend swatch, matching the
paper's own inset convention rather than a single 1D colourbar. `df` is a
`run_Lymburn_rc_sweep` result (or `summarise_experiment`'s
ensemble-summarised version -- pass e.g.
`phiP_col=:polarisation_mean_driven_ens_mean, Rcol=:R_ens_mean` in that
case).

**Which `phiP_col` to use**: pass `:polarisation_mean_driven` (requires
`run_Lymburn_rc_sweep(...; measure_phiP_driven=true)`), not the default
`:polarisation_mean` (undriven) -- undriven polarisation does not
reliably predict driven performance in this implementation (a swarm can
look fully rigid undriven and still respond normally once actually
driven), and using it produces a visibly worse match to the paper's own
Fig 8(c) shape (a spurious high-performance cluster appears at
`Φ_P≈1`that isn't in the paper's own figure).
"""
function plot_Lymburn_performance_vs_polarisation(df::DataFrame;
    phiP_col::Symbol = :polarisation_mean_driven,
    Rcol::Symbol = :R,
    Kr_col::Symbol = :Kr,
    Ka_col::Symbol = :Ka,
    fig_size::Tuple{Int,Int} = (820, 600),
    markersize::Real = 10,
    legend_res::Int = 40,
)
    logKr = log10.(df[!, Kr_col])
    logKa = log10.(df[!, Ka_col])
    kr_lo, kr_hi = extrema(logKr)
    ka_lo, ka_hi = extrema(logKa)
    kr_norm = (logKr .- kr_lo) ./ (kr_hi - kr_lo + 1e-12)
    ka_norm = (logKa .- ka_lo) ./ (ka_hi - ka_lo + 1e-12)
    colors = [_bivariate_KrKa_color(kr, ka) for (kr, ka) in zip(kr_norm, ka_norm)]

    fig = Figure(size = fig_size)
    ax = Axis(fig[1, 1],
        xlabel = L"\Phi_P",
        ylabel = L"R",
        title = "Performance vs. polarisation",
        aspect = 1,
    )
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)
    scatter!(ax, df[!, phiP_col], df[!, Rcol]; color = colors, markersize = markersize)

    legend_img = [_bivariate_KrKa_color((i - 1) / (legend_res - 1), (j - 1) / (legend_res - 1))
                  for i in 1:legend_res, j in 1:legend_res]
    ax_legend = Axis(fig[1, 2], xlabel = L"K_r", ylabel = L"K_a",
        xticks = ([1, legend_res], ["low", "high"]), yticks = ([1, legend_res], ["low", "high"]),
        aspect = 1, title = "key")
    image!(ax_legend, legend_img)
    colgap!(fig.layout, 1, 8)

    return fig
end

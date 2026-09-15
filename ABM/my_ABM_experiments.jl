# ============================================================
# my_ABM_experiments.jl
# Generic experiment helpers
# ============================================================
#
# Include this file after loading the relevant model files, because
# simulate_model dispatches to model-specific simulation functions:
#
#   Couzin:
#       simulate_Couzin_2d
#       simulate_Couzin_3d
#
#   Helbing:
#       simulate_Helbing_2d
#
#   Territorial:
#       simulate_Mizzi_2d
#
# This file is intentionally model-agnostic. Model-specific parameter
# constructors and update functions should live in the model files.
#
# Main user-facing functions:
#
#   default_params
#   update_model_params
#   simulate_model
#   parameter_combinations
#   run_model_once
#   run_model_experiment
#   summarise_experiment
#   analyse_model_output
#
# ============================================================

using DataFrames
using ProgressMeter
ProgressMeter.ijulia_behavior(:clear)
using Statistics

# ============================================================
# Model dispatch
# ============================================================

function simulate_model(
    model::Symbol,
    simcfg::SimulationConfig,
    P;
    dim::Int = 2,
    kwargs...,
)
    Tname = nameof(typeof(P))

    P_inner, extra_kwargs = if Tname === :HelbingScenario
        model in (:Helbing, :helbing) ||
            error("Received a HelbingScenario but model = $model.")
        (P.P, (init_mode = P.init_mode,))

    elseif Tname === :CouzinScenario
        model in (:Couzin, :couzin) ||
            error("Received a CouzinScenario but model = $model.")
        (P.P, NamedTuple())

    else
        (P, NamedTuple())
    end

    # Use scenario's transformed dt if available, otherwise simcfg.dt
    dt = Tname in (:CouzinScenario, :HelbingScenario) ? P.dt : simcfg.dt

    merged = merge(extra_kwargs, kwargs)

    if model in (:Couzin, :couzin)
        sim_fn = dim == 2 ? simulate_Couzin_2d :
                 dim == 3 ? simulate_Couzin_3d :
                 error("Couzin model only supports dim = 2 or dim = 3.")
    elseif model in (:Helbing, :helbing)
        sim_fn = dim == 2 ? simulate_Helbing_2d :
                 dim == 3 ? simulate_Helbing_3d :
                 error("Helbing model only supports dim = 2 or dim = 3.")
    elseif model in (:Lymburn, :lymburn)
        sim_fn = dim == 2 ? simulate_Lymburn_2d :
                 error("Lymburn model only supports dim = 2 (no 3D variant).")
    elseif model in (:Mizzi, :mizzi)
        sim_fn = dim == 2 ? simulate_Mizzi_2d :
                 error("The Mizzi model only supports dim = 2.")
    else
        error("Unknown model = $model. Use :Couzin, :Helbing, :Lymburn, or :Mizzi.")
    end

    return sim_fn(simcfg, P_inner; dt_override = dt, merged...)
end

function default_params(
    model::Symbol;
    preset::Symbol = :base,
    kwargs...,
)
    if model in (:Couzin, :couzin)
        return default_Couzin_params(; preset = preset, kwargs...)

    elseif model in (:Helbing, :helbing)
        return default_Helbing_params(; preset = preset, kwargs...)

    elseif model in (:Lymburn, :lymburn)
        return Lymburn_params_from_preset(preset; kwargs...)

    elseif model in (:Mizzi, :mizzi)
        error("Territorial agents require explicit home locations; construct MizziParams(homes; ...).")

    else
        error("Unknown model = $model. Use :Couzin, :Helbing, :Lymburn, or :Mizzi.")
    end
end


function update_model_params(
    model::Symbol,
    P;
    kwargs...,
)
    if model in (:Couzin, :couzin)
        return update_CouzinParams(P; kwargs...)

    elseif model in (:Helbing, :helbing)
        return update_HelbingParams(P; kwargs...)

    elseif model in (:Lymburn, :lymburn)
        return update_LymburnParams(P; kwargs...)

    elseif model in (:Mizzi, :mizzi)
        return update_MizziParams(P; kwargs...)

    else
        error("Unknown model = $model. Use :Couzin, :Helbing, :Lymburn, or :Mizzi.")
    end
end


# ============================================================
# Parameter grids
# ============================================================

function parameter_combinations(grid::NamedTuple)
    isempty(keys(grid)) && return [NamedTuple()]

    ks = keys(grid)
    vs = values(grid)

    return [
        NamedTuple{ks}(tuple(v...))
        for v in Iterators.product(vs...)
    ]
end

function parameter_combinations(grid::AbstractVector{<:NamedTuple})
    return grid
end

# ============================================================
# Output summaries
# Summaries currently cover order parameters.
# ============================================================

function output_summary_row(out; transient_steps::Int = 0)
    # mean_order_parameters already returns a NamedTuple with all available
    # collected diagnostics, including speed_mean/speed_std if speed was recorded.
    return mean_order_parameters(out; transient_steps = transient_steps)
end

function analyse_model_output(
    out;
    transient_steps::Int = 0,
)
    order_df = has_order(out) ? global_order_parameter_df(out) : nothing

    summary = has_order(out) ?
        mean_order_parameters(out; transient_steps = transient_steps) :
        nothing

    return (
        order_df = order_df,
        summary = summary,
    )
end


# ============================================================
# Single run
# ============================================================

function run_model_once(
    model::Symbol,
    base_simcfg::SimulationConfig,
    base_params;
    dim::Int = 2,
    pars::NamedTuple = NamedTuple(),
    seed::Int = base_simcfg.seed,
    transient_steps::Int = 0,
    collect_order::Bool = true,
    collect_history::Bool = false,
    return_output::Bool = false,
    sim_kwargs...,
)
    simcfg = SimulationConfig(
        steps         = base_simcfg.steps,
        dt            = base_simcfg.dt,
        seed          = seed,
        focal         = base_simcfg.focal,
        update_scheme = base_simcfg.update_scheme,
    )

    P = update_model_params(model, base_params; pars...)

    # Default the simulator's own per-step bar off: simulate() defaults
    # show_progress=true, and without this every run in a sweep would spawn
    # its own Progress bar, fighting the sweep-level bar for the display
    # (and, under ijulia_behavior(:clear), forcing a Jupyter output-area
    # redraw on every one of potentially thousands of runs). Still
    # overridable via sim_kwargs for single manual runs.
    sim_kwargs = merge((show_progress = false,), NamedTuple(sim_kwargs))

    out = simulate_model(
        model,
        simcfg,
        P;
        dim = dim,
        collect_order = collect_order,
        collect_history = collect_history,
        sim_kwargs...,
    )

    stats = collect_order ?
        output_summary_row(out; transient_steps = transient_steps) :
        NamedTuple()

    if return_output
        return (
            out = out,
            stats = stats,
            params = P,
        )
    else
        return (
            stats = stats,
            params = P,
        )
    end
end


# ============================================================
# Ensemble / sweep / sweep + ensemble
# ============================================================

# ============================================================
# Progress display helper
# ============================================================

function experiment_progress_values(;
    run_id::Int,
    ntotal::Int,
    combo_id::Int,
    ncombos::Int,
    ensemble_id::Int,
    nensemble::Int,
    pars::NamedTuple,
    seed::Int,
)
    base_vals = [
        (:run,      "$run_id / $ntotal"),
        (:combo,    "$combo_id / $ncombos"),
        (:ensemble, "$ensemble_id / $nensemble"),
    ]

    par_vals = [
        (Symbol(k), v)
        for (k, v) in pairs(pars)
    ]

    return vcat(base_vals, par_vals, [(:seed, seed)])
end


# ============================================================
# Ensemble / sweep / sweep + ensemble
# ============================================================
function run_model_experiment(
    model::Symbol,
    base_simcfg::SimulationConfig,
    base_params;
    dim::Int = 2,
    grid = NamedTuple(),
    nensemble::Int = 1,
    seed0::Int = base_simcfg.seed,
    transient_steps::Int = 0,
    collect_order::Bool = true,
    collect_history::Bool = false,
    return_outputs::Bool = false,
    show_progress::Bool = true,
    progress_desc::String = "Running model experiment",
    sim_kwargs...,
)
    combos = parameter_combinations(grid)

    rows = NamedTuple[]
    outputs = return_outputs ? Any[] : nothing

    ncombos = length(combos)
    ntotal = ncombos * nensemble

    prog = show_progress ?
        Progress(ntotal; desc = progress_desc, dt = 0.5, showspeed = false) :
        nothing

    run_id = 0

    for (combo_id, pars) in enumerate(combos)
        for ensemble_id in 1:nensemble
            run_id += 1
            seed = seed0 + run_id - 1

            res = run_model_once(
                model,
                base_simcfg,
                base_params;
                dim = dim,
                pars = pars,
                seed = seed,
                transient_steps = transient_steps,
                collect_order = collect_order,
                collect_history = collect_history,
                return_output = return_outputs,
                sim_kwargs...,
            )

            row = merge(
                (
                    run_id = run_id,
                    combo_id = combo_id,
                    ensemble_id = ensemble_id,
                    seed = seed,
                    model = model,
                ),
                pars,
                res.stats,
            )

            push!(rows, row)

            if return_outputs
                push!(
                    outputs,
                    merge(
                        (
                            run_id = run_id,
                            combo_id = combo_id,
                            ensemble_id = ensemble_id,
                            seed = seed,
                            pars = pars,
                        ),
                        res,
                    ),
                )
            end

            if show_progress
                next!(
                    prog;
                    showvalues = experiment_progress_values(
                        run_id = run_id,
                        ntotal = ntotal,
                        combo_id = combo_id,
                        ncombos = ncombos,
                        ensemble_id = ensemble_id,
                        nensemble = nensemble,
                        pars = pars,
                        seed = seed,
                    )
                )
            end
        end
    end

    return DataFrame(rows), outputs
end


# ============================================================
# Summaries over ensembles
# ============================================================

# ============================================================
# Summaries over ensembles
# ============================================================

function summarise_experiment(
    results::DataFrame,
    groupcols;
    statcols = [
        :polarisation_mean,
        :rotation_mean,
        :dilation_mean,
        :abs_angular_momentum_mean,
        :speed_mean,
    ],
)
    # Only summarise columns that actually exist.
    # This keeps older result tables usable.
    available_statcols = [
        col for col in statcols
        if col in Symbol.(names(results))
    ]

    grouped = groupby(results, groupcols)

    pairs = Pair[]

    for col in available_statcols
        push!(pairs, col => mean => Symbol(col, :_ens_mean))
        push!(pairs, col => std  => Symbol(col, :_ens_std))
    end

    push!(pairs, nrow => :n)

    return combine(grouped, pairs...)
end


# ============================================================
# Parameter-grid helpers
# ============================================================

function grid_parameter_names(grid::NamedTuple)
    return collect(Symbol, keys(grid))
end

function grid_parameter_names(grid::AbstractVector{<:NamedTuple})
    isempty(grid) && return Symbol[]
    return collect(Symbol, keys(first(grid)))
end

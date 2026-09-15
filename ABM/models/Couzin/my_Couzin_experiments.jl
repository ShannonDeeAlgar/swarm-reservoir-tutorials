# run_Couzin_once
# run_Couzin_experiment
# run_Couzin_phase_experiment
# summarise_Couzin_results

# run_Couzin_once/run_Couzin_experiment are the sweep engine behind
# run_Couzin_phase_experiment below, not a general-purpose alternative to
# run_model_experiment (ABM/my_ABM_experiments.jl). The difference that
# justifies a separate implementation: they take an explicit `simulator`
# function argument, so a phase-diagram sweep can choose between e.g.
# simulate_Couzin_2d (periodic) and simulate_Couzin2002_2d (unbounded,
# paper-faithful) -- run_model_experiment's `model::Symbol` dispatch only
# supports one fixed simulator per model and can't make that choice.

function run_Couzin_once(
    simulator::Function,
    base_simcfg::SimulationConfig,
    base_params::CouzinParams;
    pars::NamedTuple = NamedTuple(),
    seed::Int = base_simcfg.seed,
    transient_steps::Int = 0,
    collect_history::Bool = false,
    collect_order::Bool = true,
    return_output::Bool = false,
    sim_kwargs...,
)
    P = isempty(keys(pars)) ? base_params : update_CouzinParams(base_params; pairs(pars)...)

    simcfg = SimulationConfig(
        steps = base_simcfg.steps,
        dt    = base_simcfg.dt,
        seed  = seed,
        focal = base_simcfg.focal,
    )

    # Default the simulator's own per-step bar off: simulate_Couzin2002_2d/3d
    # default show_progress=true, and without this every run in a sweep would
    # spawn its own Progress bar, fighting the sweep-level bar for the
    # display (and, under ijulia_behavior(:clear), forcing a Jupyter
    # output-area redraw on every one of thousands of runs). Still
    # overridable via sim_kwargs for single manual runs.
    sim_kwargs = merge((show_progress = false,), NamedTuple(sim_kwargs))

    # A small fraction of grid points -- typically small Zo combined with large Za
    # (e.g. the swarm/torus corner of a phase sweep) -- can occasionally exhaust
    # init_state's own max_init_attempts retries trying to give every agent a
    # visible neighbour. Retry the whole run with a different seed a few times
    # rather than aborting a multi-hour sweep over one rare init failure.
    max_retries = 20
    out = nothing
    for retry in 0:max_retries
        try
            out = simulator(
                simcfg,
                P;
                collect_order   = collect_order,
                collect_history = collect_history,
                sim_kwargs...,
            )
            break
        catch e
            (e isa ErrorException && occursin("Could not initialise a Couzin group", e.msg)) || rethrow()
            retry == max_retries && rethrow()
            @warn "Couzin init failed to find every agent a visible neighbour (seed=$(simcfg.seed)); retrying with a different seed." retry = retry + 1
            simcfg = SimulationConfig(
                steps = simcfg.steps, dt = simcfg.dt,
                seed  = seed + (retry + 1) * 1_000_003,
                focal = simcfg.focal,
            )
        end
    end

    stats = collect_order ? mean_order_parameters(out; transient_steps = transient_steps) : nothing

    return return_output ? (out = out, stats = stats, params = P) : (stats = stats, params = P)
end

function run_Couzin_experiment(
    simulator::Function,
    base_simcfg::SimulationConfig,
    base_params::CouzinParams;
    grid::NamedTuple = NamedTuple(),
    nensemble::Int = 1,
    seed0::Int = 1,
    transient_steps::Int = 0,
    collect_history::Bool = false,
    collect_order::Bool = true,
    return_outputs::Bool = false,
    show_progress::Bool = true,
    progress_desc::String = "Running Couzin experiment",
    sim_kwargs...,
)
    combos = isempty(keys(grid)) ? [NamedTuple()] : parameter_combinations(grid)

    rows = NamedTuple[]
    outputs = return_outputs ? Any[] : nothing

    ntotal = length(combos) * nensemble
    prog = show_progress ? Progress(ntotal; desc = progress_desc, dt = 0.5) : nothing

    run_id = 0
    for (combo_id, pars) in enumerate(combos)
        for ensemble_id in 1:nensemble
            run_id += 1
            seed = seed0 + run_id - 1

            res = run_Couzin_once(
                simulator,
                base_simcfg,
                base_params;
                pars = pars,
                seed = seed,
                transient_steps = transient_steps,
                collect_history = collect_history,
                collect_order = collect_order,
                return_output = return_outputs,
                sim_kwargs...,
            )

            stats = res.stats

            row = if collect_order && stats !== nothing
                merge(
                    (
                        run_id = run_id,
                        combo_id = combo_id,
                        ensemble_id = ensemble_id,
                        seed = seed,
                    ),
                    pars,
                    (
                        dilation_mean             = stats.dilation_mean,
                        rotation_mean             = stats.rotation_mean,
                        polarisation_mean         = stats.polarisation_mean,
                        abs_angular_momentum_mean = stats.abs_angular_momentum_mean,
                        dilation_std              = stats.dilation_std,
                        rotation_std              = stats.rotation_std,
                        polarisation_std          = stats.polarisation_std,
                        abs_angular_momentum_std  = stats.abs_angular_momentum_std,
                        transient_index           = stats.transient_index,
                        transient_time            = stats.transient_time,
                    )
                )
            else
                merge(
                    (
                        run_id = run_id,
                        combo_id = combo_id,
                        ensemble_id = ensemble_id,
                        seed = seed,
                    ),
                    pars,
                )
            end

            push!(rows, row)

            if return_outputs
                push!(outputs, merge(
                    (
                        run_id = run_id,
                        combo_id = combo_id,
                        ensemble_id = ensemble_id,
                        seed = seed,
                        pars = pars,
                    ),
                    res,
                ))
            end

            if show_progress
                next!(prog)
            end
        end
    end

    return DataFrame(rows), outputs
end


function run_Couzin_phase_experiment(
    simulator::Function,
    base_simcfg::SimulationConfig,
    base_params::CouzinParams;
    Δro_vals,
    Δra_vals,
    rr,
    nensemble::Int = 1,
    seed0::Int = 1,
    transient_steps::Int = 0,
    collect_history::Bool = false,
    collect_order::Bool = true,
    return_outputs::Bool = false,
    show_progress::Bool = true,
    progress_desc::String = "Running Couzin phase experiment",
    sim_kwargs...
)
    all_results = DataFrame[]
    all_outputs = Any[]

    n_combos = length(Δro_vals) * length(Δra_vals)
    n_total_runs = n_combos * nensemble

    prog = show_progress ? Progress(n_total_runs; desc=progress_desc, dt=0.5) : nothing

    phase_combo_id = 1
    run_counter = 0

    for Δro in Δro_vals
        for Δra in Δra_vals

            # Couzin-style zone conversion
            Zr = rr
            Zo = rr + Δro
            Za = Zo + Δra

            param_grid = (
                Zr = [Zr],
                Zo = [Zo],
                Za = [Za],
            )

            for e in 1:nensemble
                seed = seed0 + run_counter

                results, outputs = run_Couzin_experiment(
                    simulator,
                    base_simcfg,
                    base_params;
                    grid = param_grid,
                    nensemble = 1,
                    seed0 = seed,
                    transient_steps = transient_steps,
                    collect_history = collect_history,
                    collect_order = collect_order,
                    return_outputs = return_outputs,
                    show_progress = false,
                    sim_kwargs...
                )

                # Add phase-diagram coordinates
                results.rr .= rr
                results.Δro .= Δro
                results.Δra .= Δra
                results.phase_combo_id .= phase_combo_id
                results.phase_combo_total .= n_combos
                results.ensemble_member .= e
                results.seed .= seed

                push!(all_results, results)

                if return_outputs && outputs !== nothing
                    append!(all_outputs, outputs)
                end

                run_counter += 1

                if show_progress
                    next!(
                        prog;
                        showvalues = [
                            (:run, "$run_counter / $n_total_runs"),
                            (:combo, "$phase_combo_id / $n_combos"),
                            (:ensemble, "$e / $nensemble"),
                            (:Δro, Δro),
                            (:Δra, Δra),
                            (:Zr, Zr),
                            (:Zo, Zo),
                            (:Za, Za),
                            (:seed, seed),
                        ]
                    )
                end
            end

            phase_combo_id += 1
        end
    end

    results_full = vcat(all_results...)

    if return_outputs
        return results_full, all_outputs
    else
        return results_full, nothing
    end
end


function summarise_Couzin_results(results::DataFrame, grid)
    groupcols = isempty(keys(grid)) ? [:combo_id] : vcat(collect(keys(grid)), [:combo_id])

    return combine(
        groupby(results, groupcols),
        :dilation_mean     => mean => :dilation_mean_avg,
        :dilation_mean     => std  => :dilation_mean_sd,
        :rotation_mean     => mean => :rotation_mean_avg,
        :rotation_mean     => std  => :rotation_mean_sd,
        :polarisation_mean => mean => :polarisation_mean_avg,
        :polarisation_mean => std  => :polarisation_mean_sd,
        :abs_angular_momentum_mean => mean => :abs_angular_momentum_mean_avg,
        :abs_angular_momentum_mean => std  => :abs_angular_momentum_mean_sd,
        nrow => :n_runs,
    )
end
using LinearAlgebra
using Statistics
using Random

"""
    validate_pipeline(res, U; observation=:raw_positions, obs=nothing,
                      target=nothing, prediction_mode=:teacher_forced,
                      input_from_output=nothing, dt_input=nothing,
                      scale_match=nothing, probe_steps=300, rng=MersenneTwister(91))

Validate a configured reservoir pipeline before an expensive training run.

This separates hard incompatibilities (`errors`) from suspicious but sometimes
intentional choices (`warnings`). It checks:

* reservoir/input dimensional compatibility and finite input values;
* coupling support;
* observation-method support and kernel dimensionality;
* rough input/swarm spatial and temporal scale ratios;
* feature constancy, effective rank, and kernel response on a short probe;
* whether the requested task can be rolled out autonomously.

For `prediction_mode=:free_run`, the readout target must either have the same
number of rows as `U`, or `input_from_output` must explicitly reconstruct a
valid next input. A scalar future-value target is a teacher-forced forecasting
task, not an autonomous generator. For a scalar delay reconstruction, train
the readout to predict the complete next delay vector (or supply a stateful
embedding update adapter).
"""
function validate_pipeline(
    res::AbstractReservoir,
    U::AbstractMatrix;
    observation::Symbol = :raw_positions,
    obs = nothing,
    target::Union{Nothing,AbstractMatrix,AbstractVector} = nothing,
    prediction_mode::Symbol = :teacher_forced,
    input_from_output::Union{Nothing,Function} = nothing,
    dt_input::Union{Nothing,Real} = nothing,
    scale_match = nothing,
    probe_steps::Int = 300,
    rng::AbstractRNG = MersenneTwister(91),
)
    errors = String[]
    warnings = String[]
    notes = String[]
    model = nameof(typeof(res))
    input_dim, T = size(U)

    T >= 2 || push!(errors, "Input needs at least two time samples.")
    all(isfinite, U) || push!(errors, "Input contains NaN or Inf values.")
    prediction_mode in (:teacher_forced, :free_run) ||
        push!(errors, "prediction_mode must be :teacher_forced or :free_run.")

    coupling = hasproperty(res, :coupling) ? res.coupling : nothing
    expected_dim = coupling isa TemperatureSpeedCoupling ? 1 :
        (model in (:CouzinReservoir, :LymburnReservoir, :MizziReservoir) ? 2 : nothing)
    expected_dim !== nothing && input_dim != expected_dim &&
        push!(errors, "$model requires $expected_dim input coordinates; got $input_dim.")

    if model in (:CouzinReservoir, :LymburnReservoir, :MizziReservoir, :LundReservoir)
        coupling isa InputCoupling ||
            push!(errors, "$model does not contain a valid InputCoupling.")
        if model == :CouzinReservoir &&
           !(coupling isa NoCoupling || coupling isa PredatorCoupling)
            push!(errors, "CouzinReservoir currently supports NoCoupling or PredatorCoupling, not $(typeof(coupling)).")
        end
        if model == :LymburnReservoir &&
           !(coupling isa NoCoupling || coupling isa PredatorCoupling)
            push!(errors, "LymburnReservoir supports NoCoupling or PredatorCoupling, not $(typeof(coupling)).")
        end
        if model == :MizziReservoir &&
           !(coupling isa NoCoupling || coupling isa MizziPreyCoupling)
            push!(errors, "MizziReservoir supports NoCoupling or MizziPreyCoupling, not $(typeof(coupling)).")
        end
        if model == :LundReservoir &&
           !(coupling isa NoCoupling || coupling isa TemperatureSpeedCoupling)
            push!(errors, "LundReservoir supports NoCoupling or TemperatureSpeedCoupling, not $(typeof(coupling)).")
        end
        if coupling isa PredatorCoupling
            coupling.rp > 0 || push!(errors, "PredatorCoupling.rp must be positive.")
            0 <= coupling.alpha <= 1 ||
                push!(warnings, "PredatorCoupling.alpha=$(coupling.alpha) is outside the usual blend/gain range [0,1].")
        end
        if coupling isa TemperatureSpeedCoupling
            coupling.base_speed > 0 || push!(errors, "TemperatureSpeedCoupling.base_speed must be positive.")
            coupling.min_speed > 0 || push!(errors, "TemperatureSpeedCoupling.min_speed must be positive.")
            coupling.max_speed >= coupling.min_speed ||
                push!(errors, "TemperatureSpeedCoupling speed bounds are reversed.")
            targets = coupling.base_speed .+ coupling.gain .* vec(U)
            clipped = count(x -> x < coupling.min_speed || x > coupling.max_speed, targets)
            clipped > 0 && push!(warnings,
                "Temperature-to-speed mapping clips $(clipped)/$(length(targets)) samples; reduce gain or rescale the scalar input.")
            push!(notes,
                "Global scalar input maps to target speed over $(round(minimum(clamp.(targets, coupling.min_speed, coupling.max_speed)), sigdigits=4))–$(round(maximum(clamp.(targets, coupling.min_speed, coupling.max_speed)), sigdigits=4)).")
        end
    end

    supported_observations = if model == :LundReservoir
        (:lund_aggregate, :raw_state, :tda_betti)
    elseif model == :LymburnReservoir
        (:raw_positions, :spatial_gaussian, :tda_betti)
    elseif model == :CouzinReservoir
        (:raw_positions, :spatial_gaussian, :coverage_kmeans, :tda_betti)
    elseif model == :MizziReservoir
        (:raw_state, :raw_positions, :tda_betti)
    else
        (:raw,)
    end
    observation in supported_observations ||
        push!(errors, "$observation is not supported by $model; use $(supported_observations).")

    if observation ∉ (:raw_positions, :raw_state, :raw, :lund_aggregate) && obs === nothing
        push!(errors, "Observation $observation requires a fitted observation layer (`obs`).")
    elseif obs isa KernelLayer
        state_dim = size(raw_state(res).P, 1)
        size(obs.C, 1) == state_dim ||
            push!(errors, "Kernel centres are $(size(obs.C,1))D but the swarm state is $(state_dim)D.")
        size(obs.C, 2) == length(obs.inv2w) ||
            push!(errors, "Kernel centre and width counts do not match.")
        all(isfinite, obs.C) && all(isfinite, obs.inv2w) ||
            push!(errors, "Observation layer contains NaN or Inf.")
        all(>(0), obs.inv2w) ||
            push!(errors, "Every Gaussian kernel must have a positive finite width.")
    elseif observation == :tda_betti
        nameof(typeof(obs)) == :TDAObservationLayer ||
            push!(errors, "Observation :tda_betti requires a TDAObservationLayer.")
        if obs !== nothing && nameof(typeof(obs)) == :TDAObservationLayer
            length(obs.εs) >= 2 || push!(errors, "TDA ε grid needs at least two values.")
            issorted(obs.εs) || push!(errors, "TDA ε grid must be sorted.")
            obs.periodic && obs.L === nothing && push!(errors, "Periodic TDA requires L.")
            obs.periodic && obs.representation != :position &&
                push!(errors, "Periodic TDA currently supports position features only.")
            expected_features = length(obs.εs) * length(obs.dims)
            push!(notes, "TDA observation has $expected_features Betti-curve features " *
                "($(length(obs.εs)) ε values × $(length(obs.dims)) dimensions).")
            res.P.N > 100 && push!(warnings,
                "TDA is being computed for $(res.P.N) agents at every sample; begin with N≤100 for interactive work.")
        end
    end

    Y = target === nothing ? nothing :
        (target isa AbstractVector ? reshape(target, 1, :) : target)
    if Y !== nothing
        size(Y, 2) == T ||
            push!(errors, "Target has $(size(Y,2)) samples but input has $T.")
        if prediction_mode == :free_run && size(Y, 1) != input_dim &&
           input_from_output === nothing
            push!(errors,
                "Autonomous free run needs $(input_dim) output coordinates to reconstruct the next input; " *
                "the target has $(size(Y,1)). Use teacher-forced forecasting, predict the full next " *
                "input/embedding vector, or provide input_from_output.")
        end
    elseif prediction_mode == :free_run
        push!(errors, "Free-run validation requires the training target.")
    end

    # Coarse physical scale audit. These are warnings because unusual scaling
    # can be an intentional experiment.
    input_span = maximum(vec(maximum(U; dims=2) .- minimum(U; dims=2)))
    input_step = median([norm(U[:, t+1] - U[:, t]) for t in 1:T-1])
    swarm_extent = if model == :CouzinReservoir
        float(res.P.L)
    elseif model == :LundReservoir
        float(res.P.L)
    elseif model == :LymburnReservoir
        2 * float(res.P.plot_extent)
    elseif model == :MizziReservoir
        2 * float(res.P.plot_extent)
    else
        NaN
    end
    swarm_step = model == :LundReservoir ? float(res.P.base_speed * res.dt) :
        (hasproperty(res, :P) && hasproperty(res.P, :speed) ?
        float(res.P.speed * res.dt) :
        (hasproperty(res, :P) && hasproperty(res.P, :s) ? float(res.P.s * res.dt) : NaN))

    spatial_input = !(coupling isa TemperatureSpeedCoupling)
    span_ratio = spatial_input && isfinite(swarm_extent) ? input_span / swarm_extent : NaN
    step_ratio = spatial_input && isfinite(swarm_step) && swarm_step > 0 ? input_step / swarm_step : NaN
    if isfinite(span_ratio) && (span_ratio < 0.02 || span_ratio > 2.0)
        push!(warnings, "Input spatial span/swarm extent = $(round(span_ratio, sigdigits=3)); inspect scale matching.")
    end
    if isfinite(step_ratio) && (step_ratio < 0.02 || step_ratio > 50.0)
        push!(warnings, "Input step/swarm step = $(round(step_ratio, sigdigits=3)); inspect temporal matching.")
    end
    dt_input !== nothing && hasproperty(res, :dt) &&
        !isapprox(float(dt_input), float(res.dt); rtol=1e-8, atol=0) &&
        push!(warnings, "Input dt=$(dt_input) differs from reservoir dt=$(res.dt); resample or document the mapping.")

    if scale_match !== nothing
        for (key, label) in ((:Rp_scaled, "spatial"), (:Vp_scaled, "speed"))
            hasproperty(scale_match, key) ||
                push!(warnings, "Scale-match result lacks $key ($label match cannot be verified).")
        end
        if hasproperty(scale_match, :Rp_scaled) && hasproperty(scale_match, :Rs)
            ratio = scale_match.Rp_scaled / scale_match.Rs
            (0.25 <= ratio <= 4) ||
                push!(warnings, "Explicit spatial match ratio Rp_scaled/Rs=$(round(ratio, sigdigits=3)).")
        end
        if hasproperty(scale_match, :Vp_scaled) && hasproperty(scale_match, :Vs)
            ratio = scale_match.Vp_scaled / scale_match.Vs
            (0.25 <= ratio <= 4) ||
                push!(warnings, "Explicit speed match ratio Vp_scaled/Vs=$(round(ratio, sigdigits=3)).")
        end
        push!(notes, "An explicit scale-match result was supplied.")
    elseif model in (:CouzinReservoir, :LymburnReservoir) && coupling isa PredatorCoupling
        push!(warnings, "No explicit scale-match result supplied; raw/z-score scaling may be intentional but is not physically validated.")
    end

    feature_summary = nothing
    if isempty(errors) && probe_steps > 1
        Tp = min(T, probe_steps)
        probe = clone_reservoir(res)
        X, _ = collect_features(probe, Matrix(U[:, 1:Tp]);
            rng=rng, log_raw=false, reset_res=true, obs=obs,
            feature_fn=feature_map, show_progress=false)
        σ = vec(std(X; dims=2))
        scale = max(maximum(abs, X), 1.0)
        constant_fraction = count(<=(1e-10 * scale), σ) / length(σ)
        s = svdvals(X .- mean(X; dims=2))
        effective_rank = isempty(s) || s[1] == 0 ? 0 :
            count(>(s[1] * 1e-8), s)
        rank_fraction = effective_rank / min(size(X)...)
        constant_fraction > 0.25 &&
            push!(warnings, "$(round(100constant_fraction, digits=1))% of probed observation features are nearly constant.")
        rank_fraction < 0.1 &&
            push!(warnings, "Probed observation has low effective-rank fraction $(round(rank_fraction, digits=3)).")
        if model == :LundReservoir && observation == :lund_aggregate && size(X, 1) >= 6
            clustered_fraction = vec(X[6, :])
            maximum(clustered_fraction) <= 0 && push!(warnings,
                "No Lund agents entered the clustered state during the probe; this is effectively a single-state reservoir. Increase the probe or retune the switching corridor.")
            minimum(clustered_fraction) >= 1 && push!(warnings,
                "All Lund agents remained clustered during the probe; the two-state mechanism has collapsed.")
            push!(notes, "Lund clustered fraction during probe: $(round(minimum(clustered_fraction), digits=3))–$(round(maximum(clustered_fraction), digits=3)).")
        end
        feature_summary = (; nfeatures=size(X,1), probe_steps=Tp,
            constant_fraction, effective_rank, rank_fraction)
    end

    return (
        valid = isempty(errors),
        model,
        input_dim,
        observation,
        prediction_mode,
        errors,
        warnings,
        notes,
        scales = (; input_span, input_step, swarm_extent, swarm_step,
                    span_ratio, step_ratio),
        features = feature_summary,
    )
end

function print_pipeline_validation(report)
    status = report.valid ? "PASS" : "FAIL"
    println("Pipeline validation: $status")
    println("  model=$(report.model), input_dim=$(report.input_dim), observation=$(report.observation), mode=$(report.prediction_mode)")
    for msg in report.errors
        println("  ERROR: ", msg)
    end
    for msg in report.warnings
        println("  WARNING: ", msg)
    end
    report.features !== nothing && println("  feature probe: ", report.features)
    return report
end



# my_reservoir_core.jl
using LinearAlgebra, Statistics, Random

using ProgressMeter
ProgressMeter.ijulia_behavior(:clear)

abstract type AbstractReservoir end

reset!(res::AbstractReservoir; rng=Random.default_rng()) =
    error("reset! not implemented for $(typeof(res))")

reservoir_step!(res::AbstractReservoir, u::AbstractVector; rng=Random.default_rng()) =
    error("reservoir_step! not implemented for $(typeof(res))")

feature_map(res::AbstractReservoir, obs=nothing) =
    error("feature_map not implemented for $(typeof(res)) with obs=$(typeof(obs))")

raw_state(res::AbstractReservoir) = nothing  # optional default


# ============================================================
# Optional reservoir-analysis interface
# ============================================================
clone_reservoir(res::AbstractReservoir) =
    error("clone_reservoir not implemented for $(typeof(res))")

perturb_state!(res::AbstractReservoir, ξ::AbstractVector) =
    error("perturb_state! not implemented for $(typeof(res))")

state_vector(res::AbstractReservoir, obs=nothing) = vec(copy(feature_map(res, obs)))

"Physical RMS distance used by perturbation diagnostics. Periodic models may override this."
function state_distance(r1::AbstractReservoir, r2::AbstractReservoir, obs=nothing)
    x1, x2 = state_vector(r1, obs), state_vector(r2, obs)
    @assert length(x1) == length(x2)
    return norm(x1 .- x2) / sqrt(length(x1))
end

# ============================================================
# Generic dataset helpers
# ============================================================

function slice_data(;
    data::AbstractMatrix,
    shift::Int,
    train_len::Int,
    predict_len::Int
)
    Ny, T = size(data)
    need = shift + train_len + predict_len + 1
    @assert need <= T """
    Not enough data: need shift+train_len+predict_len+1 ≤ T.
    Got $need vs $T.
    Increase input data length or reduce train/predict lengths.
    """

    # Train: U(t) -> Y(t)=U(t+1)
    Utrain = @view data[:, shift:(shift + train_len - 1)]
    Ytrain = @view data[:, (shift + 1):(shift + train_len)]

    # Test: free-run seed at t0, then compare to truth
    test_seed = @view data[:, (shift + train_len)]
    Ytest     = @view data[:, (shift + train_len + 1):(shift + train_len + predict_len)]

    return Utrain, Ytrain, test_seed, Ytest
end

"""
Default observation-layer builder: does nothing (returns `nothing`).

Accepts extra keyword arguments so specialised reservoirs (e.g. swarm kernels)
can pass knobs like `M`, `kneigh`, etc. without breaking other reservoirs.
"""
function build_observation_layer!(res::AbstractReservoir, Utrain::AbstractMatrix;
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    obs_kwargs...
)
    return nothing
end


# unify feature extraction: 2-arg call works for no-obs case
feature_map(res::AbstractReservoir, obs::Nothing) = feature_map(res)

# ============================================================
# Readout + training utilities
# ============================================================

struct RidgeReadout{T}
    λ::T
end

# bias + features only
function augment_with_bias(X::AbstractMatrix)
    T = size(X, 2)
    return vcat(ones(1, T), X)
end

"""
Compute per-feature mean/std across time.
X is (Nfeat × T). Returns μ, σ as vectors length Nfeat.
"""
function feature_stats(X::AbstractMatrix)
    μ = vec(mean(X; dims=2))
    σ = vec(std(X; dims=2)) .+ 1e-12
    return μ, σ
end

"""
Standardise features in-place: (X .- μ) ./ σ.
"""
function standardise!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector)
    @assert size(X,1) == length(μ) == length(σ)
    @inbounds for i in 1:size(X,1)
        X[i, :] .= (X[i, :] .- μ[i]) ./ σ[i]
    end
    return X
end

function fit_ridge(Φ::AbstractMatrix, Y::AbstractMatrix, rr::RidgeReadout)
    @assert size(Φ, 2) == size(Y, 2)
    λ = rr.λ
    A = Φ * Φ' .+ λ * I(size(Φ, 1))
    F = cholesky(Symmetric(A))
    Wout = (Y * Φ') / F
    return Wout
end

"""
Apply readout to a feature matrix X (Nfeat × T), returning Yhat (Ny × T).
Uses the same normalisation and bias convention as training.
"""
function apply_readout(Wout::AbstractMatrix, X::AbstractMatrix,
                       μx::AbstractVector, σx::AbstractVector)
    Xn = copy(X)
    standardise!(Xn, μx, σx)
    return Wout * augment_with_bias(Xn)
end

"""
Apply readout to a single feature vector x (length Nfeat), returning y (length Ny).
Uses eps() to avoid division by zero in σx.
"""
function apply_readout(Wout::AbstractMatrix, x::AbstractVector,
                       μx::AbstractVector, σx::AbstractVector)
    xn = (x .- μx) ./ (σx .+ eps())
    return Wout * vcat(1.0, xn)
end

# ============================================================
# Metrics
# ============================================================

function metrics(yhat::AbstractMatrix, ytrue::AbstractMatrix)
    @assert size(yhat) == size(ytrue)
    Ny, _ = size(ytrue)
    err = yhat .- ytrue

    mse_dim   = vec(mean(err.^2; dims=2))
    rmse_dim  = sqrt.(mse_dim)
    std_dim   = vec(std(ytrue; dims=2)) .+ 1e-12
    nrmse_dim = rmse_dim ./ std_dim

    μ_dim       = vec(mean(ytrue; dims=2))
    ss_res_dim  = vec(sum(err.^2; dims=2))
    ss_tot_dim  = [sum((view(ytrue, i, :) .- μ_dim[i]).^2) for i in 1:Ny]
    r2_dim      = 1 .- ss_res_dim ./ ss_tot_dim

    mse   = mean(vec(err).^2)
    rmse  = sqrt(mse)
    nrmse = rmse / (std(vec(ytrue)) + 1e-12)
    μ     = mean(vec(ytrue))
    r2    = 1 - sum(vec(err).^2) / (sum((vec(ytrue) .- μ).^2) + 1e-12)

    return (mse=mse, rmse=rmse, nrmse=nrmse, r2=r2,
            mse_dim=mse_dim, rmse_dim=rmse_dim, nrmse_dim=nrmse_dim, r2_dim=r2_dim)
end

function print_metrics(label::String, m)
    println("\n=== $label ===")
    println("Overall: MSE=$(m.mse)  RMSE=$(m.rmse)  NRMSE=$(m.nrmse)  R²=$(m.r2)")
    for i in 1:length(m.mse_dim)
        println("  dim $i: MSE=$(m.mse_dim[i])  RMSE=$(m.rmse_dim[i])  NRMSE=$(m.nrmse_dim[i])  R²=$(m.r2_dim[i])")
    end
end


# ============================================================
# Reservoir diagnostics
# ============================================================

"""
Basic summary statistics for a feature/state matrix X (Nfeat × T).
Useful for diagnosing richness, degeneracy, saturation, and conditioning.
"""
function state_matrix_summary(X::AbstractMatrix; rank_tol::Float64=1e-10)
    Nfeat, T = size(X)

    μ = vec(mean(X; dims=2))
    σ = vec(std(X; dims=2))

    # covariance across features
    Xc = X .- μ
    C = (Xc * Xc') / max(T - 1, 1)

    evals = eigvals(Symmetric(C))
    evals = real.(evals)
    evals_pos = max.(evals, 0.0)

    # Effective dimension / participation ratio
    s1 = sum(evals_pos)
    s2 = sum(evals_pos.^2) + 1e-12
    participation_ratio = s1^2 / s2

    # Effective rank from normalised spectrum entropy
    p = evals_pos ./ (sum(evals_pos) + 1e-12)
    spectral_entropy = -sum(pi > 0 ? pi * log(pi) : 0.0 for pi in p)
    effective_rank = exp(spectral_entropy)

    # Numerical rank of X itself
    sv = svdvals(Matrix(X))
    numerical_rank = count(>(rank_tol * maximum(sv)), sv)

    # Feature correlation summary
    corr_vals = Float64[]
    for i in 1:Nfeat-1, j in i+1:Nfeat
        si, sj = σ[i], σ[j]
        if si > 0 && sj > 0
            cij = cor(view(X, i, :), view(X, j, :))
            isfinite(cij) && push!(corr_vals, cij)
        end
    end

    # crude saturation indicators
    maxabs = maximum(abs.(X))
    meanabs = mean(abs.(X))
    frac_near_pm1 = mean(abs.(X) .> 0.95)

    # conditioning of Gram matrix
    G = X * X'
    gram_cond = try
        cond(G + 1e-10I)
    catch
        Inf
    end

    return (
        n_features = Nfeat,
        T = T,
        mean_feature_std = mean(σ),
        median_feature_std = median(σ),
        dead_feature_fraction = mean(σ .< 1e-8),
        maxabs = maxabs,
        meanabs = meanabs,
        frac_near_pm1 = frac_near_pm1,
        participation_ratio = participation_ratio,
        effective_rank = effective_rank,
        numerical_rank = numerical_rank,
        mean_abs_corr = isempty(corr_vals) ? 0.0 : mean(abs.(corr_vals)),
        max_abs_corr = isempty(corr_vals) ? 0.0 : maximum(abs.(corr_vals)),
        gram_condition = gram_cond,
        covariance_eigs = evals_pos,
        feature_std = σ
    )
end

"""
Train/test split helper for time-indexed data.
"""
function _time_split_indices(T::Int; train_fraction::Float64=0.7)
    Ttrain = clamp(floor(Int, train_fraction * T), 1, T-1)
    return 1:Ttrain, (Ttrain+1):T
end

"""
Lag-k linear memory curve using one selected input channel.

For each lag k, fit a linear readout from current reservoir features x_t
to past input u_{t-k}. Returns the per-lag R² and total MC = sum(R²_k).

This is the classic Jaeger-style linear memory measure.
"""
function memory_capacity_curve(
    res::AbstractReservoir,
    U::AbstractMatrix;
    input_dim::Int=1,
    maxlag::Int=100,
    washout::Int=100,
    ridgeλ::Float64=1e-6,
    train_fraction::Float64=0.7,
    rng=Random.default_rng(),
    obs=nothing,
    feature_fn::Function=feature_map,
    show_progress::Bool=true,
)
    @assert 1 <= input_dim <= size(U, 1)

    X, _ = collect_features(res, U;
        rng=rng, log_raw=false, reset_res=true,
        feature_fn=feature_fn, obs=obs, show_progress=show_progress,
    )

    s = vec(U[input_dim, :])
    T = size(X, 2)

    lags = collect(1:maxlag)
    r2 = fill(NaN, maxlag)

    prog = show_progress ? Progress(maxlag; desc="memory_capacity_curve (per lag)", showspeed=true) : nothing

    for k in lags
        t0 = max(washout + 1, k + 1)
        idx = t0:T
        if length(idx) < 5
            show_progress && next!(prog)
            continue
        end

        Xk = Matrix(@view X[:, idx])
        yk = reshape(s[idx .- k], 1, :)

        itr, ite = _time_split_indices(size(Xk, 2); train_fraction=train_fraction)

        Xtr = Xk[:, itr]
        Ytr = yk[:, itr]
        Xte = Xk[:, ite]
        Yte = yk[:, ite]

        Wk, μk, σk = train_readout_from_features(Xtr, Ytr; ridgeλ=ridgeλ, washout=0)
        Yhat = apply_readout(Wk, Xte, μk, σk)
        mk = metrics(Yhat, Yte)
        r2[k] = max(0.0, mk.r2)  # standard MC clips negatives
        show_progress && next!(prog)
    end

    return (
        lags = lags,
        r2 = r2,
        mc = sum(skipmissing(filter(!isnan, r2))),
        maxlag = maxlag,
        input_dim = input_dim
    )
end

"""
    memory_capacity_curve_from_features(X, U; input_dim=1, maxlag=100,
                                        washout=100, ridgeλ=1e-6,
                                        train_fraction=0.7)

Linear recall of delayed input values from an already collected feature matrix.
All lags use the same valid time window and are fitted together, avoiding another
reservoir simulation and repeated matrix factorisations. As with
`memory_capacity_curve`, this is task-conditioned recall when `U` is structured
(for example Lorenz), rather than the input-independent capacity obtained with an
i.i.d. driver.
"""
function memory_capacity_curve_from_features(
    X::AbstractMatrix,
    U::AbstractMatrix;
    input_dim::Int=1,
    maxlag::Int=100,
    washout::Int=100,
    ridgeλ::Float64=1e-6,
    train_fraction::Float64=0.7,
)
    @assert size(X, 2) == size(U, 2) "X and U must contain the same time steps."
    @assert 1 <= input_dim <= size(U, 1)
    T = size(X, 2)
    maxlag = min(maxlag, T - washout - 2)
    @assert maxlag >= 1 "Not enough post-washout samples for a memory curve."

    idx = max(washout + 1, maxlag + 1):T
    Xvalid = Matrix(@view X[:, idx])
    s = vec(U[input_dim, :])
    Ylags = reduce(vcat, [reshape(s[idx .- k], 1, :) for k in 1:maxlag])
    itr, ite = _time_split_indices(size(Xvalid, 2); train_fraction=train_fraction)
    W, μ, σ = train_readout_from_features(Xvalid[:, itr], Ylags[:, itr];
        ridgeλ=ridgeλ, washout=0)
    Yhat = apply_readout(W, Xvalid[:, ite], μ, σ)
    r2 = [max(0.0, metrics(reshape(Yhat[k, :], 1, :),
        reshape(Ylags[k, ite], 1, :)).r2) for k in 1:maxlag]
    return (lags=collect(1:maxlag), r2=r2, mc=sum(r2), maxlag=maxlag,
        input_dim=input_dim)
end

"""
Convenience wrapper returning only total memory capacity.
"""
function memory_capacity(
    res::AbstractReservoir,
    U::AbstractMatrix;
    kwargs...
)
    mc = memory_capacity_curve(res, U; kwargs...)
    return mc.mc
end

"""
Perturbation-based stability / echo-state style diagnostic.

Procedure:
1. run the same input into two identical copies of the reservoir
2. after a warmup period, perturb one copy by a tiny amount ε
3. continue driving both with identical input
4. track the distance between their states

If log-distance decays approximately linearly with negative slope,
the reservoir is contracting under that drive.
"""
function perturbation_stability(
    res::AbstractReservoir,
    U::AbstractMatrix;
    ε::Float64=1e-8,
    warmup::Int=100,
    fit_start::Union{Nothing,Int}=nothing,
    rng=Random.default_rng(),
    obs=nothing,
    feature_fn::Function=feature_map,
    show_progress::Bool=true,
)
    T = size(U, 2)
    @assert warmup + 2 <= T "Need more time steps for perturbation stability test."

    # Create one reset state, then clone it. Resetting twice would give the two
    # replicas different initial conditions whenever reset! consumes randomness.
    r1 = clone_reservoir(res)
    rng1 = deepcopy(rng)
    reset!(r1; rng=rng1)
    r2 = clone_reservoir(r1)
    rng2 = deepcopy(rng1)

    prog = show_progress ? Progress(T; desc="perturbation_stability", showspeed=true) : nothing

    # warmup identically
    for t in 1:warmup
        u = view(U, :, t)
        reservoir_step!(r1, u; rng=rng1)
        reservoir_step!(r2, u; rng=rng2)
        show_progress && next!(prog)
    end

    # perturb second reservoir
    xref = state_vector(r2, obs)
    # Do not advance either drive RNG while constructing the perturbation: the
    # replicas must continue to receive identical stochastic forcing.
    ξ = randn(deepcopy(rng1), length(xref))
    ξ ./= (norm(ξ) + 1e-12)
    perturb_state!(r2, ε .* ξ)

    δ = zeros(Float64, T - warmup)
    times = collect(1:length(δ))

    for (j, t) in enumerate((warmup+1):T)
        u = view(U, :, t)
        reservoir_step!(r1, u; rng=rng1)
        reservoir_step!(r2, u; rng=rng2)

        δ[j] = state_distance(r1, r2, obs)
        show_progress && next!(prog)
    end

    # fit slope of log-distance over tail
    fs = fit_start === nothing ? min(10, length(δ)) : fit_start
    fs = clamp(fs, 1, length(δ))
    idx = fs:length(δ)

    y = log.(δ[idx] .+ 1e-18)
    x = collect(idx)
    x̄ = mean(x)
    ȳ = mean(y)
    slope = sum((x .- x̄) .* (y .- ȳ)) / (sum((x .- x̄).^2) + 1e-12)

    return (
        δ = δ,
        times = times,
        log_slope = slope,          # < 0 suggests contraction
        contraction_ratio = median(δ[2:end] ./ (δ[1:end-1] .+ 1e-18)),
        ε = ε,
        warmup = warmup
    )
end

"""
Separability across a set of input sequences.

Each sequence is pushed through the reservoir and represented by either:
- final feature vector
- or mean feature vector over the post-washout window

Useful later for swarms: do different predator/input histories induce
clearly different reservoir states?
"""
function sequence_separability(
    res::AbstractReservoir,
    Useqs::Vector{<:AbstractMatrix};
    washout::Int=0,
    summary::Symbol=:mean,   # :mean or :final
    rng=Random.default_rng(),
    obs=nothing,
    feature_fn::Function=feature_map,
    feature_scale::Union{Nothing,AbstractVector}=nothing,
    normalise_dimension::Bool=false,
    compute_summary::Bool=true,
    show_progress::Bool=true,
)
    M = length(Useqs)
    @assert M >= 2 "Need at least two sequences for separability."

    reps = Vector{Vector{Float64}}(undef, M)

    prog = show_progress ? Progress(M; desc="sequence_separability (per sequence)", showspeed=true) : nothing

    # Use the same reset stream for every input history so distances are caused
    # by the histories, not by unrelated random initial conditions.
    reset_rng = deepcopy(rng)
    for i in 1:M
        X, _ = collect_features(res, Useqs[i];
            rng=deepcopy(reset_rng), log_raw=false, reset_res=true,
            feature_fn=feature_fn, obs=obs, show_progress=false,
        )
        t0 = min(washout + 1, size(X, 2))

        reps[i] =
            summary == :mean  ? vec(mean(@view X[:, t0:end]; dims=2)) :
            summary == :final ? vec(X[:, end]) :
            error("summary must be :mean or :final")

        show_progress && next!(prog)
    end

    D = zeros(Float64, M, M)
    for i in 1:M, j in i+1:M
        Δ = reps[i] .- reps[j]
        feature_scale === nothing || (Δ = Δ ./ (feature_scale .+ 1e-12))
        dij = norm(Δ) / (normalise_dimension ? sqrt(length(Δ)) : 1.0)
        D[i, j] = dij
        D[j, i] = dij
    end

    Rsummary = compute_summary ? state_matrix_summary(hcat(reps...)) : nothing

    pairwise = [D[i, j] for i in 1:M-1 for j in i+1:M]

    return (
        representations = reps,
        pairwise_distances = D,
        mean_pairwise_distance = mean(pairwise),
        min_pairwise_distance = minimum(pairwise),
        feature_scaled = feature_scale !== nothing,
        dimension_normalised = normalise_dimension,
        representation_summary = Rsummary
    )
end

"""
    consistency_diagnostic(res, U; n_repeats=5, noise_std=0.0, washout=0, rng, obs=nothing)

Measure how similar the reservoir's response is across repeated presentations
of the same input `U` (Ny×T) -- the "repeated or noisy-input trials" test
called for in Project T1-S1's shared diagnostic harness.

With `noise_std=0.0`, each repeat replays `U` unchanged, so any divergence
between repeats comes only from the reservoir's own internal randomness
(e.g. Couzin's noise term or ESN initial-state jitter). With `noise_std>0`,
each repeat instead sees an independently perturbed copy `U .+ noise_std .* randn(...)`,
testing robustness to small input perturbations rather than pure replay
consistency.

Returns the mean pairwise feature-vector distance between repeats at each
post-washout timestep (`per_timestep`), and its mean over time
(`mean_consistency`; 0 = identical responses across repeats every repeat).
"""
function consistency_diagnostic(
    res::AbstractReservoir,
    U::AbstractMatrix;
    n_repeats::Int = 5,
    noise_std::Float64 = 0.0,
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    obs = nothing,
    feature_fn::Function = feature_map,
    show_progress::Bool = true,
)
    @assert n_repeats >= 2 "Need at least two repeats to measure consistency."
    Ny, T = size(U)
    @assert washout < T "washout must be < T"

    Xs = Vector{Matrix{Float64}}(undef, n_repeats)
    prog = show_progress ? Progress(n_repeats; desc="consistency_diagnostic (per repeat)", showspeed=true) : nothing
    for r in 1:n_repeats
        Ur = noise_std > 0 ? (U .+ noise_std .* randn(rng, Ny, T)) : U
        X, _ = collect_features(res, Ur; rng=rng, log_raw=false, reset_res=true,
                                feature_fn=feature_fn, obs=obs, show_progress=false)
        Xs[r] = X
        show_progress && next!(prog)
    end

    Nfeat = size(Xs[1], 1)
    per_t = zeros(Float64, T - washout)
    for (j, t) in enumerate((washout + 1):T)
        d = 0.0
        npairs = 0
        for i in 1:n_repeats, k in (i + 1):n_repeats
            d += norm(@view(Xs[i][:, t]) .- @view(Xs[k][:, t])) / sqrt(Nfeat)
            npairs += 1
        end
        per_t[j] = d / npairs
    end

    return (
        per_timestep = per_t,
        mean_consistency = mean(per_t),
        n_repeats = n_repeats,
        noise_std = noise_std,
        washout = washout,
    )
end

"""
Collect a useful bundle of diagnostics from already-computed feature matrices.
"""
function reservoir_feature_diagnostics(out)
    Xtrain = getproperty(out, :Xtrain)
    Xgen   = getproperty(out, :Xgen)

    diag_train = state_matrix_summary(Xtrain)
    diag_gen   = state_matrix_summary(Xgen)

    Wout = getproperty(out, :Wout)

    return (
        train = diag_train,
        freerun = diag_gen,
        readout = (
            frob_norm = norm(Wout),
            maxabs = maximum(abs.(Wout)),
            cond = try cond(Wout) catch; Inf end
        )
    )
end

# ============================================================
# Feature collection
# ============================================================

function collect_features(res::AbstractReservoir, U::AbstractMatrix;
    rng=Random.default_rng(),
    log_raw::Bool=true,
    reset_res::Bool=true,
    feature_fn::Function=feature_map,
    obs=nothing,
    debug_features::Bool=false,
    debug_every::Int=200,
    debug_firstk::Int=10,
    show_progress::Bool=true,
)
    Nu, T = size(U)
    reset_res && reset!(res; rng=rng)

    # Prime one step so feature dimension is well-defined
    reservoir_step!(res, view(U, :, 1); rng=rng)
    x1 = feature_fn(res, obs)
    Nfeat = length(x1)

    X = zeros(Float64, Nfeat, T)
    S = log_raw ? Vector{Any}(undef, T) : nothing

    # record t=1 (already stepped)
    X[:, 1] .= x1
    log_raw && (S[1] = raw_state(res))

    prog = show_progress ? Progress(T - 1; desc="collect_features", showspeed=true) : nothing

    for t in 2:T
        reservoir_step!(res, view(U, :, t); rng=rng)
        x = feature_fn(res, obs)
        X[:, t] .= x
        log_raw && (S[t] = raw_state(res))

        if debug_features && (t % debug_every == 0 || t == T)
            xmax  = maximum(abs.(x))
            xnorm = norm(x)
            xmean = mean(x)
            xstd  = std(x)
            idx   = argmax(abs.(x))
            println("t=$t  ‖x‖=$xnorm  max|x|=$xmax at idx=$idx  mean=$xmean std=$xstd")
            println("  first $(debug_firstk) feats: ", x[1:min(debug_firstk, length(x))])
        end

        show_progress && next!(prog)
    end

    return X, S
end

# kept as a named wrapper since you use it elsewhere
function compute_features_teacher_forced(res::AbstractReservoir, U::AbstractMatrix;
    rng=Random.default_rng(),
    reset_res::Bool=true,
    log_raw::Bool=true,
    feature_fn::Function=feature_map,
    obs=nothing,
    show_progress::Bool=true,
)
    return collect_features(res, U; rng=rng, log_raw=log_raw, reset_res=reset_res,
                            feature_fn=feature_fn, obs=obs, show_progress=show_progress)
end

"""
train_readout_from_features(X, Y; washout, ridgeλ)

X: Nfeat×T (raw features)
Y: Ny×T   (targets)

Returns Wout and normalisation stats.
"""
function train_readout_from_features(X::AbstractMatrix, Y::AbstractMatrix;
    ridgeλ::Float64=1e-3,
    washout::Int=0
)
    @assert size(X, 2) == size(Y, 2)

    if washout > 0
        X2 = copy(@view X[:, (washout+1):end])
        Y2 = @view Y[:, (washout+1):end]
    else
        X2 = copy(X)
        Y2 = Y
    end

    μx, σx = feature_stats(X2)
    standardise!(X2, μx, σx)

    Φ = augment_with_bias(X2)
    Wout = fit_ridge(Φ, Y2, RidgeReadout(ridgeλ))

    return Wout, μx, σx
end

"""
    train_readout_cv(X, Y; ridge_grid=10.0 .^ (-3:1:4),
                     n_folds=5, gap=0, washout=0)

Select the ridge penalty by blocked cross-validation without shuffling the
time series. `gap` excludes samples adjacent to each validation block.
The selected model is refitted to all post-washout training data.

Returns `(Wout, μx, σx, best_λ, val_scores)`; validation scores are mean
fold MSE values.
"""
function train_readout_cv(X::AbstractMatrix, Y::AbstractMatrix;
    ridge_grid = 10.0 .^ (-3:1:4),
    n_folds::Int = 5,
    gap::Int = 0,
    washout::Int = 0,
)
    @assert size(X, 2) == size(Y, 2)
    @assert n_folds >= 2
    @assert gap >= 0

    if washout > 0
        X = @view X[:, (washout+1):end]
        Y = @view Y[:, (washout+1):end]
    end

    T = size(X, 2)
    edges = round.(Int, range(0, T; length = n_folds + 1))
    fold_ranges = [(edges[i] + 1):edges[i+1] for i in 1:n_folds]
    @assert all(!isempty, fold_ranges) "Not enough data for $n_folds folds in train_readout_cv."

    val_scores = Tuple{Float64,Float64}[]
    for λ in ridge_grid
        fold_mses = Float64[]
        for val_range in fold_ranges
            excluded = max(1, first(val_range) - gap):min(T, last(val_range) + gap)
            tr_idx = setdiff(1:T, excluded)
            Xsub, Ysub = X[:, tr_idx], Y[:, tr_idx]
            Xval, Yval = X[:, val_range], Y[:, val_range]
            Wout, μx, σx = train_readout_from_features(Xsub, Ysub; ridgeλ = Float64(λ), washout = 0)
            Yhat = apply_readout(Wout, Xval, μx, σx)
            push!(fold_mses, mean((Yhat .- Yval) .^ 2))
        end
        push!(val_scores, (Float64(λ), mean(fold_mses)))
    end

    best_λ = val_scores[argmin(last.(val_scores))][1]
    Wout, μx, σx = train_readout_from_features(X, Y; ridgeλ = best_λ, washout = 0)

    return Wout, μx, σx, best_λ, val_scores
end

# ============================================================
# Training / prediction API
# ============================================================

function train_readout(res::AbstractReservoir, U::AbstractMatrix, Y::AbstractMatrix;
    ridgeλ=1e-3,
    washout::Int=0,
    rng=Random.default_rng(),
    log_raw::Bool=true,
    obs=nothing,
    feature_fn::Function=feature_map,
    show_progress::Bool=true,
)
    @assert size(U, 2) == size(Y, 2)

    X, S = collect_features(res, U; rng=rng, log_raw=log_raw, reset_res=true,
                            feature_fn=feature_fn, obs=obs, show_progress=show_progress)

    # train stats on post-washout window
    if washout > 0
        X2 = copy(@view X[:, (washout+1):end])
        Y2 = @view Y[:, (washout+1):end]
    else
        X2 = copy(X)
        Y2 = Y
    end
    μx, σx = feature_stats(X2)
    standardise!(X2, μx, σx)

    Φ = augment_with_bias(X2)
    Wout = fit_ridge(Φ, Y2, RidgeReadout(ridgeλ))

    return Wout, X, S, μx, σx
end

function predict_teacher_forced(res::AbstractReservoir, Wout::AbstractMatrix, U::AbstractMatrix,
    μx::AbstractVector, σx::AbstractVector;
    rng=Random.default_rng(),
    log_raw::Bool=true,
    reset_res::Bool=true,
    obs=nothing,
    feature_fn::Function=feature_map,
    show_progress::Bool=true,
)
    X, S = collect_features(res, U; rng=rng, log_raw=log_raw, reset_res=reset_res,
                            feature_fn=feature_fn, obs=obs, show_progress=show_progress)

    Yhat = apply_readout(Wout, X, μx, σx)
    return Yhat, X, S
end

function predict_free_run(res::AbstractReservoir, Wout::AbstractMatrix, u0::AbstractVector, T::Int,
    μx::AbstractVector, σx::AbstractVector;
    input_from_output::Union{Nothing,Function} = nothing,
    rng=Random.default_rng(),
    log_raw::Bool=true,
    reset_res::Bool=true,
    obs=nothing,
    feature_fn::Function=feature_map,
    show_progress::Bool=true,
)
    reset_res && reset!(res; rng=rng)

    Nu = length(u0)
    Ny = size(Wout, 1)
    if input_from_output === nothing
        Ny == Nu || error(
            "Autonomous free run cannot feed a $Ny-dimensional readout back into a " *
            "$Nu-dimensional reservoir input. Predict the full next input/embedding " *
            "vector or provide an explicit input_from_output adapter.")
        input_from_output = y -> y
    end

    u = copy(u0)

    # one step to establish feature length + start state
    reservoir_step!(res, u; rng=rng)
    x = feature_fn(res, obs)
    Nfeat = length(x)

    Ugen = zeros(Float64, Nu, T)
    Ygen = zeros(Float64, Ny, T)
    Xgen = zeros(Float64, Nfeat, T)
    S    = log_raw ? Vector{Any}(undef, T) : nothing

    # t = 1 (already stepped with u0)
    Ugen[:, 1] .= u
    Xgen[:, 1] .= x
    y = apply_readout(Wout, x, μx, σx)
    Ygen[:, 1] .= y
    log_raw && (S[1] = raw_state(res))

    u = vec(input_from_output(y))
    @assert length(u) == Nu

    prog = show_progress ? Progress(T - 1; desc="predict_free_run", showspeed=true) : nothing

    for t in 2:T
        Ugen[:, t] .= u
        reservoir_step!(res, u; rng=rng)

        x = feature_fn(res, obs)
        Xgen[:, t] .= x

        y = apply_readout(Wout, x, μx, σx)
        Ygen[:, t] .= y

        log_raw && (S[t] = raw_state(res))

        u = vec(input_from_output(y))
        show_progress && next!(prog)
        @assert length(u) == Nu
    end

    return Ugen, Ygen, Xgen, S
end

function train_and_evaluate_reservoir(res::AbstractReservoir;
    data::Union{Nothing,AbstractMatrix}=nothing,
    data_fn::Union{Nothing,Function}=nothing,
    shift::Int=300,
    train_len::Int=5000,
    predict_len::Int=1250,
    washout::Int=200,
    ridgeλ::Float64=1e-3,
    seed::Int=1,
    rng::AbstractRNG=MersenneTwister(seed),
    log_raw::Bool=true,
    labels::Union{Nothing,Vector{String}}=nothing,
    dt_data::Union{Nothing,Float64}=nothing,
    build_observation_layer!::Function = build_observation_layer!,
    obs_kwargs::NamedTuple = NamedTuple(),

    compute_reservoir_diagnostics::Bool = true,
    memory_maxlag::Int = 100,
    memory_input_dim::Int = 1,
    stability_ε::Float64 = 1e-8,
    separability_sequences::Union{Nothing,Vector{<:AbstractMatrix}} = nothing,

    show_progress::Bool = true
)
    # Number of coarse stages
    nsteps = 8
    if compute_reservoir_diagnostics
        nsteps += 3
        if separability_sequences !== nothing
            nsteps += 1
        end
    end

    p = show_progress ? Progress(
        nsteps;
        desc="train_and_evaluate_reservoir",
        showspeed=true,
        barlen=30
    ) : nothing

    # Set current stage label without incrementing progress
    function _stage!(msg)
        if show_progress
            ProgressMeter.update!(p, p.counter; showvalues=[(:stage, msg)])  # qualified: CairoMakie also exports `update!`
        end
        return nothing
    end

    # Mark one stage complete and optionally update displayed stage label
    function _done!(msg="")
        if show_progress
            next!(p; showvalues=[(:stage, msg)])
        end
        return nothing
    end

    meta = NamedTuple()

    # ------------------------------------------------------------
    # 1) Load / generate data
    # ------------------------------------------------------------
    _stage!("load data")
    if data === nothing
        @assert data_fn !== nothing "Provide either `data` (Ny×T) or `data_fn(rng)`."
        outd = data_fn(rng)
        if outd isa NamedTuple
            @assert hasproperty(outd, :data)
            data = outd.data
            meta = Base.structdiff(outd, (data=outd.data,))
        elseif outd isa Tuple && length(outd) == 2
            data, meta = outd
        else
            error("data_fn must return NamedTuple with field `data`, or a (data, meta) tuple.")
        end
    end
    @assert ndims(data) == 2 "data must be a matrix Ny×T."

    if labels === nothing && hasproperty(meta, :labels)
        labels = meta.labels
    end
    if dt_data === nothing && hasproperty(meta, :dt_data)
        dt_data = meta.dt_data
    end
    _done!("data ready")

    # ------------------------------------------------------------
    # 2) Slice data
    # ------------------------------------------------------------
    _stage!("slice data")
    Utrain, Ytrain, test_seed, Ytest =
        slice_data(data=data; shift=shift, train_len=train_len, predict_len=predict_len)
    _done!("slice complete")

    # ------------------------------------------------------------
    # 3) Observation layer
    # ------------------------------------------------------------
    _stage!("build observation layer")
    K = build_observation_layer!(res, Utrain; washout=washout, rng=rng, show_progress=false, obs_kwargs...)
    _done!("observation layer ready")

    # ------------------------------------------------------------
    # 4) Teacher-forced features
    # ------------------------------------------------------------
    _stage!("teacher-forced features")
    Xtrain, state = compute_features_teacher_forced(res, Utrain;
        rng=rng, reset_res=true, log_raw=log_raw,
        feature_fn=feature_map, obs=K, show_progress=false)
    _done!("teacher-forced features ready")

    # ------------------------------------------------------------
    # 5) Train readout
    # ------------------------------------------------------------
    _stage!("train readout")
    Wout, μx, σx = train_readout_from_features(Xtrain, Ytrain;
        ridgeλ=ridgeλ, washout=washout)
    _done!("readout trained")

    # ------------------------------------------------------------
    # 6) Teacher-forced predictions
    # ------------------------------------------------------------
    _stage!("teacher-forced prediction")
    Yhat_tf = apply_readout(Wout, Xtrain, μx, σx)
    _done!("teacher-forced prediction ready")

    # ------------------------------------------------------------
    # 7) Free-run rollout
    # ------------------------------------------------------------
    _stage!("free-run rollout")
    Ugen, Ygen, Xgen, Sgen =
        predict_free_run(res, Wout, collect(test_seed), predict_len, μx, σx;
            input_from_output = nothing,
            rng=rng, log_raw=log_raw, reset_res=true,
            feature_fn=feature_map, obs=K, show_progress=false)
    _done!("free-run rollout ready")

    # ------------------------------------------------------------
    # 8) Metrics
    # ------------------------------------------------------------
    _stage!("metrics")
    tf_start = min(washout + 1, size(Yhat_tf, 2))
    m_tf = metrics(@view(Yhat_tf[:, tf_start:end]),
                   @view(Ytrain[:,  tf_start:end]))
    m_fr = metrics(Ygen, Ytest)

    Ny = size(data, 1)
    labels === nothing && (labels = ["y$(i)(t)" for i in 1:Ny])
    dt_data === nothing && (dt_data = 1.0)
    _done!("metrics ready")

    # ------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------
    reservoir_diag = nothing
    memory_diag = nothing
    stability_diag = nothing
    separability_diag = nothing

    if compute_reservoir_diagnostics
        _stage!("feature diagnostics")
        reservoir_diag = (
            features = reservoir_feature_diagnostics((
                Xtrain = Xtrain,
                Xgen   = Xgen,
                Wout   = Wout
            )),
        )
        _done!("feature diagnostics ready")

        _stage!("memory capacity")
        memory_diag = memory_capacity_curve(res, Matrix(Utrain);
            input_dim = memory_input_dim,
            maxlag    = memory_maxlag,
            washout   = washout,
            ridgeλ    = ridgeλ,
            rng       = rng,
            obs       = K,
            feature_fn = feature_map,
            show_progress = show_progress,   # this stage is the one most likely to actually be slow
        )
        _done!("memory capacity ready")

        _stage!("stability")
        stability_diag = perturbation_stability(res, Matrix(Utrain);
            ε         = stability_ε,
            warmup    = washout,
            rng       = rng,
            obs       = K,
            feature_fn = feature_map,
            show_progress = show_progress,
        )
        _done!("stability ready")

        if separability_sequences !== nothing
            _stage!("separability")
            separability_diag = sequence_separability(res, separability_sequences;
                washout    = washout,
                summary    = :mean,
                rng        = rng,
                obs        = K,
                feature_fn = feature_map,
                show_progress = show_progress,
            )
            _done!("separability ready")
        end
    end

    show_progress && finish!(p)

    return merge(meta, (
        rng = rng,
        seed = seed,
        dt_data = dt_data,
        shift = shift,
        train_len = train_len,
        predict_len = predict_len,
        washout = washout,
        ridgeλ = ridgeλ,
        labels = labels,

        data = Matrix(data),
        Utrain = Matrix(Utrain),
        Ytrain = Matrix(Ytrain),
        test_seed = Vector(test_seed),
        Ytest = Matrix(Ytest),

        res = res,
        K = K,
        Wout = Wout,
        μx = μx,
        σx = σx,

        Xtrain = Xtrain,
        state = state,

        Yhat_tf = Yhat_tf,
        Ugen = Ugen,
        Ygen = Ygen,
        Xgen = Xgen,
        Sgen = Sgen,

        m_tf = m_tf,
        m_fr = m_fr,

        reservoir_diag = reservoir_diag,
        memory_diag = memory_diag,
        stability_diag = stability_diag,
        separability_diag = separability_diag
    ))
end

# ============================================================
# my_consistency.jl
# Consistency capacity (Lymburn et al., Chaos 2021, Appendix A) -- a more
# rigorous alternative to the simpler `consistency_diagnostic` in
# my_reservoir_core.jl.
# ============================================================
#
# Two estimators are implemented:
#
#   consistency_capacity            generic over any AbstractReservoir +
#                                    feature_fn (Eqs A1-A4) -- applies to
#                                    BasicESN, kernel-observation-layer
#                                    swarm features, or raw swarm positions
#                                    alike.
#
#   consistency_capacity_symmetric  specialised for raw-position features
#                                    of N identical, permutation-symmetric
#                                    D-dimensional agents (Appendix A2,
#                                    Eq A5) -- NOT applicable to BasicESN or
#                                    to kernel features, neither of which
#                                    has this symmetry.
#
# The paper is explicit that the *naive* (raw-position) case needs the
# symmetric estimator: "The profile is calculated with a method which
# utilizes the permutation symmetry (App. A2) to reduce finite size
# effects" (main text, describing Fig 2b). Plain replica averaging on raw
# positions is exactly the regime Appendix A2 warns is badly
# noise-dominated at finite sample sizes -- most of the O(N^2) entries of
# C_xx/C_ss are theoretically ~0 (individual agents aren't intrinsically
# consistent, only the centre of mass is), but empirically noisy, and
# whitening amplifies that noise into spurious "consistency". The
# symmetric estimator collapses those down to a handful of
# symmetry-class averages (Eq A5) instead, which is a much lower-variance
# estimate of the same theoretical quantity.
#
# Both estimators also follow Appendix A's requirement that replica
# initial conditions be *close* ("chosen to be close, in order to make
# sure the two trajectories are on the same attractor, in case there is
# more than one") rather than independently random -- see
# `_close_replicas` below.

using LinearAlgebra
using Statistics
using Random
using ProgressMeter

# ------------------------------------------------------------
# Shared: close (not independent) replica initial conditions
# ------------------------------------------------------------

"""
    _close_replicas(res, n_repeats, jitter_scale, rng)

Builds `n_repeats` independent reservoir copies, all started from the same
base state (`res`'s current state) plus a small random perturbation --
"close" initial conditions, per App. A, rather than independent draws from
`reset!`. `jitter_scale` is relative to the RMS magnitude of the base
state vector (`state_vector`), so it's meaningful across reservoir types
with different natural units (ESN activations vs. swarm positions). Falls
back to an absolute floor (`jitter_scale` itself) if the base state has
~zero magnitude (e.g. `BasicESN`'s `reset!` always zeros its state, so a
purely-relative jitter would silently vanish to nothing there).
"""
function _close_replicas(res::AbstractReservoir, n_repeats::Int, jitter_scale::Float64, rng::AbstractRNG)
    base_vec = state_vector(res)
    rms = norm(base_vec) / sqrt(length(base_vec))
    scale = rms > 1e-8 ? jitter_scale * rms : jitter_scale

    reps = Vector{typeof(res)}(undef, n_repeats)
    for r in 1:n_repeats
        rep = clone_reservoir(res)
        ξ = scale .* randn(rng, length(base_vec))
        perturb_state!(rep, ξ)
        reps[r] = rep
    end
    return reps
end

# ------------------------------------------------------------
# Generic estimator (Eqs A1-A4)
# ------------------------------------------------------------

"""
    consistency_capacity(res, U; n_repeats=5, washout=0, rng, obs=nothing, feature_fn=feature_map,
                          jitter_scale=0.01, reg=1e-9)

Reproduces Lymburn et al. (2021), Appendix A (Eqs A1-A4), generic over any
`AbstractReservoir`/`feature_fn`. Returns a NamedTuple:
- `gamma2`: the consistency spectrum {gamma_k^2}, sorted descending
- `Theta`: the consistency capacity, `sum(gamma2) == Tr(C_ss)`
- `n_repeats`, `washout`, `Nfeat`

`res`'s *current* state is used as the shared base for `n_repeats` close
replicas (see `_close_replicas`) -- call `reset!(res; rng=...)` yourself
first if you want a fresh/random base state. Each replica then drives
through the *same* `U`, teacher-forced, collecting features via
`feature_fn`/`obs` exactly as `collect_features` does elsewhere in this
file. `reg` is added to `C_xx`'s diagonal before whitening (App. A: 1e-9).

For raw permutation-symmetric agent positions specifically (not kernel
features, not ESN), `consistency_capacity_symmetric` below is a much
lower-variance estimator of the same quantity -- see its docstring.
"""
function consistency_capacity(
    res::AbstractReservoir,
    U::AbstractMatrix;
    n_repeats::Int = 5,
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    obs = nothing,
    feature_fn::Function = feature_map,
    jitter_scale::Float64 = 0.01,
    reg::Float64 = 1e-9,
    show_progress::Bool = true,
)
    @assert n_repeats >= 2 "Need at least 2 replicas for the consistency (replica) test."

    reps = _close_replicas(res, n_repeats, jitter_scale, rng)

    prog = show_progress ? Progress(n_repeats; desc = "consistency_capacity (per replica)", showspeed = true) : nothing

    Xs = Vector{Matrix{Float64}}(undef, n_repeats)
    for r in 1:n_repeats
        X, _ = collect_features(reps[r], U;
            rng = rng, log_raw = false, reset_res = false,
            feature_fn = feature_fn, obs = obs, show_progress = false,
        )
        Xs[r] = Matrix(X[:, (washout + 1):end])
        show_progress && next!(prog)
    end

    Nfeat, Tuse = size(Xs[1])

    # Ensemble mean over time and replicas -> centre each replica (App. A
    # assumes zero-mean signal/noise decomposition).
    pooled_mean = zeros(Nfeat)
    for r in 1:n_repeats
        pooled_mean .+= vec(mean(Xs[r]; dims = 2))
    end
    pooled_mean ./= n_repeats
    for r in 1:n_repeats
        Xs[r] .-= pooled_mean
    end

    # Pooled covariance C_xx (Eq A2), averaged over replicas and time.
    Cxx = zeros(Nfeat, Nfeat)
    for r in 1:n_repeats
        Cxx .+= (Xs[r] * Xs[r]') ./ Tuse
    end
    Cxx ./= n_repeats

    # Whitening transform T = Q Sigma^{-1} Q^T where C_xx = Q Sigma^2 Q^T,
    # per App. A's own prescription: add a small regularisation to C_xx
    # before whitening.
    Cxx_reg = Cxx + reg * I(Nfeat)
    F = eigen(Symmetric(Cxx_reg))
    λ = F.values
    Tmat = F.vectors * Diagonal(1.0 ./ sqrt.(λ))

    Xo = [Tmat' * Xs[r] for r in 1:n_repeats]

    # Cross-covariance of the (whitened) signal component, averaged over
    # all ordered replica pairs (Eq A3; reduces to the paper's exact
    # 2-replica formula when n_repeats == 2).
    Css = zeros(Nfeat, Nfeat)
    npairs = 0
    for i in 1:n_repeats, j in 1:n_repeats
        i == j && continue
        Css .+= (Xo[i] * Xo[j]') ./ Tuse
        npairs += 1
    end
    Css ./= npairs
    Css = Matrix(Symmetric((Css + Css') / 2))

    Fss = eigen(Symmetric(Css))
    γ2 = clamp.(Fss.values, 0.0, 1.0)   # consistency spectrum {γ_k^2}, Eq A4 (bounded in [0,1] up to numerical noise)
    Θ = sum(γ2)                          # consistency capacity, Eq A4

    return (
        gamma2 = sort(γ2; rev = true),
        Theta = Θ,
        n_repeats = n_repeats,
        washout = washout,
        Nfeat = Nfeat,
    )
end

# ------------------------------------------------------------
# Symmetric estimator (Appendix A2, Eq A5)
# ------------------------------------------------------------

"""
    _symmetrize_by_class(C, N, D; same_agent_distinct)

Replaces each entry of `C` (an `N*D x N*D` matrix, agent-major/interleaved
order -- index `D*(i-1)+d` for agent `i`, dimension `d`) by the empirical
average over its permutation-symmetry class: entries are grouped by
(unordered) dimension pair `(d_p, d_q)`, and -- if `same_agent_distinct`
-- further split into "same agent" (`i_p == i_q`) vs. "different agent"
classes. This is Eq A5's block structure (`C_xx = A ⊗ I_N + B ⊗ 1_N` has a
same-agent class per dimension pair for the `A` term plus the shared
`B` term; `C_ss = H ⊗ 1_N` collapses same/different-agent into one class
per dimension pair, since the signal component is identical across
agents).
"""
function _symmetrize_by_class(C::AbstractMatrix, N::Int, D::Int; same_agent_distinct::Bool)
    Nfeat = N * D
    @assert size(C) == (Nfeat, Nfeat)

    agent_of(p) = (p - 1) ÷ D + 1
    dim_of(p) = (p - 1) % D + 1

    sums = Dict{Tuple,Float64}()
    counts = Dict{Tuple,Int}()

    @inbounds for p in 1:Nfeat, q in 1:Nfeat
        dp, dq = dim_of(p), dim_of(q)
        dim_key = dp <= dq ? (dp, dq) : (dq, dp)
        key = same_agent_distinct ? (agent_of(p) == agent_of(q), dim_key) : (dim_key,)
        sums[key] = get(sums, key, 0.0) + C[p, q]
        counts[key] = get(counts, key, 0) + 1
    end

    means = Dict(k => sums[k] / counts[k] for k in keys(sums))

    Csym = similar(C)
    @inbounds for p in 1:Nfeat, q in 1:Nfeat
        dp, dq = dim_of(p), dim_of(q)
        dim_key = dp <= dq ? (dp, dq) : (dq, dp)
        key = same_agent_distinct ? (agent_of(p) == agent_of(q), dim_key) : (dim_key,)
        Csym[p, q] = means[key]
    end
    return Csym
end

"""
    consistency_capacity_symmetric(res, U, N, D; n_repeats=5, washout=0, rng,
                                    jitter_scale=0.01, reg=1e-9, show_progress=true)

Appendix A2's permutation-symmetry-exploiting consistency-capacity
estimator, for **raw-position features of `N` identical, permutation-
symmetric `D`-dimensional agents** (e.g. `feature_map(res)` for a
`CouzinReservoir`/`LymburnReservoir` with no observation layer -- the
"naive" case in the paper's Figs 2/5). **Not applicable** to `BasicESN`
(units aren't interchangeable) or to kernel-observation-layer features
(kernels sit at fixed, non-interchangeable positions) -- use the generic
`consistency_capacity` for those.

Assumes agent-major/interleaved feature order: index `D*(i-1)+d` for agent
`i` (1..N), dimension `d` (1..D) -- matching `vec(raw_state(res).P)`.

Rather than estimating the full `Nfeat x Nfeat` covariance matrices
empirically (which is what `consistency_capacity` does, and which the
paper notes is badly noise-dominated for exactly this symmetric case, see
this file's header comment), entries are grouped into symmetry classes and
replaced by their class average (`_symmetrize_by_class`, Eq A5) both for
`C_xx` (before whitening) and for the cross-replica covariance (after
whitening). In the idealised/infinite-time limit this makes `C_ss` exactly
rank `D` (one consistent direction per spatial dimension, aligned with the
centre of mass) -- so with finite data, `Theta` should land much closer to
`D` than the unstructured `consistency_capacity` estimator gives.
"""
function consistency_capacity_symmetric(
    res::AbstractReservoir,
    U::AbstractMatrix,
    N::Int,
    D::Int;
    n_repeats::Int = 5,
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    jitter_scale::Float64 = 0.01,
    reg::Float64 = 1e-9,
    show_progress::Bool = true,
)
    @assert n_repeats >= 2 "Need at least 2 replicas for the consistency (replica) test."
    Nfeat = N * D

    reps = _close_replicas(res, n_repeats, jitter_scale, rng)

    prog = show_progress ? Progress(n_repeats; desc = "consistency_capacity_symmetric (per replica)", showspeed = true) : nothing

    Xs = Vector{Matrix{Float64}}(undef, n_repeats)
    for r in 1:n_repeats
        X, _ = collect_features(reps[r], U;
            rng = rng, log_raw = false, reset_res = false,
            feature_fn = feature_map, obs = nothing, show_progress = false,
        )
        Xs[r] = Matrix(X[:, (washout + 1):end])
        show_progress && next!(prog)
    end

    @assert size(Xs[1], 1) == Nfeat "feature dimension $(size(Xs[1], 1)) does not match N*D=$Nfeat -- wrong (N, D), or this reservoir's naive feature_map isn't raw agent-major positions."
    Tuse = size(Xs[1], 2)

    pooled_mean = zeros(Nfeat)
    for r in 1:n_repeats
        pooled_mean .+= vec(mean(Xs[r]; dims = 2))
    end
    pooled_mean ./= n_repeats
    for r in 1:n_repeats
        Xs[r] .-= pooled_mean
    end

    Cxx = zeros(Nfeat, Nfeat)
    for r in 1:n_repeats
        Cxx .+= (Xs[r] * Xs[r]') ./ Tuse
    end
    Cxx ./= n_repeats
    Cxx_sym = _symmetrize_by_class(Cxx, N, D; same_agent_distinct = true)

    Cxx_reg = Cxx_sym + reg * I(Nfeat)
    F = eigen(Symmetric(Cxx_reg))
    λ = F.values
    Tmat = F.vectors * Diagonal(1.0 ./ sqrt.(λ))

    Xo = [Tmat' * Xs[r] for r in 1:n_repeats]

    Css = zeros(Nfeat, Nfeat)
    npairs = 0
    for i in 1:n_repeats, j in 1:n_repeats
        i == j && continue
        Css .+= (Xo[i] * Xo[j]') ./ Tuse
        npairs += 1
    end
    Css ./= npairs
    Css_sym = _symmetrize_by_class(Css, N, D; same_agent_distinct = false)
    Css_sym = Matrix(Symmetric((Css_sym + Css_sym') / 2))

    Fss = eigen(Symmetric(Css_sym))
    γ2 = clamp.(Fss.values, 0.0, 1.0)
    Θ = sum(γ2)

    order = sortperm(γ2; rev = true)

    return (
        gamma2 = γ2[order],
        Theta = Θ,
        n_repeats = n_repeats,
        washout = washout,
        Nfeat = Nfeat,
        N = N,
        D = D,
        eigvecs = Fss.vectors[:, order],
        whitening_transform = Tmat,
        pooled_mean = pooled_mean,
    )
end

"""
    project_onto_consistent_components(Xfeat, cc_result; k=2)

Project a raw agent-position feature matrix (`Nfeat x T`, agent-major,
built the same way as whatever `Xfeat` was used to compute `cc_result`
via `consistency_capacity_symmetric`) onto the top `k` "consistent"
directions found by that call, and separately compute the raw centre-of-
mass trajectory for the same data.

Reproduces Lymburn et al.'s Fig 2(c): overlaying the two demonstrates that
the "consistent" directions found by the Appendix A2 permutation-symmetry
argument are, up to the whitening transform, the centre-of-mass
directions -- exactly what Eq A5's `C_ss = H⊗1_N` predicts (`1_N`'s only
nonzero eigenvector is the all-ones/CoM-averaging direction). Both
outputs are returned standardised (zero mean, unit variance per row/
dimension) since the projection lives in a whitened basis with no reason
to share the CoM's own physical units or scale -- comparing *shape*
(and correlation) is the point, not raw magnitude.

Returns `(projection = k×T matrix, com = D×T matrix)`.
"""
function project_onto_consistent_components(Xfeat::AbstractMatrix, cc_result; k::Int = 2)
    N, D = cc_result.N, cc_result.D
    @assert size(Xfeat, 1) == N * D "Xfeat's feature dimension doesn't match cc_result's N*D."
    @assert k <= size(cc_result.eigvecs, 2) "k=$k exceeds the number of available eigenvectors."

    Xc = Xfeat .- cc_result.pooled_mean
    Xw = cc_result.whitening_transform' * Xc
    proj = cc_result.eigvecs[:, 1:k]' * Xw

    T = size(Xfeat, 2)
    com = zeros(D, T)
    @inbounds for d in 1:D
        idx = d:D:(N*D)
        com[d, :] = vec(mean(view(Xfeat, idx, :); dims = 1))
    end

    _zscore_rows!(M) = (M .- mean(M; dims = 2)) ./ (std(M; dims = 2) .+ eps())
    return (projection = _zscore_rows!(proj), com = _zscore_rows!(com))
end

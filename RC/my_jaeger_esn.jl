# my_jaeger_esn.jl
#
# A faithful reproduction of the specific leaky-integrator, output-feedback
# echo state network Jaeger (2001, Section 6, "The 'echo state' approach to
# analysing and training recurrent neural networks", GMD Report 148) uses
# for the Mackey-Glass chaotic-attractor learning task.
#
# Deliberately NOT built on the AbstractReservoir/BasicESN interface used
# elsewhere in this codebase (RC/my_esn.jl, RC/my_reservoir_core.jl):
# Jaeger's architecture here has two structural features that interface
# doesn't support -- independent leak/decay constants (C and a, not a
# single `leak`), and output feedback (Wback*y(n) driving the state update,
# rather than an externally supplied input). Mirrors how the Lymburn swarm
# reservoir (SWARM_RC/) is also its own thing built on the shared low-level
# utilities (fit_ridge-style regression, metrics), not the generic
# AbstractReservoir pipeline.

using Random, LinearAlgebra, Statistics

mutable struct JaegerFeedbackESN
    Nh    :: Int
    Ny    :: Int
    x     :: Vector{Float64}
    Win   :: Vector{Float64}   # bias-input weights, length Nh (u(n) is a scalar constant)
    W     :: Matrix{Float64}   # recurrent weights, Nh x Nh
    Wback :: Matrix{Float64}   # output-feedback weights, Nh x Ny
    C     :: Float64
    a     :: Float64
    delta :: Float64
end

"""
    JaegerFeedbackESN(Ny=1; Nh=400, raw_spectral_radius=0.79, C=0.44, a=0.9,
                       delta=1.0, feedback_scale=0.56, rng=...)

Builds the network from Jaeger (2001) Section 6.2: `W` rescaled so
`|λmax(W)|≈0.79`; leaky update with stepsize `delta=1`, time constant
`C=0.44`, decay rate `a=0.9` (giving an effective spectral radius
`|λmax(W̃)|≈0.95` per his Prop. 4(b) -- not enforced here directly, it
falls out of these three numbers); bias-input weights drawn from
`{0, 0.14, -0.14}` with probabilities `{0.5, 0.25, 0.25}`; output-feedback
weights uniform on `[-0.56, 0.56]`.
"""
function JaegerFeedbackESN(
    Ny::Int = 1;
    Nh::Int = 400,
    raw_spectral_radius::Float64 = 0.79,
    C::Float64 = 0.44,
    a::Float64 = 0.9,
    delta::Float64 = 1.0,
    bias_weight_options::NTuple{3,Float64} = (0.0, 0.14, -0.14),
    bias_weight_probs::NTuple{3,Float64} = (0.5, 0.25, 0.25),
    feedback_scale::Float64 = 0.56,
    rng::AbstractRNG = Random.default_rng(),
)
    x = zeros(Float64, Nh)

    cum = cumsum(collect(bias_weight_probs))
    Win = Float64[bias_weight_options[searchsortedfirst(cum, rand(rng))] for _ in 1:Nh]

    W = 2 .* rand(rng, Nh, Nh) .- 1
    v = rand(rng, Nh)
    for _ in 1:100
        v = W * v
        v ./= (norm(v) + 1e-12)
    end
    ρ = norm(W * v) + 1e-12
    W .*= (raw_spectral_radius / ρ)

    Wback = feedback_scale .* (2 .* rand(rng, Nh, Ny) .- 1)

    return JaegerFeedbackESN(Nh, Ny, x, Win, W, Wback, C, a, delta)
end

reset!(res::JaegerFeedbackESN) = (fill!(res.x, 0.0); res)

"""
    step!(res, y; u=0.2, noise_scale=0.0, rng=...)

One update of Jaeger's eqn (24):

    x(n+1) = (1-δCa) x(n) + δC · tanh(Win·u + W·x(n) + Wback·y + ν(n))

`y` is the (teacher-forced or fed-back) output driving `Wback`; `ν(n)` is
drawn uniformly from `[-noise_scale, noise_scale]^Nh` (Jaeger's training-time
state noise -- pass `noise_scale=0` for generation/testing, matching his
protocol of noise during training only).
"""
function step!(res::JaegerFeedbackESN, y::AbstractVector;
    u::Float64 = 0.2, noise_scale::Float64 = 0.0, rng::AbstractRNG = Random.default_rng(),
)
    pre = res.Win .* u .+ res.W * res.x .+ res.Wback * y
    if noise_scale > 0
        pre = pre .+ noise_scale .* (2 .* rand(rng, res.Nh) .- 1)
    end
    xnew = tanh.(pre)
    decay = 1 - res.delta * res.C * res.a
    gain  = res.delta * res.C
    res.x .= decay .* res.x .+ gain .* xnew
    return res
end

"""
    train_jaeger_feedback!(res, Y; washout, noise_scale=1e-5, ridgeλ=1e-8, rng=...)

Teacher-forced training on target sequence `Y` (Ny×T, e.g. `mg_squash`'d
Mackey-Glass): at each step n, `res` is advanced using the *true* `Y[:,n-1]`
fed back through `Wback` (Y[:,0] ≡ 0), the resulting state `x(n)` is paired
with target `Y[:,n]`. The first `washout` pairs are discarded, and `Wout` is
fit by ridge regression (Jaeger describes plain linear regression; a small
`ridgeλ` here is purely for numerical conditioning, not real shrinkage).
`noise_scale` is Jaeger's training-time state noise -- 0 for τ=17 ("no noise
insertion was required"), ~1e-5 for τ=30 ("crucial for stability").
"""
function train_jaeger_feedback!(res::JaegerFeedbackESN, Y::AbstractMatrix;
    washout::Int, noise_scale::Float64 = 0.0, ridgeλ::Float64 = 1e-8,
    u::Float64 = 0.2, rng::AbstractRNG = Random.default_rng(), do_reset::Bool = true,
)
    do_reset && reset!(res)
    Ny, T = size(Y)
    X = zeros(Float64, res.Nh, T)
    y_prev = zeros(Float64, Ny)
    for n in 1:T
        step!(res, y_prev; u=u, noise_scale=noise_scale, rng=rng)
        X[:, n] .= res.x
        y_prev = Y[:, n]
    end
    Xtr = @view X[:, (washout + 1):end]
    Ytr = @view Y[:, (washout + 1):end]
    Wout = (Ytr * Xtr') / (Xtr * Xtr' + ridgeλ * I)
    return Wout, copy(res.x)
end

"""
    evaluate_nrmse84(res, Wout; tau=17, n_trials=50, teacher_len=1000, horizon=84,
                      sigma2=0.067, u=0.2, discard=1000, rng=...)

Reproduces Jaeger (2001) Section 6.3's NRMSE84 protocol exactly: one long
Mackey-Glass run of `n_trials*(teacher_len+horizon)` output steps is split
into `n_trials` consecutive, non-overlapping blocks ("50 subsequent
1084-step evolutions of a 50×1084 run"). For each block: reset `res`,
teacher-force the first `teacher_len` steps with the true (squashed)
sequence, then free-run `horizon` steps feeding the network's own output
back through `Wback`. The free-run prediction at the end of the horizon is
un-squashed back to original Mackey-Glass coordinates and compared to
ground truth.

`sigma2` defaults to Jaeger's own stated variance of the original attractor
signal (≈0.067), so `nrmse84` is directly comparable to his reported
values; pass `sigma2=nothing` to normalise by this run's own empirical
variance instead (a useful cross-check if the two disagree).
"""
function evaluate_nrmse84(res::JaegerFeedbackESN, Wout::AbstractMatrix;
    tau::Int = 17, n_trials::Int = 50, teacher_len::Int = 1000, horizon::Int = 84,
    sigma2::Union{Nothing,Float64} = 0.067, u::Float64 = 0.2, discard::Int = 1000,
    rng::AbstractRNG = Random.default_rng(),
)
    block_len = teacher_len + horizon
    mg = mackey_glass_data(tau=tau, n_out=n_trials * block_len, discard=discard)
    y_orig_full = vec(mg.data)

    sq_errs = zeros(Float64, n_trials)
    truths  = zeros(Float64, n_trials)
    preds   = zeros(Float64, n_trials)

    for i in 1:n_trials
        seg = @view y_orig_full[((i - 1) * block_len + 1):(i * block_len)]
        Y = reshape(mg_squash(collect(seg)), 1, :)

        reset!(res)
        y_prev = zeros(1)
        for n in 1:teacher_len
            step!(res, y_prev; u=u, noise_scale=0.0, rng=rng)
            y_prev = Y[:, n]
        end
        for _ in 1:horizon
            step!(res, y_prev; u=u, noise_scale=0.0, rng=rng)
            y_prev = vec(Wout * res.x)
        end

        pred_orig  = mg_unsquash(y_prev)[1]
        truth_orig = seg[teacher_len + horizon]
        sq_errs[i] = (pred_orig - truth_orig)^2
        truths[i]  = truth_orig
        preds[i]   = pred_orig
    end

    s2 = sigma2 === nothing ? var(truths) : sigma2
    nrmse84 = sqrt(mean(sq_errs) / s2)

    return (nrmse84=nrmse84, sq_errs=sq_errs, truths=truths, preds=preds, sigma2=s2, n_trials=n_trials)
end

"""
    jaeger_mackey_glass_reproduction(; tau=17, train_len=3000, washout=1000,
                                       Nh=400, noise_scale=0.0, seed=1)

End-to-end reproduction of Jaeger (2001) Section 6's Mackey-Glass NRMSE84
benchmark: build the network per his spec, train on `train_len` steps
(his own values: 3000 or 21000), evaluate NRMSE84 over 50 trials. Returns
everything needed to inspect the fit.
"""
function jaeger_mackey_glass_reproduction(;
    tau::Int = 17, train_len::Int = 3000, washout::Int = 1000,
    Nh::Int = 400, noise_scale::Float64 = 0.0, seed::Int = 1, ridgeλ::Float64 = 1e-8,
)
    rng = MersenneTwister(seed)
    res = JaegerFeedbackESN(1; Nh=Nh, rng=rng)

    mg_train = mackey_glass_data(tau=tau, n_out=train_len, discard=1000)
    Y = reshape(mg_squash(vec(mg_train.data)), 1, :)

    Wout, _ = train_jaeger_feedback!(res, Y; washout=washout, noise_scale=noise_scale, ridgeλ=ridgeλ, rng=rng)

    ev = evaluate_nrmse84(res, Wout; tau=tau, rng=rng)

    return (res=res, Wout=Wout, nrmse84=ev.nrmse84, eval=ev, train_len=train_len, tau=tau, seed=seed)
end

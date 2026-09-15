# reservoir_esn.jl
using Random, LinearAlgebra   # Random: RNGs + rand(); LinearAlgebra: norm(), matrix mult, etc.
using ProgressMeter

# -----------------------------
# A minimal Echo State Network (ESN) reservoir
# Implements your AbstractReservoir interface.
# -----------------------------
mutable struct BasicESN <: AbstractReservoir
    Nu::Int                # input dimension (length of u)
    Nh::Int                # reservoir (hidden) dimension
    x::Vector{Float64}     # current reservoir state, size Nh
    Win::Matrix{Float64}   # input weights, size Nh × (1+Nu) (includes bias term)
    W::Matrix{Float64}     # recurrent weights, size Nh × Nh
    leak::Float64          # leaky integration parameter in [0,1] (1.0 = no leak)
end

"""
    BasicESN(Nu; Nh=500, spectral_radius=0.9, input_scale=0.2, leak=1.0, density=0.05, rng=...)

Construct a random ESN reservoir:

- `Win` is dense uniform in [-input_scale, input_scale], with an extra bias column.
- `W` is sparse-ish by masking entries with probability `density`, then scaled so that
  its spectral radius is approximately `spectral_radius` (using a crude power iteration).
"""
function BasicESN(
    Nu::Int;
    Nh=500,
    spectral_radius=0.9,
    input_scale=0.2,
    leak=1.0,
    density=0.05,
    rng=Random.default_rng()
)
    # Initialise reservoir state at zero
    x = zeros(Float64, Nh)

    # Input weight matrix:
    # - draw from Uniform(-1, 1)
    # - scale by input_scale
    # - size is Nh × (1+Nu) because we prepend a bias "1.0" to the input vector
    Win = input_scale .* (2 .* rand(rng, Nh, 1+Nu) .- 1)

    # Recurrent weight matrix W:
    # - start dense in Uniform(-1, 1)
    # - then sparsify: keep an entry with probability `density`, else set to zero
    W = (2 .* rand(rng, Nh, Nh) .- 1)
    W .*= (rand(rng, Nh, Nh) .< density)

    # --- crude power-iteration spectral radius scaling ---
    # Goal: scale W so that its spectral radius ≈ spectral_radius.
    #
    # We approximate the largest eigenvalue magnitude by repeatedly applying W to a vector
    # and normalising (power iteration). This targets the dominant eigen-direction.
    v = rand(rng, Nh)                    # random initial vector
    for _ in 1:50                        # usually enough; increase if Nh is very large or density very low
        v = W * v                        # apply linear map
        v ./= (norm(v) + 1e-12)          # normalise to avoid blow-up; epsilon avoids divide-by-zero
    end
    ρ = norm(W * v) + 1e-12              # estimate dominant growth factor ≈ |λ_max|
    W .*= (spectral_radius / ρ)          # rescale so estimated ρ becomes spectral_radius

    return BasicESN(Nu, Nh, x, Win, W, leak)
end

# Reset the reservoir to its initial (zero) state.
# `rng` is accepted for interface symmetry, but unused here.
reset!(res::BasicESN; rng=Random.default_rng()) = (fill!(res.x, 0.0); res)

# Expose the raw internal state (for logging/debugging or for your training loop).
raw_state(res::BasicESN) = copy(res.x)

"""
    reservoir_step!(res, u)

Advance the reservoir by one time step using input vector `u` (length Nu):

- form augmented input φ = [1; u] to include a bias
- compute pre-activation: Win*φ + W*x
- apply tanh nonlinearity
- update state with leaky integration:
    x ← (1-leak)*x + leak*tanh(pre)
"""
function reservoir_step!(res::BasicESN, u::AbstractVector; rng=Random.default_rng())
    @assert length(u) == res.Nu          # ensure input matches declared input dimension

    φ = vcat(1.0, u)                     # augmented input with bias (length 1+Nu)

    pre = res.Win * φ .+ res.W * res.x   # affine + recurrent drive, elementwise + for vectors
    xnew = tanh.(pre)                    # apply nonlinearity elementwise

    # Leaky integrator update:
    # - leak=1.0: x becomes xnew immediately
    # - leak<1.0: x blends old state and new activation (longer memory / smoother dynamics)
    res.x .= (1 - res.leak) .* res.x .+ res.leak .* xnew

    return res
end

# Feature map used by the readout/training code.
# Here it's just the reservoir state itself (no augmentation like [1; u; x] etc.).
# BasicESN has no observation layer, so the generic 2-arg
# `feature_map(res, obs::Nothing) = feature_map(res)` fallback in
# my_reservoir_core.jl is what makes `feature_map(res, nothing)` work; the
# `obs` argument only matters for reservoirs with a real observation layer
# (e.g. CouzinReservoir's KernelLayer in SWARM_RC/my_swarmRC.jl).
feature_map(res::BasicESN) = res.x


# ============================================================
# Analysis helpers for BasicESN
# ============================================================

clone_reservoir(res::BasicESN) =
    BasicESN(res.Nu, res.Nh, copy(res.x), copy(res.Win), copy(res.W), res.leak)

perturb_state!(res::BasicESN, ξ::AbstractVector) = begin
    @assert length(ξ) == res.Nh
    res.x .+= ξ
    res
end

state_vector(res::BasicESN, obs=nothing) = copy(res.x)

"""
Approximate Jacobian of one ESN step at the current state and input u.

For the leaky tanh ESN:
x⁺ = (1-leak)x + leak*tanh(Win*[1;u] + W*x)

So
J = (1-leak)I + leak*Diag(1 - tanh(pre)^2)*W
"""
function local_jacobian(res::BasicESN, u::AbstractVector)
    @assert length(u) == res.Nu
    φ = vcat(1.0, u)
    pre = res.Win * φ .+ res.W * res.x
    dσ = 1 .- tanh.(pre).^2
    return (1 - res.leak) * I(res.Nh) + res.leak * Diagonal(dσ) * res.W
end

"""
Optional instantaneous Jacobian-based stability summary along a driven trajectory.
This is complementary to perturbation_stability(...).
"""
function jacobian_stability_profile(
    res::BasicESN,
    U::AbstractMatrix;
    washout::Int=0,
    rng=Random.default_rng(),
    show_progress::Bool=true,
)
    r = clone_reservoir(res)
    reset!(r; rng=rng)

    T = size(U, 2)
    vals = zeros(Float64, T)

    # Each step here is an Nh x Nh operator-norm (SVD-based), i.e. O(Nh^3) --
    # this is the diagnostic flagged as slow for large Nh, hence the bar.
    prog = show_progress ? Progress(T; desc="jacobian_stability_profile", showspeed=true) : nothing

    for t in 1:T
        u = view(U, :, t)
        J = Matrix(local_jacobian(r, u))
        vals[t] = opnorm(J, 2)
        reservoir_step!(r, u; rng=rng)
        show_progress && next!(prog)
    end

    t0 = min(washout + 1, T)

    return (
        opnorms = vals,
        mean_opnorm = mean(vals[t0:end]),
        max_opnorm = maximum(vals[t0:end]),
        frac_contracting = mean(vals[t0:end] .< 1.0)
    )
end
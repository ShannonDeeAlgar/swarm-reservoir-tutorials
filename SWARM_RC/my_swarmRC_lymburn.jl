# ============================================================
# my_swarmRC_lymburn.jl
# Wires the Lymburn swarm model (ABM/models/Lymburn/) into the generic
# reservoir-computing interface (RC/my_reservoir_core.jl), as an
# alternative to CouzinReservoir (SWARM_RC/my_swarmRC.jl).
# ============================================================
#
# Kept in its own file so the existing, working CouzinReservoir path is
# untouched. Everything here is new dispatch (new type, new methods on
# existing generic function names) rather than a modification of
# my_swarmRC.jl.
#
# Load order:
#   ABM/load_ABM.jl
#   ABM/models/Lymburn/load_Lymburn.jl
#   RC/my_reservoir_core.jl
#   SWARM_RC/my_swarmRC.jl            -- defines KernelLayer, feature_map dispatch base
#   SWARM_RC/my_swarmRC_lymburn.jl    -- this file
#
# Lymburn's paper-specific predator remains a force term (Eq 10/11), while
# the reservoir wrapper uses the same InputCoupling selection as Couzin.
# NoCoupling disables that term; PredatorCoupling supplies its radius,
# strength multiplier and flee/chase sign.

using Random, LinearAlgebra
using ProgressMeter

"""
    LymburnReservoir <: AbstractReservoir

A Lymburn swarm optionally driven through an `InputCoupling`. With
`PredatorCoupling`, the 2D input is a predator position entering through Eq 10.
With `NoCoupling`, the supplied input is ignored.
Build with `build_Lymburn_reservoir`; conforms to the
`AbstractReservoir` interface (`reset!`, `reservoir_step!`, `feature_map`,
`raw_state`, `build_observation_layer!`), so `train_and_evaluate_reservoir`
works on it directly, exactly as for `CouzinReservoir`.
"""
Base.@kwdef mutable struct LymburnReservoir <: AbstractReservoir
    rng::AbstractRNG
    state::SwarmState{SVector2}
    P::LymburnParams
    dt::Float64
    coupling::InputCoupling
    init_radius::Float64 = 5.0
end

"""
    build_Lymburn_reservoir(P, dt; init_radius=5.0, rng)

Build a swarm-reservoir from Lymburn parameters `P` and step size `dt`.
`coupling=PredatorCoupling(rp=P.rp)` reproduces the paper's force input;
`coupling=NoCoupling()` disables input explicitly. For predator coupling,
`P.Kp * coupling.alpha` is the force strength.
"""
function build_Lymburn_reservoir(P::LymburnParams, dt::Float64;
    coupling::InputCoupling = PredatorCoupling(rp = P.rp),
    init_radius::Float64 = 5.0,
    rng::AbstractRNG = Random.default_rng(),
)
    coupling isa NoCoupling || coupling isa PredatorCoupling ||
        error("LymburnReservoir supports NoCoupling or PredatorCoupling.")
    state = init_state_Lymburn_2d(P; rng = rng, init_radius = init_radius)
    return LymburnReservoir(rng = rng, state = state, P = P, dt = dt,
        coupling = coupling, init_radius = init_radius)
end

# ------------------------------------------------------------
# AbstractReservoir interface
# ------------------------------------------------------------

function reset!(res::LymburnReservoir; rng::AbstractRNG = Random.default_rng())
    res.rng = rng
    res.state = init_state_Lymburn_2d(res.P; rng = rng, init_radius = res.init_radius)
    return res
end

function reservoir_step!(res::LymburnReservoir, u::AbstractVector; rng::AbstractRNG = Random.default_rng())
    if res.coupling isa NoCoupling
        Lymburn_step_2d!(res.state, res.P, res.dt; rng = rng)
    elseif res.coupling isa PredatorCoupling
        length(u) == 2 || throw(DimensionMismatch(
            "PredatorCoupling requires a 2D predator position; got $(length(u)) value(s)."))
        c = res.coupling
        Lymburn_step_with_input!(res.state, res.P, res.dt, SVector2(u[1], u[2]);
            predator_gain = c.alpha, predator_radius = c.rp,
            predator_flee = c.flee, rng = rng)
    else
        error("Input coupling $(typeof(res.coupling)) is not implemented for LymburnReservoir.")
    end
    return res
end

function raw_state(res::LymburnReservoir)
    N = length(res.state.pos)
    Pmat = Matrix{Float64}(undef, 2, N)
    Vmat = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        Pmat[1, i] = res.state.pos[i].x
        Pmat[2, i] = res.state.pos[i].y
        Vmat[1, i] = res.state.vel[i].x
        Vmat[2, i] = res.state.vel[i].y
    end
    return (P = Pmat, V = Vmat)
end

# ------------------------------------------------------------
# Optional reservoir-analysis interface (needed by perturbation_stability)
# ------------------------------------------------------------
# Same rationale as CouzinReservoir: state_vector/perturb_state! operate on
# the raw dynamical state (positions+velocities), not on feature_map's
# output, since the Gaussian-kernel features are a many-to-one projection.

clone_reservoir(res::LymburnReservoir) = LymburnReservoir(
    rng = res.rng,
    state = copy_state(res.state),
    P = res.P,
    dt = res.dt,
    coupling = res.coupling,
    init_radius = res.init_radius,
)

function state_vector(res::LymburnReservoir, obs = nothing)
    rs = raw_state(res)
    return vcat(vec(rs.P), vec(rs.V))
end

"""
    perturb_state!(res::LymburnReservoir, ξ)

Add a perturbation `ξ` (length 4N: 2N position components then 2N velocity
components, matching `state_vector`'s layout) directly to agent positions
and velocities.
"""
function perturb_state!(res::LymburnReservoir, ξ::AbstractVector)
    N = length(res.state.pos)
    @assert length(ξ) == 4N "perturb_state!: expected length 4N=$(4N), got $(length(ξ))"
    @inbounds for i in 1:N
        p = res.state.pos[i]
        res.state.pos[i] = SVector2(p.x + ξ[2i - 1], p.y + ξ[2i])
    end
    @inbounds for i in 1:N
        v = res.state.vel[i]
        res.state.vel[i] = SVector2(v.x + ξ[2N + 2i - 1], v.y + ξ[2N + 2i])
    end
    return res
end

"""
    feature_map(res::LymburnReservoir)

The "naive" reservoir state: raw 2N particle-position coordinates (Sec IIB
of the paper), with no observation layer at all. Dispatches automatically
via the generic `feature_map(res, obs::Nothing) = feature_map(res)`
fallback in RC/my_reservoir_core.jl when `train_and_evaluate_reservoir` is
called with `build_observation_layer! = (res, U; kwargs...) -> nothing` --
this is what reproduces the paper's naive-vs-kernel comparison (Fig 2 vs
Fig 5, R≈0.19 vs R≈0.74).
"""
feature_map(res::LymburnReservoir) = vec(raw_state(res).P)

"""
    feature_map(res::LymburnReservoir, K::KernelLayer)

3M features per timestep: (Gaussian-kernel density, vx-weighted density,
vy-weighted density) at each of the M kernel centres in `K` -- Eqs 16-18,
identical math to `feature_map(res::CouzinReservoir, ...)` in
`SWARM_RC/my_swarmRC.jl`, just reading `LymburnReservoir`'s state and using
plain (non-periodic) distances since this model has no domain.
"""
function feature_map(res::LymburnReservoir, K::KernelLayer)
    C = K.C
    inv2w = K.inv2w
    M = size(C, 2)

    r1 = zeros(Float64, M)
    r2 = zeros(Float64, M)
    r3 = zeros(Float64, M)

    @inbounds for i in eachindex(res.state.pos)
        ax, ay = res.state.pos[i].x, res.state.pos[i].y
        vx, vy = res.state.vel[i].x, res.state.vel[i].y
        for m in 1:M
            dx = ax - C[1, m]
            dy = ay - C[2, m]
            ψ = exp(-(dx * dx + dy * dy) * inv2w[m])
            r1[m] += ψ
            r2[m] += ψ * vx
            r3[m] += ψ * vy
        end
    end

    return vcat(r1, r2, r3)
end

# ------------------------------------------------------------
# Observation-layer construction (Eq 15: random agent+time sampling,
# width = distance to kneigh-th nearest neighbour)
# ------------------------------------------------------------

"""
    build_observation_layer!(res::LymburnReservoir, Utrain; washout=0, rng, method=:spatial_gaussian, kwargs...)

Required override of the generic `build_observation_layer!` hook (see
RC/my_reservoir_core.jl). `method=:spatial_gaussian` is implemented for
`LymburnReservoir` (the paper's observation layer, Eq 15; legacy alias
`:lymburn`) --
`:coverage_kmeans` is a `CouzinReservoir`-only extension for now.
"""
function build_observation_layer!(res::LymburnReservoir, Utrain::AbstractMatrix;
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    method::Symbol = :spatial_gaussian,
    obs_kwargs...,
)
    method in (:spatial_gaussian, :lymburn) || error(
        "LymburnReservoir supports method=:spatial_gaussian (legacy alias :lymburn); got $method.")
    return build_observation_layer_spatial_gaussian!(res, Utrain; washout = washout, rng = rng, obs_kwargs...)
end

"""
    build_observation_layer_spatial_gaussian!(res::LymburnReservoir, Utrain; washout=0, rng, M=200, kneigh=5)

Lymburn-style kernel placement (Eq 15): for each of `M` kernels, pick a
random post-washout time and a random agent; the kernel centre is that
agent's position, and its width is set from the distance to its
`kneigh`-th nearest neighbour at that time (drives the reservoir once,
teacher-forced by `Utrain`, to harvest these samples). Identical algorithm
to `build_observation_layer_spatial_gaussian!(res::CouzinReservoir, ...)` in
`SWARM_RC/my_swarmRC.jl`, using plain Euclidean distance instead of the
periodic-boundary version since this model has no domain.

`inv2w[m] = 1/(2w)` where `w` is that raw kNN distance -- confirmed
against the original authors' MATLAB (`RBF_state.m`: `exp(-d^2/(2*width))`
with `width` the kNN distance directly, i.e. used as a variance-like term,
not squared into one).

`w` is the `(kneigh-1)`-th nearest *other* agent, not the `kneigh`-th --
confirmed against `swarm_lorenz_predict.m`, whose own distance list
includes the chosen agent itself (at distance 0) before sorting, so its
`sorted_d(kth_neighbour)` with `kth_neighbour=5` lands on the 4th other
agent, not the 5th. `kth_neighbour_dist` below excludes self from `d`
already, so it indexes `kneigh-1` to match the MATLAB implementation.
"""
function build_observation_layer_spatial_gaussian!(res::LymburnReservoir, Utrain::AbstractMatrix;
    washout::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    M::Int = 200,
    kneigh::Int = 5,
    show_progress::Bool = true,
)
    reset!(res; rng = rng)

    Nu, T = size(Utrain)
    @assert T > washout + 1 "washout too large for training length"

    N = res.P.N
    @assert 2 <= kneigh < N "kneigh must be >= 2 (indexes the (kneigh-1)-th other agent, matching MATLAB's self-inclusive indexing -- see kth_neighbour_dist) and < number of agents"

    t_samp = rand(rng, (washout + 1):T, M)
    i_samp = rand(rng, 1:N, M)

    by_t = Dict{Int, Vector{Int}}()
    for m in 1:M
        push!(get!(by_t, t_samp[m], Int[]), m)
    end

    C = zeros(2, M)
    inv2w = zeros(Float64, M)

    function kth_neighbour_dist(i::Int)
        xi, yi = res.state.pos[i].x, res.state.pos[i].y
        d = Vector{Float64}(undef, N - 1)
        idx = 1
        @inbounds for j in 1:N
            j == i && continue
            dx = res.state.pos[j].x - xi
            dy = res.state.pos[j].y - yi
            d[idx] = hypot(dx, dy)
            idx += 1
        end
        sort!(d)
        # MATLAB's own `swarm_lorenz_predict.m` includes the agent itself (at
        # distance 0) in the sorted list before indexing, so its
        # `kth_neighbour` is really the (kth_neighbour-1)-th *other* agent --
        # `d` here already excludes self, so match that by indexing kneigh-1.
        return d[kneigh - 1]
    end

    prog = show_progress ? Progress(T; desc = "fit spatial Gaussian kernels", showspeed = true) : nothing

    for t in 1:T
        reservoir_step!(res, view(Utrain, :, t); rng = rng)

        idxs = get(by_t, t, nothing)
        if idxs !== nothing
            for m in idxs
                i = i_samp[m]
                C[1, m] = res.state.pos[i].x
                C[2, m] = res.state.pos[i].y

                w = max(kth_neighbour_dist(i), 1e-6)
                inv2w[m] = 1 / (2w)
            end
        end

        show_progress && next!(prog)
    end

    return KernelLayer(C, inv2w)
end

build_observation_layer_lymburn!(res::LymburnReservoir, Utrain::AbstractMatrix; kwargs...) =
    build_observation_layer_spatial_gaussian!(res, Utrain; kwargs...)

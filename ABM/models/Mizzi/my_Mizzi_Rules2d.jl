# ============================================================
# Territorial-agent force law and explicit-Euler update
# Mizzi et al., "Reservoir Computing with Territorial Agents", Sec. 2
# ============================================================

using Random
using LinearAlgebra
using StaticArrays

@inline mizzi_displacement_2d(a::SVector2, b::SVector2, ::Real) = b - a

"""
    mizzi_forces(pos, vel, P, prey=nothing)

Compute `Fh + Fd + Ff` for every agent. If prey `p` lies in territory `j`,
the driving force applies to agent `j` and agents whose Voronoi cells share a
boundary with `j`. `Fd` has fixed magnitude `kd`, except at zero separation
where it is defined as zero.
"""
function mizzi_forces(
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    P::MizziParams,
    prey::Union{Nothing,SVector2} = nothing,
)
    length(pos) == P.N == length(vel) || error("State size does not match MizziParams.N.")
    active = falses(P.N)
    owner = nothing
    if prey !== nothing
        owner = mizzi_territory_owner(P, prey)
        active[owner] = true
        active[P.neighbours[owner]] .= true
    end

    F = Vector{SVector2}(undef, P.N)
    @inbounds for i in 1:P.N
        Fh = P.kh * (P.homes[i] - pos[i])
        Ff = -P.kf * vel[i]
        Fd = SVector2(0.0, 0.0)
        if prey !== nothing && active[i]
            Δ = prey - pos[i]
            d = norm(Δ)
            d > eps(Float64) && (Fd = (P.kd / d) * Δ)
        end
        F[i] = Fh + Fd + Ff
    end
    return F, owner, active
end

"""
    Mizzi_step_2d!(state, P, dt; prey=nothing)

Apply the paper's explicit Euler equations exactly: position uses the old
velocity, while velocity uses the force evaluated from the old state.
"""
function Mizzi_step_2d!(
    state::SwarmState{SVector2}, P::MizziParams, dt::Float64;
    prey::Union{Nothing,SVector2} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    update_scheme::Symbol = :synchronous,
)
    dt > 0 || error("dt must be positive.")
    update_scheme == :synchronous || error("Territorial agents use a synchronous explicit-Euler update.")
    F, _, _ = mizzi_forces(state.pos, state.vel, P, prey)
    old_pos = copy(state.pos)
    old_vel = copy(state.vel)
    @inbounds for i in 1:P.N
        state.vel[i] = old_vel[i] + F[i] * dt
        state.pos[i] = old_pos[i] + old_vel[i] * dt
        state.desired[i] = F[i]
    end
    return state
end

"A non-periodic order-parameter summary for plotting/diagnostics only."
function Mizzi_order_parameters_2d(
    pos::Vector{SVector2}, vel::Vector{SVector2}, L::Real; displacement,
)
    N = length(pos)
    N == 0 && return (0.0, 0.0, 0.0, 0.0)
    centre = SVector2(0.0, 0.0)
    mean_heading = SVector2(0.0, 0.0)
    @inbounds for i in 1:N
        centre += pos[i]
        nv = norm(vel[i])
        nv > 1e-12 && (mean_heading += vel[i] / nv)
    end
    centre /= N
    mean_heading /= N
    polarisation = norm(mean_heading)
    rot = dil = ang = 0.0
    count = 0
    @inbounds for i in 1:N
        r = pos[i] - centre
        nr, nv = norm(r), norm(vel[i])
        if nr > 1e-12 && nv > 1e-12
            rhat, vhat = r / nr, vel[i] / nv
            that = SVector2(-rhat.y, rhat.x)
            rot += dot(vhat, that)
            dil += dot(vhat, rhat)
            ang += r.x * vhat.y - r.y * vhat.x
            count += 1
        end
    end
    count == 0 && return (0.0, 0.0, polarisation, 0.0)
    return (dil / count, abs(rot / count), polarisation, abs(ang / count))
end

"""
    simulate_Mizzi_2d(simcfg, P; prey=nothing, ...)

Simulate the basic force model. `prey` may be `nothing` or a 2×steps matrix;
column `k` drives Euler step `k`.
"""
function simulate_Mizzi_2d(
    simcfg::SimulationConfig, P::MizziParams;
    prey::Union{Nothing,AbstractMatrix} = nothing,
    dt_override::Union{Nothing,Float64} = nothing,
    position_noise::Float64 = 0.0,
    velocity_noise::Float64 = 0.0,
    collect_order::Bool = true,
    collect_history::Bool = true,
    show_progress::Bool = true,
)
    dt = something(dt_override, simcfg.dt)
    if prey !== nothing
        size(prey, 1) == 2 || error("prey must be a 2×T matrix.")
        size(prey, 2) >= simcfg.steps || error("prey needs at least $(simcfg.steps) columns.")
        all(isfinite, prey) || error("prey contains NaN or Inf.")
    end

    step_index = Ref(0)
    step_fn! = function (state, P, dt; rng=Random.default_rng(), update_scheme=:synchronous)
        step_index[] += 1
        p = prey === nothing ? nothing : SVector2(prey[1, step_index[]], prey[2, step_index[]])
        Mizzi_step_2d!(state, P, dt; prey=p, rng=rng, update_scheme=update_scheme)
    end
    init_fn = (P; rng=Random.default_rng()) -> init_state_Mizzi_2d(P;
        rng=rng, position_noise=position_noise, velocity_noise=velocity_noise)

    return simulate(simcfg, P;
        dt=dt, init_state=init_fn, step! = step_fn!,
        get_pos=state -> state.pos, get_vel=state -> state.vel,
        domain_size=(P, state) -> 2P.plot_extent,
        displacement=mizzi_displacement_2d,
        order_parameters=Mizzi_order_parameters_2d,
        collect_order=collect_order, collect_history=collect_history,
        show_progress=show_progress)
end

using Random
using LinearAlgebra
using Statistics

Base.@kwdef mutable struct LundReservoir <: AbstractReservoir
    rng::AbstractRNG
    state::LundState
    P::LundParams
    dt::Float64 = P.dt
    coupling::InputCoupling = TemperatureSpeedCoupling(
        base_speed=P.base_speed, gain=P.temperature_gain,
        min_speed=P.min_speed, max_speed=P.max_speed)
end

function build_Lund_reservoir(P::LundParams=LundParams();
    coupling::InputCoupling=TemperatureSpeedCoupling(
        base_speed=P.base_speed, gain=P.temperature_gain,
        min_speed=P.min_speed, max_speed=P.max_speed),
    rng::AbstractRNG=Random.default_rng())
    coupling isa TemperatureSpeedCoupling || coupling isa NoCoupling ||
        error("LundReservoir supports TemperatureSpeedCoupling or NoCoupling.")
    LundReservoir(rng=rng, state=init_state_Lund_2d(P; rng=rng), P=P,
        dt=P.dt, coupling=coupling)
end

function reset!(res::LundReservoir; rng::AbstractRNG=Random.default_rng())
    res.rng = rng
    res.state = init_state_Lund_2d(res.P; rng=rng)
    return res
end

function reservoir_step!(res::LundReservoir, u::AbstractVector;
    rng::AbstractRNG=Random.default_rng())
    length(u) == 1 || throw(DimensionMismatch(
        "LundReservoir expects one scalar temperature input; got $(length(u))."))
    value = only(u)
    starget = res.coupling isa NoCoupling ? res.P.base_speed :
        target_speed(res.coupling, value)
    Lund_step_2d!(res.state, res.P, value;
        target_speed_override=starget, rng=rng)
    return res
end

function raw_state(res::LundReservoir)
    N = res.P.N
    Pmat = Matrix{Float64}(undef, 2, N)
    Vmat = similar(Pmat)
    @inbounds for i in 1:N
        Pmat[1, i], Pmat[2, i] = res.state.pos[i].x, res.state.pos[i].y
        Vmat[1, i], Vmat[2, i] = res.state.vel[i].x, res.state.vel[i].y
    end
    return (P=Pmat, V=Vmat, agent_state=copy(res.state.agent_state),
        local_energy=copy(res.state.local_energy))
end

function _lund_convex_hull_area(points::Vector{SVector2})
    length(points) < 3 && return 0.0
    pts = sort(unique(points); by=p -> (p.x, p.y))
    length(pts) < 3 && return 0.0
    cross(o, a, b) = (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x)
    lower = SVector2[]
    for p in pts
        while length(lower) >= 2 && cross(lower[end-1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    upper = SVector2[]
    for p in Iterators.reverse(pts)
        while length(upper) >= 2 && cross(upper[end-1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    hull = vcat(lower[1:end-1], upper[1:end-1])
    return abs(sum(hull[i].x*hull[mod1(i+1,end)].y -
        hull[mod1(i+1,end)].x*hull[i].y for i in eachindex(hull))) / 2
end

"Paper Table 2 aggregate observation features (nine invariant statistics)."
function feature_map(res::LundReservoir)
    P, state, N = res.P, res.state, res.P.N
    speeds = norm.(state.vel)
    mean_speed = mean(speeds)
    alignment = norm(mean(state.vel)) / (mean_speed + eps())
    pair_distance = mean(norm(lund_displacement(state.pos[i], state.pos[j], P.L))
        for i in 1:N for j in 1:N if i != j)
    clustered = state.agent_state .== 1
    dispersed = .!clustered
    speed0 = any(dispersed) ? mean(speeds[dispersed]) : 0.0
    speed1 = any(clustered) ? mean(speeds[clustered]) : 0.0
    centroid = mean(state.pos)
    angular_momentum = mean((state.pos[i].x-centroid.x)*state.vel[i].y -
        (state.pos[i].y-centroid.y)*state.vel[i].x for i in 1:N)
    return [_lund_convex_hull_area(state.pos), pair_distance, alignment,
        mean_speed, std(speeds), mean(clustered), angular_momentum, speed0, speed1]
end

clone_reservoir(res::LundReservoir) = LundReservoir(rng=res.rng,
    state=copy_state(res.state), P=res.P, dt=res.dt, coupling=res.coupling)

state_vector(res::LundReservoir, obs=nothing) = vcat(vec(raw_state(res).P),
    vec(raw_state(res).V))

"RMS physical distance with minimum-image position differences on Lund's torus."
function state_distance(r1::LundReservoir, r2::LundReservoir, obs=nothing)
    r1.P.N == r2.P.N || error("Lund replicas must contain the same number of agents.")
    accum = 0.0
    @inbounds for i in 1:r1.P.N
        dp = lund_displacement(r1.state.pos[i], r2.state.pos[i], r1.P.L)
        dv = r1.state.vel[i] - r2.state.vel[i]
        accum += abs2(dp.x) + abs2(dp.y) + abs2(dv.x) + abs2(dv.y)
    end
    return sqrt(accum / (4r1.P.N))
end

function perturb_state!(res::LundReservoir, ξ::AbstractVector)
    N = res.P.N
    length(ξ) == 4N || error("Lund perturbation expects 4N continuous components.")
    @inbounds for i in 1:N
        res.state.pos[i] += SVector2(ξ[2i-1], ξ[2i])
        res.state.vel[i] += SVector2(ξ[2N+2i-1], ξ[2N+2i])
    end
    return res
end

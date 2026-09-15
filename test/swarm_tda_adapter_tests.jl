using Test

include("../TDA/my_swarm_TDA.jl")

struct MockPoint2
    x::Float64
    y::Float64
end

abstract type AbstractReservoir end

mutable struct MockReservoir <: AbstractReservoir
    state
end

clone_reservoir(res::MockReservoir) = deepcopy(res)
reset!(res::MockReservoir; rng=nothing) = res
function reservoir_step!(res::MockReservoir, u; rng=nothing)
    for i in eachindex(res.state.pos)
        p = res.state.pos[i]
        res.state.pos[i] = MockPoint2(p.x + 0.01 * only(u), p.y)
    end
    return res
end

tda_feature_map(res; εs, dims, kwargs...) = repeat(Float64.(εs), length(dims))
include("../TDA/my_TDA_observation.jl")

@testset "TDA observation layer" begin
    res = MockReservoir((pos=[MockPoint2(0, 0), MockPoint2(1, 0), MockPoint2(0, 1)],
        vel=[MockPoint2(1, 0), MockPoint2(0, 1), MockPoint2(-1, 0)]))
    U = reshape(range(-1, 1; length=20), 1, :)
    obs = build_tda_observation_layer!(res, U; washout=2, n_epsilon=8,
        calibration_steps=5, dims=(0, 1))
    @test obs isa TDAObservationLayer
    @test length(obs.εs) == 8
    @test first(obs.εs) == 0
    @test last(obs.εs) > 0
    @test length(feature_map(res, obs)) == 16
    @test_throws ArgumentError TDAObservationLayer(0:0.1:1;
        representation=:position_velocity, periodic=true, L=10)
end

@testset "swarm_snapshot adapter" begin
    raw = (
        P=[0.0 1.0 2.0; 3.0 4.0 5.0],
        V=[1.0 0.0 -1.0; 0.0 1.0 0.0],
    )
    snap = swarm_snapshot(raw)
    @test size(snap.positions) == (3, 2)
    @test snap.positions[2, :] == [1.0, 4.0]
    @test size(snap.velocities) == (3, 2)

    # D×N remains unambiguous even for two agents in three dimensions.
    raw3 = (P=[0.0 1.0; 2.0 3.0; 4.0 5.0],)
    @test swarm_snapshot(raw3).positions == [0.0 2.0 4.0; 1.0 3.0 5.0]

    history = (
        pos_hist=[[MockPoint2(0, 0), MockPoint2(1, 0)],
                  [MockPoint2(0, 1), MockPoint2(1, 1)]],
        vel_hist=[[MockPoint2(1, 0), MockPoint2(1, 0)],
                  [MockPoint2(0, 1), MockPoint2(0, 1)]],
        t=[0.0, 0.25],
    )
    frame = swarm_snapshot(history, 2)
    @test frame.positions == [0.0 1.0; 1.0 1.0]
    @test frame.time == 0.25
    @test frame.frame == 2

    driven = (
        states=[(P=raw.P, V=raw.V), (P=raw.P .+ 1, V=raw.V)],
        t=Float32[0.0, 0.1],
    )
    @test swarm_snapshot(driven, 2).positions[1, :] == [1.0, 4.0]

    current = (state=(
        pos=[MockPoint2(0, 0), MockPoint2(1, 0)],
        vel=[MockPoint2(1, 0), MockPoint2(0, 1)],
        agent_state=Int8[0, 1]), dt=0.1)
    current_snap = swarm_snapshot(current)
    @test current_snap.agent_state == Int8[0, 1]
    @test swarm_pointcloud(current; representation=:position_velocity) ==
        [0.0 0.0 1.0 0.0; 1.0 0.0 0.0 1.0]

    D = swarm_dissimilarity(raw)
    @test size(D) == (3, 3)
    @test D ≈ transpose(D)
    @test all(iszero, diag(D))
    @test_throws ArgumentError swarm_snapshot(history)
    @test_throws ArgumentError swarm_pointcloud((P=raw.P,);
        representation=:position_velocity)
end

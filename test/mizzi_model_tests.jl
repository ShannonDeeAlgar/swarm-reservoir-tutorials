using Test
using Random
using LinearAlgebra

include("../SWARM_RC/load_SwarmRC_Mizzi.jl")
include("../SWARM_RC/my_predator.jl")

@testset "Mizzi-only loader compatibility" begin
    @test isdefined(Main, :zscore_rescale)
    @test isdefined(Main, :run_offline_predator_drive!)
end

@testset "Mizzi geometry and force law" begin
    homes = [
        SVector2(-1.0, -1.0), SVector2(1.0, -1.0),
        SVector2(1.0, 1.0), SVector2(-1.0, 1.0),
        SVector2(0.0, 0.0),
    ]
    P = MizziParams(homes; kh=0.0, kd=2.0, kf=0.0)
    @test P.N == 5
    @test all(i in P.neighbours[5] for i in 1:4)
    @test all(5 in P.neighbours[i] for i in 1:4)

    pos = copy(homes)
    vel = fill(SVector2(0.0, 0.0), P.N)
    prey = SVector2(0.9, 0.9)
    F, owner, active = mizzi_forces(pos, vel, P, prey)
    @test owner == 3
    @test active[owner]
    @test all(active[j] for j in P.neighbours[owner])
    @test count(active) == 1 + length(P.neighbours[owner])
    @test all(isapprox(norm(F[i]), 2.0; atol=1e-12) for i in findall(active))
    @test all(norm(f) == 0.0 for f in F[findall(.!active)])
end

@testset "Paper explicit-Euler ordering" begin
    P = MizziParams([SVector2(0.0, 0.0)]; kh=2.0, kd=0.0, kf=1.0)
    state = SwarmState{SVector2}(
        [SVector2(1.0, 0.0)], [SVector2(0.5, 0.0)], [SVector2(0.0, 0.0)])
    Mizzi_step_2d!(state, P, 0.1)
    @test isapprox(state.pos[1].x, 1.05) && isapprox(state.pos[1].y, 0.0)  # old velocity
    @test isapprox(state.vel[1].x, 0.25) && isapprox(state.vel[1].y, 0.0)  # F=(-2.5, 0)
end

@testset "Reservoir state and interface" begin
    U = [range(-2.0, 2.0; length=40)';
         sin.(range(0.0, 2π; length=40))']
    res = build_Mizzi_reservoir(U, 6, 0.02; rng=MersenneTwister(4))
    @test res.P.kh == 80.0
    @test res.P.kd == 60.0
    @test res.P.kf == 20.0
    @test length(feature_map(res)) == 4 * res.P.N
    @test feature_map(res) ≈ zeros(4 * res.P.N)

    reservoir_step!(res, U[:, 1])
    @test all(isfinite, feature_map(res))

    res.state.pos[1] = res.P.homes[1] + SVector2(1.0, 2.0)
    res.state.vel[1] = SVector2(3.0, 4.0)
    f = feature_map(res)
    N = res.P.N
    @test isapprox(f[1], 1.0)
    @test isapprox(f[N + 1], 2.0)
    @test isapprox(f[2 * N + 1], 3.0)
    @test isapprox(f[3 * N + 1], 4.0)

    report = validate_pipeline(res, U;
        observation=:raw_state, probe_steps=20, rng=MersenneTwister(8))
    @test isempty(report.errors)
end

@testset "Driven ABM simulation" begin
    U = [range(-1.5, 1.5; length=25)';
         cos.(range(0.0, 2π; length=25))']
    homes = mizzi_homes_from_input(U, 5; rng=MersenneTwister(2))
    P = MizziParams(homes)
    out = simulate_Mizzi_2d(
        SimulationConfig(steps=20, dt=0.02, seed=3), P;
        prey=U[:, 1:20], show_progress=false)
    @test length(out.pos_hist) == 21
    @test all(isfinite, reduce(vcat, [[p.x, p.y] for p in out.pos_hist[end]]))
end

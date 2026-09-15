using Test
using Random

include("../SWARM_RC/load_SwarmRC_Lymburn.jl")
include("../ABM/models/Lund/load_Lund.jl")
include("../SWARM_RC/my_swarmRC_lund.jl")

@testset "Lund two-state swarm reservoir" begin
    P = LundParams(N=24, L=128.0, noise_std=0.0,
        energy_threshold=0.9, hysteresis_factor=0.1)
    coupling = TemperatureSpeedCoupling(base_speed=P.base_speed,
        gain=P.temperature_gain, min_speed=P.min_speed, max_speed=P.max_speed)
    res = build_Lund_reservoir(P; coupling=coupling, rng=MersenneTwister(1))

    @test length(feature_map(res)) == 9
    @test size(raw_state(res).P) == (2, P.N)
    @test all(==(0), res.state.agent_state)
    @test target_speed(coupling, 1.0) == 1.3

    reservoir_step!(res, [0.5]; rng=MersenneTwister(2))
    @test all(p -> 0 <= p.x < P.L && 0 <= p.y < P.L, res.state.pos)
    @test_throws DimensionMismatch reservoir_step!(res, [0.1, 0.2])

    U = reshape(sin.(range(0, 8; length=30)), 1, :)
    report = validate_pipeline(res, U;
        observation=:lund_aggregate, probe_steps=12, rng=MersenneTwister(3))
    @test report.valid
    @test report.input_dim == 1

    lymburn = build_Lymburn_reservoir(
        Lymburn_params_from_preset(:critical; N=12), 0.02;
        coupling=PredatorCoupling(rp=2.0), rng=MersenneTwister(4))
    @test_throws ErrorException build_Lymburn_reservoir(lymburn.P, 0.02;
        coupling=coupling, rng=MersenneTwister(4))
end

@testset "Lund paper preset and companion-step parity" begin
    Ppaper = Lund_params_from_preset(:paper)
    @test Ppaper.N == 200
    @test Ppaper.L == 512.0
    @test Ppaper.dt == 0.1
    @test Ppaper.energy_threshold == 3.0
    @test Ppaper.hysteresis_factor == 0.05
    @test Ppaper.inertia_alpha == 1.0
    @test Ppaper.noise_std == 0.01
    @test_throws ArgumentError Lund_params_from_preset(:unknown)

    # Expected values are from the authors' Python MultiStateBoidFlock.
    P = LundParams(N=4, L=512.0, inertia_alpha=1.0, noise_std=0.0,
        energy_threshold=3.0, hysteresis_factor=0.05)
    state = LundState(
        [SVector2(10.0, 10.0), SVector2(20.0, 10.0),
         SVector2(500.0, 10.0), SVector2(250.0, 250.0)],
        [SVector2(1.0, 0.0), SVector2(2.0, 0.0),
         SVector2(4.0, 0.0), SVector2(0.0, 1.0)],
        Int8[0, 0, 0, 1], zeros(4))
    coupling = TemperatureSpeedCoupling(base_speed=1.0, gain=0.30)
    @test target_speed(coupling, 0.25) == 1.075
    Lund_step_2d!(state, P, 0.25;
        target_speed_override=target_speed(coupling, 0.25),
        rng=MersenneTwister(1))

    expected_energy = [3.0, 2.5, 1.5, 1.0]
    expected_pos = [
        SVector2(10.454615766316183, 10.008843545667704),
        SVector2(20.16155810866645, 10.013771065395686),
        SVector2(499.2979352150526, 10.00826591064885),
        SVector2(250.00062879917436, 250.1054286615698)]
    expected_vel = [
        SVector2(4.546157663161822, 0.088435456677037),
        SVector2(1.615581086664508, 0.137710653956865),
        SVector2(-7.020647849473988, 0.082659106488494),
        SVector2(0.006287991743725, 1.054286615697828)]
    @test state.agent_state == Int8[0, 0, 0, 0]
    @test state.local_energy ≈ expected_energy atol=1e-12
    @test all(norm(a - b) <= 1e-8 for (a, b) in zip(state.pos, expected_pos))
    @test all(norm(a - b) <= 1e-8 for (a, b) in zip(state.vel, expected_vel))
end

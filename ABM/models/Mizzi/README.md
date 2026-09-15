# Mizzi territorial-agent model

This directory implements the basic force model from Mizzi et al.,
*Reservoir Computing with Territorial Agents*. It intentionally does **not**
implement consistency capacity, minimum-description-length scoring, or the
evolutionary territory optimiser.

For agent `i`,

```text
Fh = kh (home[i] - position[i])
Fd = kd (prey - position[i]) / ||prey - position[i]||
Ff = -kf velocity[i]
```

`Fd` is present only when the prey lies in agent `i`'s Voronoi cell or a
cell sharing a Voronoi boundary with it. The code finds those neighbours from
the dual Delaunay graph of the fixed homes. Updates use the paper's explicit
Euler ordering:

```text
velocity(t+dt) = velocity(t) + F(t) dt
position(t+dt) = position(t) + velocity(t) dt
```

Defaults are `dt=0.02` and `(kh,kd,kf)=(80,60,20)`.

## ABM use

```julia
include("ABM/load_ABM.jl")
include("ABM/models/Mizzi/load_Mizzi.jl")

# U is a 2×T prey trajectory, normally (u(t-τ), u(t)).
homes = mizzi_homes_from_input(U[:, 1:Ttrain], 30;
    rng=MersenneTwister(1))
P = MizziParams(homes)
out = simulate_Mizzi_2d(
    SimulationConfig(steps=1000, dt=0.02, seed=1), P;
    prey=U[:, 1:1000])
```

## Reservoir use

```julia
include("SWARM_RC/load_SwarmRC_Mizzi.jl")

res = build_Mizzi_reservoir(U[:, 1:Ttrain], 30, 0.02;
    rng=MersenneTwister(1))
validate_pipeline(res, U;
    observation=:raw_state, target=target,
    prediction_mode=:teacher_forced, dt_input=0.02)
```

The reservoir feature vector follows the paper exactly:

```text
(x_positions - x_homes) ⊕ (y_positions - y_homes)
⊕ x_velocities ⊕ y_velocities
```

Homes must be sampled from training data only. Sampling them from a complete
trajectory before the train/test split leaks test-set geometry into the
reservoir design.

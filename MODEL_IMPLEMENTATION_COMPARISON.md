# Published models and this implementation

Last cross-checked: 14 September 2026.

This document separates three questions that are easy to conflate:

1. Does the code implement the published dynamical rule?
2. Does the tutorial use the paper's parameters and observation layer?
3. Does the tutorial reproduce the paper's training and evaluation experiment?

Passing the tutorial tests establishes software consistency. It does not, by
itself, reproduce a published result. Numerical reproductions and longer model
checks belong in `advanced/Existing_Model_Validation.ipynb`.

## Mizzi territorial reservoir

Reference: Antony Mizzi, Shannon D. Algar, Thomas Lymburn, David M. Walker and
Michael Small, *Reservoir Computing with Territorial Agents* (supplied
manuscript). Update this entry when final publication details are available.

| Component | Supplied manuscript | This code | Status |
|---|---|---|---|
| State | One two-dimensional position, velocity and fixed home per agent | `SwarmState{SVector2}` plus one home per agent in `MizziParams` | Matched |
| Home force | $k_h(h^{(i)}-x^{(i)})$ | Same expression | Matched |
| Prey force | Magnitude $k_d$, directed towards the prey | Same expression; defined as zero at exact coincidence | Matched, with a numerical guard |
| Agents receiving prey force | Agent owning the prey's Voronoi cell and agents in immediately neighbouring cells | Owner is the nearest home; neighbours are cells sharing a non-zero Voronoi boundary | Matched |
| Friction | $-k_f\dot{x}^{(i)}$ | Same expression | Matched |
| Update | Synchronous explicit Euler; position uses the old velocity | Old position and velocity are copied before all agents are updated | Matched and unit-tested |
| Standard values | $\Delta t=0.02$ and $(k_h,k_d,k_f)=(80,60,20)$ | Same defaults | Matched |
| Initial state | Not specified in the supplied methods text | Agents begin at their homes with zero velocity unless optional noise is requested | Implementation choice |
| Prey/input | $p(t)=(u(t-\tau),u(t))$ | Same two-coordinate interface | Matched |
| Lorenz example | Lorenz $x$, $\Delta t=0.02$, lag 7, divided by 7.34 | Same sampling, lag and scaling | Matched input preparation; adaptive Tsitouras integration replaces RK4 |
| Random homes | Sample embedded training points, or sample uniformly from $[-2.5,2.5]^2$ | Both methods are implemented; the tutorial preset samples 30 embedded training points | Matched options |
| Observation | $4N$ home-relative position and velocity components, plus an intercept in the readout | Same feature order; the shared readout supplies the intercept | Matched |
| Ridge regression | Fixed $\alpha=10^{-8}$ for one-step prediction | Five-fold blocked selection over a ridge grid | Deliberate change |
| Autonomous forecast | Feed the scalar prediction back into its lag embedding; report valid horizon in Lyapunov times | Beginner tutorial reports held-out teacher-forced correlation | Not reproduced |
| Ensembles | Fifty reservoirs in the forecasting experiments | One seeded reservoir | Not reproduced |
| Consistency capacity | Central diagnostic | Not run in the beginner tutorial | Omitted from beginner route |
| MDL territory selection | Evolutionary optimisation of home number and locations | Not implemented | Out of scope |

The basic force system and direct $4N$ observation are close reproductions of
the supplied manuscript. The tutorial score is not comparable with the paper's
forecasting horizons because training and evaluation differ. The manuscript
also does **not** claim that trajectory-sampled homes always forecast better:
they give higher consistency, while uniform homes can give longer forecasts in
some cases.

## Lymburn swarm reservoir

Reference: Thomas Lymburn, Shannon D. Algar, Michael Small and Thomas Jüngling
(2021), “Reservoir computing with swarms,” *Chaos* 31, 033121,
<https://doi.org/10.1063/5.0039745>.

| Component | Paper / released MATLAB | This code | Status |
|---|---|---|---|
| Dynamics | Non-periodic 2D modified Reynolds swarm with repulsion, alignment, global homing and target-speed friction | Same force terms and non-periodic geometry | Matched |
| Predator force | Inverse-distance force within radius $r_p$ | Same, with a zero-distance guard | Matched |
| Standard driven case | $N=200$, $\Delta t=0.02$, $(K_r,K_a,K_h,K_f,K_p)=(1,0.1,2,20,100)$ and $(r_r,r_a,r_p)=(1,1,2)$ | Available as `:driven` | Matched option |
| Tutorial regime | Paper point B: $(K_r,K_a)=(2,0.01)$ | `:critical` selects point B | Matched operating point |
| Lorenz driver | Lorenz $(x,y)$, sampled at 0.02 and rescaled to zero mean and standard deviation 2 | Same representation and scale | Matched |
| Prediction target | Future Lorenz $x$, 0.5 time units ahead | Same target and horizon | Matched |
| Force cap | Printed equation is ambiguous about vector application; released MATLAB applies $\alpha\tanh(\beta F)$ componentwise | Bare `LymburnParams` and validation mode support componentwise; tutorial presets use a rotationally symmetric magnitude cap | Deliberate tutorial change |
| Euler update | Printed Eq. 9 uses old velocity; released MATLAB updates position with the new velocity | Default is semi-implicit Euler, matching the released MATLAB; explicit Euler remains available | Matches released code, not the printed equation |
| Raw observation | $2N$ agent positions | Available as the baseline | Matched |
| Kernel observation | $M=200$ random agent/time centres; width from that agent's fifth neighbour; density and two velocity-weighted sums ($3M$ features) | Same centre/width construction and three measurements | Matched construction |
| Readout fitting | Ridge regression with training, validation and test sets | Blocked cross-validation over a ridge grid | Same method family, different selection protocol |
| Main paper score | Test correlation; reported over the paper's run lengths and configurations | One seeded, shorter tutorial experiment | Not a numerical reproduction |
| Parameter sweep | $K_r,K_a\in[0.001,100]$, 50,000 steps after a 1,000-step transient | Long saved/validation workflow; beginner tutorial shows a reduced sweep | Full paper experiment not run in beginner route |
| Consistency analysis | Consistency spectrum/capacity for raw and kernel states | Validation material only | Outside beginner route |

The force-cap choice means the beginner preset is intentionally not a literal
trajectory reproduction. Use `cap_mode=:componentwise` when comparing numbers
with the paper. The paper's main qualitative conclusion is more precise than
“intermediate order is best”: high performance occurs where the swarm changes
its behaviour responsively under the predator drive, near the transition
regime.

## Lund two-state swarm

Reference: Tanner Lund, Alyssa Adams, Nathanael Aubert-Kato and Takashi Ikegami
(2026), “State transitions unlock temporal memory in swarm-based reservoir
computing,” *PeerJ Computer Science* 12:e3763,
<https://doi.org/10.7717/peerj-cs.3763>. Companion implementation:
<https://github.com/Nylan17/state-transitions-swarm-reservoirs>.

The Julia implementation provides two presets. `:tutorial` is recalibrated to
show both states in a short run. `:paper` matches the companion repository's
default two-state dynamics. Neither preset by itself reproduces the paper's
memory-capacity experiment, which also requires its input, observation,
preprocessing and evaluation protocol.

| Component | Paper companion configuration | This code / tutorial | Status |
|---|---|---|---|
| Domain and step | 2D torus, side 512, $\Delta t=0.1$ | `:paper` uses 512/0.1; `:tutorial` uses 256/0.1 | Paper preset matched |
| Population | Companion default uses $N=200$; memory scaling also uses $N=800$–2000 | Both presets default to $N=200$; larger $N$ can be requested | Default matched |
| Input | Global scalar interpreted as temperature and mapped to target speed | Scalar Lorenz $x$ mapped to target speed | Same coupling concept; different task signal |
| Linear temperature map | $s_{target}=s_0(1+g u)$, clipped | Same | Matched |
| Local state signal | Mean speed of neighbours within radius 35, falling back to own speed | Same definition | Matched |
| Switching | Two states with hysteresis; companion defaults threshold 3.0 and factor 0.05 | `:paper` uses 3.0/0.05; `:tutorial` uses 1.10/0.10 | Paper preset matched |
| State-dependent rules | Dispersed/clustered radii 28/35; alignment 0.5/1.0; cohesion 0.5/2.0; separation 1.2/1.0; inner radius 15 | Same values | Matched |
| Inertial velocity mixing | Companion default is $\alpha=1$ | `:paper` uses 1.0; `:tutorial` uses 0.4 | Paper preset matched |
| Soft speed normalisation | Relaxation 0.7, clipping 10 | Same | Matched |
| Noise | Companion default 0.01; paper scaling also examines moderate noise | Core default 0.01 | Matched default |
| Centre anchoring | Weight 0.01 in companion implementation | Same | Matched |
| Observation | Companion `MULTI_STATE_FEATURES` contains 17 aggregates plus per-agent speeds and headings | Julia tutorial uses nine aggregates | Deliberate subset |
| Temporal/pre-processing options | Companion code supports leaky features, standardisation, whitening and polynomial expansion; pure-MC protocol explicitly removes polynomial terms | Shared Julia tutorial uses its normal feature standardisation/ridge path and no companion pipeline reproduction | Different pipeline |
| Main result | Pure memory capacity under i.i.d. input, linear readout, no polynomial expansion; scaling over large $N$ and multiple seeds | Task-conditioned memory diagnostic plus one-step Lorenz prediction at $N=200$ | Not reproduced |

Consequently, the tutorial can demonstrate temperature coupling, hysteretic
agent states and the distinction between prediction and memory. It cannot be
used to claim the paper's two-order-of-magnitude memory improvement or its
linear memory-scaling law. Those claims require the authors' pure-MC protocol,
large populations and ensemble runs.

The `:paper` parameters and a deterministic update match the companion
configuration and `MultiStateBoidFlock.step` at commit `36701bb`; see
`test/lund_model_tests.jl`.

## Claims that are safe in the student tutorial

- The three examples share a software interface for input coupling, state
  evolution, observation and linear readout.
- The Mizzi force law and direct $4N$ observation reproduce the supplied basic
  model, while the paper's MDL optimisation and evaluation are omitted.
- The Lymburn example demonstrates why a permutation-invariant observation
  layer can expose useful collective response.
- The Lund-inspired example demonstrates how a scalar global input and a slow,
  hysteretic internal state can be incorporated into the same pipeline.
- One-step held-out correlations from different inputs, targets and sampling
  intervals are not a ranking of the three models.

Do not describe this repository as reproducing every published result. The
validation notebook should identify each result it actually reproduces and the
tolerance used to judge agreement.

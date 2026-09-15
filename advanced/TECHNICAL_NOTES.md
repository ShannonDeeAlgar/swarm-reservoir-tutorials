# Tutorial technical notes and content audit

This file preserves implementation provenance, reproduction caveats, and
the content audit that led to separate student and specialist notebooks:

- `Tutorial_ABM.ipynb`: student ABM quickstart.
- `Tutorial_ESN.ipynb`: student ESN quickstart.
- `Tutorial_swarmRC.ipynb`: student swarm-reservoir quickstart.
- `advanced/Existing_Model_Validation.ipynb`: validation of published models and full
  reproductions.

## Intended student route

The swarm-reservoir student notebook follows one focused story:

1. Select the Lymburn swarm dynamics, inspect snapshots, render an animation,
   and locate the choice in a saved parameter sweep.
2. Select an input signal and coupling, then compare undriven and driven
   collective behaviour.
3. Select raw positions or spatial Gaussian kernels as the observation.
4. Train the readout, compare forecasting performance, and visualise which
   kernels receive the largest standardised readout weights.
5. Return to the observation layer after introducing TDA and propose a
   topological or hybrid feature map.
6. Use `validate_pipeline` before substituting a new input, swarm, or
   observation.

Basic ESN teaching now lives in `Tutorial_ESN.ipynb`; Jaeger quantitative
reference results remain in `advanced/Existing_Model_Validation.ipynb`.

The parameter sweeps, topology, full paper reproductions, and diagnostic
deep dives are project-specific branches rather than prerequisites.

## Full content audit

### Keep in the main student narrative

- The model-system comparison table and one compact figure showing measured
  signal, known state space, and scalar reconstruction.
- One selected swarm, its animation, and a parameter-map view.
- The explicit component contract and `validate_pipeline`.
- A visible before/after comparison when input coupling is enabled.
- The driven Lymburn trajectory, raw versus spatial-kernel observation, and
  held-out forecasting result.
- One saved parameter-sweep figure as evidence that the implementation
  reproduces more than a single attractive trajectory.
- A clear bridge suggesting TDA as a replacement or addition to the
  observation layer.

### Keep, but label optional

- The ESN sampling/conditioning/ridge failure case. It is a good debugging
  lesson, but too detailed for the first explanation of an ESN.
- Couzin phase sweeps, interaction networks, fluidity, and full paper-scale
  reproductions.
- Ridge-regularisation trade-offs for the Lymburn readout.
- Feature conditioning, ESN baselines, consistency capacity, predator-contact
  analysis, and transient estimation.
- Full Lymburn parameter sweeps and figure reconstruction.

### Move out of the main narrative

- Historical explanations of implementation mistakes that have already been
  fixed.
- Exact MATLAB-versus-physically-symmetric force-cap decisions.
- Unresolved reproduction discrepancies.
- Long discussions of why a particular intermediate diagnostic once looked
  wrong.
- Repeated descriptions of the same train/test split or ridge fit.
- Raw parameter-sweep plumbing and confirmation phrases. The notebook should
  show a small example and the saved result; production execution belongs in
  scripts or a reproducibility document.
- Duplicate network-topology explanations in both the 2D and 3D Couzin paths.

### Remove when the notebook is next regenerated

- Large stale image outputs from optional cells. They account for most of the
  notebook's current size.
- Repeated `include(...)` calls where a section loader already provides the
  dependency.
- Duplicate plots that answer the same question. In particular, the logistic
  map's known return relation and lag-1 reconstruction are the same coordinate
  pair; if both remain, the caption should explicitly say why.
- Narrative statements containing transient numerical results unless those
  numbers are a published validation target or generated in the immediately
  preceding cell.

Before the split, the notebook contained 161 cells (68 code and 93 markdown)
and was about 16 MB. Embedded outputs, rather than source code, were the main
cause of its size. The three-notebook structure now keeps the student route
short while retaining validation evidence and research diagnostics.

## Model-system plotting decisions

The logistic map is discrete. A lag-1 return plot should therefore use points,
not a line that visually implies a continuous path between successive return
pairs.

For the demonstrated Lorenz trajectory, `dt=0.05` and lag 3 give a delay of
`0.15` time units. This retains the two-lobed reconstruction without the
over-unfolded crossing produced by lag 10 (`0.5` time units). This remains an
illustrative choice, not an automatically certified embedding. For measured
data, estimate a lag and embedding dimension and test robustness.

The four-dimensional hyperchaotic Rössler equations and standard parameters
follow the [Scholarpedia hyperchaos reference](https://www.scholarpedia.org/article/Hyperchaos).
That reference reports two positive Lyapunov exponents for this trajectory.
A low-dimensional projection cannot by itself verify hyperchaos.

## Lymburn implementation provenance

### Friction and force cap

The friction term is implemented as a bounded correction proportional to
deviation from cruise speed in the velocity direction, rather than multiplying
the correction by raw velocity magnitude again. The distinction matters when
the total force is subsequently capped.

The released MATLAB applies the cap componentwise. That introduces an
axis-dependent bias. The codebase also supplies an isotropic cap for physically
rotationally symmetric experiments. Exact paper reproduction and physically
preferred modelling are therefore separate configurations and should be named
as such.

### Input scaling

The paper-faithful experiment standardises each Lorenz coordinate to standard
deviation 2. `match_predator_to_lymburn` and
`match_predator_to_couzin` instead match spatial extent, speed, and timestep
to a specific swarm. The latter is preferable for controlled comparisons but
changes the scientific definition of the input adapter and must be reported.

## Open reproduction items

- The naive consistency capacity does not yet match the paper's reported
  magnitude or rank structure, and its dominant consistent direction has not
  reproduced the reported centre-of-mass relationship.
- The smaller high-polarisation performance rise in the original Fig. 8(c)
  has not been recovered; the low/moderate-polarisation region containing
  point B is reproduced.
- Componentwise versus isotropic force caps materially affect results. The
  choice must be recorded alongside any claimed numerical reproduction.

These issues should remain visible in research documentation, but they should
not interrupt the starter workflow.

## Why the scalar Lymburn target is not a free run

The paper's future-\(x\) target is evaluated while the true two-coordinate
predator trajectory continues to drive the swarm. It is a teacher-forced
forecasting task. Autonomous generation with a two-dimensional swarm input
must predict both next coordinates, predict the complete next delay vector, or
provide an explicit stateful output-to-input adapter. `validate_pipeline`
enforces this distinction.

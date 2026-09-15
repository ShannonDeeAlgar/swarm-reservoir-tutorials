# How Physical Reservoirs Compute

Teaching code for student projects on agent-based models, reservoir computing
and topological data analysis.

## Tutorials

- [`Tutorial_ABM.ipynb`](Tutorial_ABM.ipynb): simulate collective motion;
- [`Tutorial_ESN.ipynb`](Tutorial_ESN.ipynb): build a standard echo-state network;
- [`Tutorial_swarmRC.ipynb`](Tutorial_swarmRC.ipynb): use a swarm as a reservoir;
- [`Tutorial_TDA.ipynb`](Tutorial_TDA.ipynb): describe swarm states using persistent homology.

Start with one tutorial and run it from top to bottom. The swarm-reservoir
tutorial is the main entry point for the current student projects.

[`advanced/Existing_Model_Validation.ipynb`](advanced/Existing_Model_Validation.ipynb)
contains longer checks against published results. It is not needed for a first
run.

## Setup

Install [Julia](https://julialang.org/install/), clone this whole repository,
and open a terminal in the repository folder. Run:

```sh
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Start Julia or Jupyter from the same folder using this project environment.
The first run will be slow while Julia compiles the plotting packages.

The notebooks are not standalone files. They use `Project.toml`,
`Manifest.toml` and the source folders in this repository, so keep the folder
structure unchanged.

## Swarm-reservoir tutorial

The tutorial contains presets for three models:

- **Lymburn:** interacting agents driven by a moving predator;
- **Mizzi:** territorial agents driven by moving prey;
- **Lund:** two-state boids driven by a global temperature signal.

Lund has a short teaching preset and `:lund_paper`, which matches the companion
repository's default dynamics but not its memory-capacity experiment.

Choose a preset in the first control cell, restart Julia and use **Run All**.
Use `run_mode=:results_only` for a first run. `run_mode=:full` also generates
animations and takes longer.

Approximate clean-run times on the machine used for testing are:

| preset | results only | full |
|---|---:|---:|
| Lymburn | 3 min | 39 min |
| Mizzi | 2 min | 26 min |
| Lund | 3 min | 13 min |
| Lund paper dynamics | 3 min | 8 min |

Generated figures and animations are saved locally and are not generally kept
in the repository.

## Scientific sources

The tutorials retain the main mechanisms of the published models, but they do
not reproduce every experiment in the papers. The differences are recorded in
[`MODEL_IMPLEMENTATION_COMPARISON.md`](MODEL_IMPLEMENTATION_COMPARISON.md).
See [`REFERENCES.md`](REFERENCES.md) for the papers to cite.

## Licence and acknowledgement

The code is by Shannon Dee Algar and is released under the standard
[MIT licence](LICENSE). ChatGPT and Claude assisted at different stages with
editing, review and testing.

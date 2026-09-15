# ============================================================
# load_ABM.jl
# Convenience loader for the generic (model-agnostic) ABM machinery.
# ============================================================
#
# Assumes this file lives inside the ABM/ folder. To simulate a specific
# model (e.g. Couzin), also include its own loader, e.g.
# models/Couzin/load_Couzin.jl, after this one.
#
# Usage from ScratchPad.ipynb, assuming ScratchPad.ipynb is beside ABM/:
#
#     using Revise
#     include("ABM/load_ABM.jl")
#
# ============================================================

# --- core: needed to run and analyse any swarm model ---
include("my_ABM_helpers.jl")     # vector/periodic-boundary utilities, noise modes
include("my_ABM_core.jl")        # SwarmState, SimulationConfig, generic `simulate` driver
include("my_ABM_analysis.jl")    # order parameters, clustering

# --- optional / advanced: not required to run a first simulation ---
include("my_ABM_networks.jl")    # interaction-graph statistics
include("my_ABM_plotting.jl")    # animations
include("my_ABM_phase_utils.jl") # phase-diagram utilities

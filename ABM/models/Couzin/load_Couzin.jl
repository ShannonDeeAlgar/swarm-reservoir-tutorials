# ============================================================
# load_Couzin.jl
# Convenience loader for the Couzin swarm model.
# ============================================================
#
# Assumes this file lives inside models/Couzin/, and that ABM/load_ABM.jl
# has already been included (this file depends on SwarmState, CouzinNoise's
# NoiseMode, SimulationConfig, order_parameters, ... from the generic ABM
# machinery).
#
# Usage from ScratchPad.ipynb, assuming ScratchPad.ipynb is beside ABM/:
#
#     using Revise
#     include("ABM/load_ABM.jl")
#     include("ABM/models/Couzin/load_Couzin.jl")
#
# ============================================================

# --- core: needed to build parameters and run a simulation ---
include("my_Couzin_core.jl")       # CouzinParams, regime presets
include("my_Couzin_scaling.jl")    # nondimensionalise!, scale_time!/scale_space!/scale_swarm_dynamics!
include("my_Couzin_Rules2d.jl")    # 2D step logic (also defines simulate_Couzin_2d)
include("my_Couzin_Rules3d.jl")    # 3D step logic

# --- optional / advanced: not required to run a first simulation ---
include("my_Couzin_experiments.jl") # parameter-sweep helpers
include("my_Couzin_phase.jl")       # phase-diagram utilities
include("my_Couzin_networks.jl")    # interaction-graph statistics
include("my_Couzin_plotting.jl")    # animations
include("my_Couzin_diagnostics.jl") # per-agent rule breakdowns

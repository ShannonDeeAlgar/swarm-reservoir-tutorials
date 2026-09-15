# ============================================================
# load_Lymburn.jl
# Convenience loader for the Lymburn swarm model (Lymburn et al., Chaos
# 31, 033121, 2021).
# ============================================================
#
# Assumes ABM/load_ABM.jl has already been included (this file depends on
# SwarmState, SimulationConfig, simulate, has_order, ... from the generic
# ABM machinery).
#
# Usage:
#
#     include("ABM/load_ABM.jl")
#     include("ABM/models/Lymburn/load_Lymburn.jl")
#
# ============================================================

include("my_Lymburn_core.jl")          # LymburnParams, presets
include("my_Lymburn_diagnostics.jl")   # paper-exact order parameters (needed by Rules2d's simulate_Lymburn_2d)
include("my_Lymburn_Rules2d.jl")       # 2D step logic (also defines simulate_Lymburn_2d)

# --- optional / advanced: not required to run a first simulation ---
# include("my_Lymburn_experiments.jl") # parameter-sweep helpers
# include("my_Lymburn_plotting.jl")    # animations

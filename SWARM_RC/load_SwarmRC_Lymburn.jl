# ============================================================
# load_SwarmRC_Lymburn.jl
# Convenience loader that adds the Lymburn swarm-reservoir pipeline
# (Lymburn et al., Chaos 2021) on top of load_SwarmRC.jl's Couzin path.
# ============================================================
#
# `my_swarmRC.jl` (where `KernelLayer` and the shared observation-layer
# machinery live) already depends on Couzin's types, so this doesn't try
# to load Lymburn in isolation -- it loads the existing Couzin pipeline
# first (unmodified), then adds Lymburn's model and reservoir on top. Both
# models are available together after this, which is the intended setup
# (Couzin remains available as the alternative model).
#
# Usage:
#
#     include("SWARM_RC/load_SwarmRC_Lymburn.jl")
#
#     P = Lymburn_params_from_preset(:driven; N=200)
#     res = build_Lymburn_reservoir(P, 0.02)
#
# ============================================================

include("load_SwarmRC.jl")                       # ABM + RC + Couzin + my_swarmRC.jl (KernelLayer, etc.)
include("../ABM/models/Lymburn/load_Lymburn.jl")  # LymburnParams, simulate_Lymburn_2d
include("my_swarmRC_lymburn.jl")                  # LymburnReservoir + dispatch methods

# --- optional / advanced: not required for a first baseline ---
# include("my_swarmRC_plotting.jl")  # animations, kernel visualisation (needs CairoMakie)
# include("my_predator.jl")          # chaotic (e.g. Lorenz-driven) predator input + scale matching

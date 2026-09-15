# ============================================================
# load_SwarmRC.jl
# Convenience loader for the swarm-reservoir pipeline (Theme 1 baseline).
# ============================================================
#
# Assumes this file lives inside SWARM_RC/, beside ABM/ and RC/ (i.e. all
# under a common RESEARCH/ parent directory).
#
# Usage:
#
#     include("SWARM_RC/load_SwarmRC.jl")
#
#     scenario = Couzin_params_from_preset(:milling; nondim=true, dt=0.1)
#     res = build_Couzin_reservoir(scenario.P, scenario.dt;
#                                   coupling=PredatorCoupling(rp=3.0))
#
# See the worked example at the bottom of my_swarmRC.jl for the full
# train/evaluate pipeline (Project T1-S1).
# ============================================================

include("../ABM/load_ABM.jl")
include("../ABM/models/Couzin/load_Couzin.jl")
include("../RC/my_reservoir_core.jl")
include("my_swarmRC.jl")
include("my_pipeline_validation.jl")

# --- optional / advanced: not required for a first baseline ---
# include("my_swarmRC_plotting.jl")  # animations, kernel visualisation (needs CairoMakie)
# include("my_predator.jl")          # chaotic (e.g. Lorenz-driven) predator input + scale matching

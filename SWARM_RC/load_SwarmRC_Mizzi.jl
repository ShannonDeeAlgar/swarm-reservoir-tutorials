# Convenience loader for the Mizzi territorial-agent reservoir.
# Loads the shared reservoir interface and input-coupling types through the
# existing shared path, then adds the Mizzi ABM and wrapper.
include("load_SwarmRC.jl")
include("../ABM/models/Mizzi/load_Mizzi.jl")
include("my_swarmRC_mizzi.jl")

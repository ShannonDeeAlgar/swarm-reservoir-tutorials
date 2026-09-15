# Persistent-homology observation layer for the generic reservoir interface.

using Random
using Statistics

struct TDAObservationLayer
    εs::Vector{Float64}
    dims::Vector{Int}
    representation::Symbol
    standardize::Bool
    periodic::Bool
    L::Union{Nothing,Float64}
    min_lifetime::Float64
end

function TDAObservationLayer(εs::AbstractVector{<:Real}; dims=0:1,
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing, min_lifetime::Real=0.0)
    grid = Float64.(εs)
    length(grid) >= 2 || throw(ArgumentError("A TDA observation needs at least two ε samples."))
    issorted(grid) || throw(ArgumentError("εs must be sorted."))
    first(grid) >= 0 || throw(ArgumentError("εs cannot contain negative scales."))
    dimensions = Int.(collect(dims))
    isempty(dimensions) && throw(ArgumentError("dims cannot be empty."))
    minimum(dimensions) >= 0 || throw(ArgumentError("homology dimensions must be nonnegative."))
    representation in (:position, :position_velocity) || throw(ArgumentError(
        "representation must be :position or :position_velocity."))
    periodic && representation != :position && throw(ArgumentError(
        "Periodic TDA currently supports representation=:position only."))
    periodic && L === nothing && throw(ArgumentError("Periodic TDA requires L."))
    return TDAObservationLayer(grid, dimensions, representation, standardize,
        periodic, L === nothing ? nothing : Float64(L), Float64(min_lifetime))
end

if isdefined(@__MODULE__, :AbstractReservoir)
    @eval function feature_map(res::AbstractReservoir, obs::TDAObservationLayer)
        return tda_feature_map(res; εs=obs.εs, dims=obs.dims,
            representation=obs.representation, standardize=obs.standardize,
            periodic=obs.periodic, L=obs.L, min_lifetime=obs.min_lifetime)
    end
end

"""
    build_tda_observation_layer!(res, Utrain; ...)

Calibrate a fixed ε grid from a short, evenly sampled teacher-forced trajectory.
The returned layer is immutable and is reused unchanged for training and test.
Calibration uses a cloned reservoir and does not alter `res`.
"""
function build_tda_observation_layer!(res, Utrain::AbstractMatrix;
    washout::Int=0, rng::AbstractRNG=Random.default_rng(), dims=0:1,
    representation::Symbol=:position, standardize::Bool=false,
    periodic::Bool=false, L=nothing, n_epsilon::Int=24,
    epsilon_quantile::Real=0.75, calibration_steps::Int=120,
    min_lifetime::Real=0.0, show_progress::Bool=false)
    0 < epsilon_quantile <= 1 || throw(ArgumentError("epsilon_quantile must lie in (0,1]."))
    n_epsilon >= 2 || throw(ArgumentError("n_epsilon must be at least 2."))
    calibration_steps >= 1 || throw(ArgumentError("calibration_steps must be positive."))
    size(Utrain, 2) > washout || throw(ArgumentError("washout consumes the calibration input."))

    probe = clone_reservoir(res)
    reset!(probe; rng=rng)
    last_t = size(Utrain, 2)
    sample_idxs = unique(round.(Int,
        range(washout + 1, last_t; length=min(calibration_steps, last_t - washout))))
    sample_set = Set(sample_idxs)
    distances = Float64[]
    for t in 1:last_t
        reservoir_step!(probe, view(Utrain, :, t); rng=rng)
        if t in sample_set
            D = swarm_dissimilarity(probe; representation, standardize, periodic, L)
            append!(distances, D[D .> 0])
        end
    end
    isempty(distances) && throw(ArgumentError(
        "TDA calibration found no positive inter-agent distances."))
    εmax = quantile(distances, epsilon_quantile)
    εmax > 0 || throw(ArgumentError("The calibrated ε range is degenerate."))
    return TDAObservationLayer(range(0, εmax; length=n_epsilon); dims,
        representation, standardize, periodic, L, min_lifetime)
end

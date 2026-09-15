# my_time_series_analysis.jl
"""
takens_embed(x; m=2, τ=1, horizon=1, start=1, stop=nothing)

Offline Takens delay embedding from scalar series x.

Returns (U, Y, idx) where:
- U is m×T_eff with U[j,k] = x[t - (j-1)τ]
- Y is 1×T_eff with Y[1,k] = x[t + horizon]   (or x[t] if horizon=0)
- idx is the original t indices used.

Set `horizon=0` if you want Y == x[t].
"""
function takens_embed(x::AbstractVector;
    m::Int = 2,
    τ::Int = 1,
    horizon::Int = 1,
    start::Int = 1,
    stop::Union{Nothing,Int} = nothing,
)
    @assert m ≥ 1 "m must be ≥ 1"
    @assert τ ≥ 1 "τ must be ≥ 1"
    @assert horizon ≥ 0 "horizon must be ≥ 0"
    n = length(x)
    @assert n ≥ (m-1)*τ + 1 + horizon "time series too short for (m, τ, horizon)"

    tmin = 1 + (m-1)*τ
    tmax = n - horizon
    @assert tmin ≤ tmax "no valid indices for these (m, τ, horizon)"

    tstart = max(start, tmin)
    tstop  = min(something(stop, tmax), tmax)
    @assert tstart ≤ tstop "invalid start/stop after clamping"

    idx = tstart:tstop
    T_eff = length(idx)

    U = Matrix{Float64}(undef, m, T_eff)
    Y = Matrix{Float64}(undef, 1, T_eff)

    @inbounds for (k, t) in enumerate(idx)
        for j in 1:m
            U[j, k] = float(x[t - (j-1)*τ])
        end
        Y[1, k] = horizon == 0 ? float(x[t]) : float(x[t + horizon])
    end

    return U, Y, idx
end

# takens_embed_U(x::AbstractVector; kwargs...) = (takens_embed(x; kwargs...))[1]


"""
takens_coords_from_scalar!(hist, u; m, τ)

Online Takens embedding update.

Arguments:
- hist :: Vector{Float64}   length = 1 + (m-1)*τ
- u    :: Float64           new scalar input

Returns:
- coords :: Vector{Float64} length = m
    [x(t), x(t-τ), x(t-2τ), ...]
"""
@inline function takens_coords_from_scalar!(
    hist::Vector{Float64},
    u::Float64;
    m::Int,
    τ::Int,
)
    @assert length(hist) == 1 + (m-1)*τ
    @assert m >= 2 "Need m≥2 to return (u1,u2)."
    @assert τ >= 1

    # shift history right: hist[1] is MOST RECENT
    @inbounds for k in length(hist):-1:2
        hist[k] = hist[k-1]
    end
    @inbounds hist[1] = u

    # standard/latest-first coords:
    # u1 = x_t, u2 = x_{t-τ}
    @inbounds begin
        u1 = hist[1]
        u2 = hist[1 + τ]
        return u1, u2
    end
end

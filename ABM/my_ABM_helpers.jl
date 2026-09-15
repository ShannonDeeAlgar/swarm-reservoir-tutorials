# NoiseMode
# wrap_angle

# signed_angle
# rot2

# periodic_mean
# periodic_centre_of_mass_2d
# periodic_centre_of_mass_3d
# periodic_disp

# τ

# encode_xy_to_sincos
# decode_sincos_to_xy

# step_lengths
# mean_agent_displacement
# median_agent_displacement
# mean_agent_speed


# ============================================================
# External packages used by generic ABM helpers
# ============================================================

using LinearAlgebra
using Random
using Statistics
using StaticArrays

# ============================================================
# Shared project-level helpers
# ============================================================

include("../HELPERS/my_geometry_helpers.jl")
include("../HELPERS/my_plot_helpers.jl")

# signed angle from u to v in 2D
function signed_angle(u::SVector2, v::SVector2)
    det = u.x * v.y - u.y * v.x
    d   = u.x * v.x + u.y * v.y
    return atan(det, d)
end

function rot2(v::SVector2, θ::Real)
    c, s = cos(θ), sin(θ)
    return SVector2(c * v.x - s * v.y, s * v.x + c * v.y)
end

@enum NoiseMode begin
    NoNoise
    WrappedStepGaussian
    Diffusion
end

@inline wrap_angle(θ::Float64) = mod(θ + π, 2π) - π


# ============================================================
# Generic periodic helpers
# ============================================================
function periodic_mean(xs::Vector{Float64}, L::Float64)
    angles = (2π / L) .* xs
    c = mean(cos.(angles))
    s = mean(sin.(angles))
    θ = atan(s, c)
    θ < 0 && (θ += 2π)
    return (L / (2π)) * θ
end

function periodic_centre_of_mass_2d(pos::Vector{SVector2}, L::Real)
    xs = Vector{Float64}(undef, length(pos))
    ys = Vector{Float64}(undef, length(pos))
    @inbounds for i in eachindex(pos)
        xs[i] = pos[i].x
        ys[i] = pos[i].y
    end
    return SVector2(periodic_mean(xs, L), periodic_mean(ys, L))
end

function periodic_centre_of_mass_3d(pos::Vector{SVector3}, L::Real)
    xs = Vector{Float64}(undef, length(pos))
    ys = Vector{Float64}(undef, length(pos))
    zs = Vector{Float64}(undef, length(pos))
    @inbounds for i in eachindex(pos)
        xs[i] = pos[i].x
        ys[i] = pos[i].y
        zs[i] = pos[i].z
    end
    return SVector3(
        periodic_mean(xs, L),
        periodic_mean(ys, L),
        periodic_mean(zs, L),
    )
end

@inline function periodic_disp(x1, y1, x2, y2, L)
    dx = x2 - x1
    dy = y2 - y1
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    return dx, dy
end

const τ = 2π

encode_xy_to_sincos(U, L) = begin
    @assert size(U,1) == 2
    x = @view U[1,:]
    y = @view U[2,:]
    X = τ .* x ./ L
    Y = τ .* y ./ L
    vcat(cos.(X)', sin.(X)', cos.(Y)', sin.(Y)')
end

decode_sincos_to_xy(y, L) = begin
    θx = atan(y[2], y[1])
    θy = atan(y[4], y[3])
    x = mod(L * θx / τ, L)
    yv = mod(L * θy / τ, L)
    SVector2(x, yv)
end


@inline function step_lengths(prev_pos::Vector{SVector2},
                              curr_pos::Vector{SVector2},
                              L::Real)
    @assert length(prev_pos) == length(curr_pos)
    steps = Vector{Float64}(undef, length(prev_pos))
    @inbounds for i in eachindex(prev_pos)
        dx, dy = periodic_disp(prev_pos[i].x, prev_pos[i].y,
                               curr_pos[i].x, curr_pos[i].y, L)
        steps[i] = hypot(dx, dy)
    end
    return steps
end


@inline mean_agent_displacement(prev_pos, curr_pos, L) =
    mean(step_lengths(prev_pos, curr_pos, L))

@inline median_agent_displacement(prev_pos, curr_pos, L) =
    median(step_lengths(prev_pos, curr_pos, L))

@inline function mean_agent_speed(vel::AbstractVector)
    s = 0.0
    @inbounds for v in vel
        s += norm(v)
    end
    return s / length(vel)
end

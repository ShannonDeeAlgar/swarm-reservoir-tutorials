# ============================================================
# my_Lymburn_diagnostics.jl
# Order parameters exactly as defined in the paper (Eqs 19-20). These do
# NOT match the generic/Couzin order_parameters_2d formulas -- in
# particular the rotation metric is the mean *sign* of each agent's
# angular momentum (Eq 20), not the mean tangential-alignment used
# elsewhere in this codebase.
# ============================================================

using Statistics
using LinearAlgebra

"""
    Lymburn_order_parameters_2d(pos, vel, L; displacement) -> (dilation, rotation, polarisation, abs_angular_momentum)

Matches the generic 4-tuple `order_parameters` interface expected by
`simulate` (`ABM/my_ABM_core.jl`). `rotation` and `polarisation` are the
paper's exact Φ_R (Eq 20) and Φ_P (Eq 19); `dilation`/`abs_angular_momentum`
are computed with the same generic formulas used elsewhere in this
codebase (not paper-defined, but harmless extra diagnostics). `L` and
`displacement` are accepted for interface compatibility but the centre of
mass here is a plain (non-periodic) mean, since this model has no domain.
"""
function Lymburn_order_parameters_2d(pos::Vector{SVector2}, vel::Vector{SVector2}, L; displacement)
    N = length(pos)
    N == 0 && return (0.0, 0.0, 0.0, 0.0)

    mx = 0.0
    my = 0.0
    @inbounds for i in 1:N
        v̂ = unit(vel[i])
        mx += v̂.x
        my += v̂.y
    end
    polarisation = sqrt((mx / N)^2 + (my / N)^2)   # Eq 19, Φ_P

    cx = mean(p.x for p in pos)
    cy = mean(p.y for p in pos)
    c = SVector2(cx, cy)

    rot_sign_sum = 0.0
    dil_sum = 0.0
    count = 0
    abs_ang_num = 0.0
    abs_ang_den = 0.0

    @inbounds for i in 1:N
        r = pos[i] - c   # x_{C,i}, Eq 20
        ρ = norm(r)
        v = vel[i]
        vmag = norm(v)
        crossz = r.x * v.y - r.y * v.x

        if ρ >= 1e-9
            r̂ = r / ρ
            v̂ = unit(v)
            dil_sum += dot(v̂, r̂)
            count += 1
        end

        if abs(crossz) >= 1e-12
            rot_sign_sum += sign(crossz)
        end

        if ρ >= 1e-9 && vmag >= 1e-12
            abs_ang_num += abs(crossz)
            abs_ang_den += ρ * vmag
        end
    end

    rotation = abs(rot_sign_sum / N)   # Eq 20, Φ_R
    dilation = count > 0 ? dil_sum / count : 0.0
    abs_angular_momentum = abs_ang_den > 0 ? abs_ang_num / abs_ang_den : 0.0

    return (dilation, rotation, polarisation, abs_angular_momentum)
end

"""
    mean_Lymburn_order_parameters(out; transient_steps=0)

Time-average of Φ_P (`polarisation`) and Φ_R (`rotation`) after discarding
an initial transient -- matches the paper's Sec III methodology (transient
of 1000 steps out of 5x10^4 total).
"""
function mean_Lymburn_order_parameters(out; transient_steps::Int = 0)
    @assert has_order(out) "Cannot compute averages: order parameters were not recorded."
    i0 = min(transient_steps + 1, length(out.t))
    return (
        polarisation_mean = mean(@view out.polarisation[i0:end]),
        polarisation_std  = std(@view out.polarisation[i0:end]),
        rotation_mean     = mean(@view out.rotation[i0:end]),
        rotation_std      = std(@view out.rotation[i0:end]),
        transient_index   = i0,
        transient_time    = out.t[i0],
    )
end

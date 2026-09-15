# ============================================================
# my_Lymburn_core.jl
# Lymburn et al. (Chaos 31, 033121, 2021) swarm model: a modified
# Reynolds-boids flock with a global homing force in place of a periodic
# domain -- see the paper's Eqs 1-9 for the model and Eq 10/11 for the
# predator-driven variant.
# ============================================================
#
# This file assumes the generic ABM machinery already provides:
#   - SVector2
#   - SwarmState
#   - SimulationConfig
#   - simulate
#
# Rules live in my_Lymburn_Rules2d.jl; order parameters in
# my_Lymburn_diagnostics.jl.

using Random

# ============================================================
# LymburnParams
# ============================================================
#
# There is no domain size L and no periodic boundary in this model -- the
# swarm is held together by a global homing force to xh instead (Eq 3).
# `plot_extent` is purely cosmetic (axis limits for plotting) and plays no
# role in the dynamics.

Base.@kwdef struct LymburnParams
    N::Int              = 200
    Kr::Float64          = 1.0     # repulsion strength
    Ka::Float64          = 0.1     # alignment strength
    Kh::Float64          = 2.0     # homing strength
    Kf::Float64          = 20.0    # friction strength
    Kp::Float64          = 0.0     # predator strength (0 = undriven / "safe world")
    rr::Float64          = 1.0     # repulsion radius
    ra::Float64          = 1.0     # alignment radius
    rp::Float64          = 2.0     # predator interaction radius
    s::Float64           = 10.0    # target cruising speed (friction force)
    alpha::Float64       = 200.0   # tanh force-cap amplitude (Eq 6)
    beta::Float64        = 0.1     # tanh force-cap steepness (Eq 6)
    noise_amp::Float64   = 0.0     # per-step Gaussian force noise (std), 0 = deterministic; present but 0 in the paper's own default params
    cap_mode::Symbol     = :componentwise   # :componentwise (matches the paper's released MATLAB) or :isotropic (rotationally-symmetric alternative)
    integration_scheme::Symbol = :semi_implicit   # :semi_implicit (matches the paper's released MATLAB) or :explicit
    xh::SVector2         = SVector2(0.0, 0.0)   # home position
    plot_extent::Float64 = 20.0    # nominal half-range for plotting only
end

function update_LymburnParams(P::LymburnParams;
    N::Int = P.N,
    Kr::Float64 = P.Kr, Ka::Float64 = P.Ka, Kh::Float64 = P.Kh,
    Kf::Float64 = P.Kf, Kp::Float64 = P.Kp,
    rr::Float64 = P.rr, ra::Float64 = P.ra, rp::Float64 = P.rp,
    s::Float64 = P.s, alpha::Float64 = P.alpha, beta::Float64 = P.beta,
    noise_amp::Float64 = P.noise_amp, cap_mode::Symbol = P.cap_mode,
    integration_scheme::Symbol = P.integration_scheme,
    xh::SVector2 = P.xh, plot_extent::Float64 = P.plot_extent,
)
    return LymburnParams(
        N=N, Kr=Kr, Ka=Ka, Kh=Kh, Kf=Kf, Kp=Kp, rr=rr, ra=ra, rp=rp,
        s=s, alpha=alpha, beta=beta, noise_amp=noise_amp, cap_mode=cap_mode,
        integration_scheme=integration_scheme, xh=xh, plot_extent=plot_extent,
    )
end

# ============================================================
# Presets
# ============================================================
#
# :baseline / :driven are the paper's Fig 1/2 parameters (Sec IIB): the
# same force weights and radii, differing only in Kp (0 = undriven,
# 100 = driven by the Lorenz-scaled predator).
#
# :critical is a different point, from the paper's own (Kr, Ka) sweep
# (Sec III, Figs 6-9) rather than its Fig 1/2 illustration: labelled "B" in
# Fig 7/8, their identified best-performing regime -- "the swarm's ability
# to respond to the predator while maintaining cohesive and at least
# locally ordered motion with 'interesting' oscillations and pulsations."
# Their broader rule of thumb is a triangular region Ka < Kr < Kp (repulsion,
# which decorrelates agents into a diverse reservoir, dominating alignment,
# which correlates them into rigid, low-diversity lockstep -- their point
# "D", Ka=10 > Kr=1, is the cautionary opposite case). :baseline/:driven's
# Kr=1, Ka=0.1 already technically satisfies Ka<Kr<Kp, but isn't tuned to
# their specific best point; :critical is.

# All three presets pin cap_mode=:isotropic, overriding LymburnParams's own
# bare default of :componentwise. Confirmed against the paper's released
# MATLAB, :componentwise is what actually generated the published figures
# -- but it caps a force pointing along a Cartesian axis differently than
# one at 45°, which has no physical justification for a flocking model and
# produces a measurable, spurious preference for N/S/E/W motion tied to an
# arbitrary choice of coordinate axes (visible as a cross-shaped, not
# radially symmetric, occupancy pattern -- see the interaction-density
# figures in `Tutorial_swarmRC.ipynb`). These presets favour a physically
# well-motivated default for the DP program's own use over exact numerical
# reproduction of the paper; pass `cap_mode=:componentwise` explicitly
# (e.g. `Lymburn_params_from_preset(:critical; cap_mode=:componentwise)`)
# when you specifically want to match the paper's literal numbers.
const LYMBURN_PRESETS = Dict(
    :baseline => (Kr=1.0, Ka=0.1,  Kh=2.0, Kf=20.0, Kp=0.0,   rr=1.0, ra=1.0, rp=2.0, cap_mode=:isotropic),
    :driven   => (Kr=1.0, Ka=0.1,  Kh=2.0, Kf=20.0, Kp=100.0, rr=1.0, ra=1.0, rp=2.0, cap_mode=:isotropic),
    :critical => (Kr=2.0, Ka=0.01, Kh=2.0, Kf=20.0, Kp=100.0, rr=1.0, ra=1.0, rp=2.0, cap_mode=:isotropic),
)

"""
    Lymburn_params_from_preset(preset; N=200, kwargs...) -> LymburnParams

`preset` is `:baseline` (undriven, Kp=0, the paper's Fig 1/2 parameters),
`:driven` (same parameters, Kp=100, matches the paper's "risky world"
scenario), or `:critical` (Kp=100, `Kr=2, Ka=0.01` -- the paper's own
Fig 7/8 "point B", their identified best-performing, "dynamically rich"
regime from the Sec III parameter sweep, distinct from the Fig 1/2 demo
parameters `:baseline`/`:driven` use). Any field of `LymburnParams` can be
overridden via `kwargs`, e.g. `Lymburn_params_from_preset(:driven; N=50)`
for a faster/smaller demo swarm.
"""
function Lymburn_params_from_preset(preset::Symbol; N::Int = 200, kwargs...)
    haskey(LYMBURN_PRESETS, preset) || error(
        "Unknown Lymburn preset = $preset. Valid presets are $(sort(collect(keys(LYMBURN_PRESETS))))."
    )
    vals = LYMBURN_PRESETS[preset]
    P = LymburnParams(; N=N, vals...)
    return isempty(kwargs) ? P : update_LymburnParams(P; kwargs...)
end

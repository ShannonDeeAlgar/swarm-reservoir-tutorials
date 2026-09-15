using OrdinaryDiffEq
using Random
using Distributions: Uniform #selective import used to avoid Distributions.params clash
using CairoMakie

# function lorenz!(du, u, p, t)
#     σ, ρ, β = p
#     du[1] = σ * (u[2] - u[1])
#     du[2] = u[1] * (ρ - u[3]) - u[2]
#     du[3] = u[1] * u[2] - β * u[3]
# end



function lorenz!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ*(y - x)
    du[2] = x*(ρ - z) - y
    du[3] = x*y - β*z
end


function lorenz_data(;
    rng::AbstractRNG = Random.default_rng(),
    dtmax::Float64 = 0.01,
    dt_data::Float64 = 0.2,          # <- keep this as your preferred plotting/sample step
    tspan = (0.0, 40.0),
    p = (10.0, 28.0, 8/3),
    u0 = (
        rand(rng, Uniform(-15, 15)),
        rand(rng, Uniform(-20, 20)),
        rand(rng, Uniform(5, 40))
    ),
)
    prob = ODEProblem(lorenz!, collect(u0), tspan, collect(p))

    sol = solve(prob, Tsit5();
        dtmax      = dtmax,
        abstol     = 1e-9,
        reltol     = 1e-9,
        save_start = true,
        maxiters   = 10^8
        # no saveat -> keep full adaptive-step trajectory
    )


    data = Array(sol)   # 3×T_adaptive
    t    = sol.t        # length T_adaptive

    return (data=data, t=t, sol=sol, dt_data=dt_data, labels=["x(t)", "y(t)", "z(t)"])
end


"""
view_timeseries_and_state_space(lor; show_z_ts=false, fig_size=(1200,600), linewidth=0.5, colormap=:viridis)

Uses lor.dt_data as the uniform plotting step. If `lor.sol` is present, it will
interpolate onto that grid for nice plots (even if lor.data is adaptive).
"""
function view_timeseries_and_state_space(lor;
    show_z_ts::Bool = false,
    fig_size::Tuple{Int,Int} = (1200, 600),
    linewidth::Float64 = 0.5,
    colormap = :viridis,
)
    dt = lor.dt_data
    t0, tf = lor.sol.prob.tspan  # robust even if lor.t isn't uniform
    tgrid = collect(t0:dt:tf)

    # Prefer interpolation if we have the solution object
    if hasproperty(lor, :sol) && lor.sol !== nothing
        X = Array(lor.sol(tgrid))  # 3×length(tgrid)
        t = tgrid
    else
        # Fallback: assume data is already uniform at dt_data
        X = lor.data
        T = size(X, 2)
        t = (0:T-1) .* dt
    end

    x, y, z = X[1, :], X[2, :], X[3, :]

    fig = Figure(size = fig_size)

    ax1 = Axis(fig[1, 1],
        xlabel = "time",
        ylabel = "value",
        title  = "Lorenz time series"
    )
    lines!(ax1, t, x, label = "x(t)")
    lines!(ax1, t, y, label = "y(t)")
    show_z_ts && lines!(ax1, t, z, label = "z(t)")
    axislegend(ax1; position = :rt)

    ax2 = Axis3(fig[1, 2],
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        title  = "Lorenz attractor (3D trajectory)"
    )
    plt = lines!(ax2, x, y, z;
        color = t,
        colormap = colormap,
        colorrange = (t[1], t[end]),
        linewidth = linewidth
    )
    Colorbar(fig[1, 3], plt; label = "time")

    colgap!(fig.layout, 15)
    colsize!(fig.layout, 1, Relative(0.45))
    colsize!(fig.layout, 2, Relative(0.45))
    colsize!(fig.layout, 3, Relative(0.10))

    return fig
end


function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] = x + a*y
    du[3] = b + z*(x - c)
end


"""
    rossler_data(; ...)

Standard three-dimensional Rössler flow. Compared with Lorenz, its attractor
has one dominant rotation and a localized nonlinear fold, making it a useful
first continuous-time chaotic system for phase portraits, return maps, and
delay-embedding exercises.
"""
function rossler_data(;
    dtmax::Float64 = 0.01,
    dt_data::Float64 = 0.1,
    tspan = (0.0, 200.0),
    p = (0.2, 0.2, 5.7),
    u0 = (0.1, 0.0, 0.0),
)
    prob = ODEProblem(rossler!, collect(u0), tspan, collect(p))
    sol = solve(prob, Tsit5();
        dtmax=dtmax, abstol=1e-9, reltol=1e-9, saveat=dt_data,
        save_start=true, maxiters=10^8)
    return (
        data=Array(sol), t=sol.t, sol=sol, dt_data=dt_data,
        labels=["x(t)", "y(t)", "z(t)"],
    )
end


function hyper_rossler!(du, u, p, t)
    a, b, c, d = p
    x, y, z, w = u
    du[1] = -y - z
    du[2] = x + a*y + w
    du[3] = b + x*z
    du[4] = -c*z + d*w
end


"""
    hyper_rossler_data(; ...)

Rössler's four-dimensional hyperchaotic flow

    ẋ=-y-z,  ẏ=x+ay+w,  ż=b+xz,  ẇ=-cz+dw

at the standard hyperchaotic parameters `(a,b,c,d)=(0.25,3,0.5,0.05)`.
Hyperchaos means at least two positive Lyapunov exponents. A projection can
look deceptively like noisy ordinary chaos, so use this when a method should
face genuinely higher-dimensional dynamics; do not infer hyperchaos from a
phase portrait alone.
"""
function hyper_rossler_data(;
    dtmax::Float64 = 0.005,
    dt_data::Float64 = 0.05,
    tspan = (0.0, 400.0),
    p = (0.25, 3.0, 0.5, 0.05),
    u0 = (-10.0, -6.0, 0.0, 10.0),
)
    prob = ODEProblem(hyper_rossler!, collect(u0), tspan, collect(p))
    sol = solve(prob, Tsit5();
        dtmax=dtmax, abstol=1e-9, reltol=1e-9, saveat=dt_data,
        save_start=true, maxiters=10^8)
    return (
        data=Array(sol), t=sol.t, sol=sol, dt_data=dt_data,
        labels=["x(t)", "y(t)", "z(t)", "w(t)"],
    )
end



function logistic_data(; r=3.9, x0=0.2, T=10000)
    x = zeros(Float64, T)
    x[1] = x0
    for t in 1:T-1
        x[t+1] = r*x[t]*(1-x[t])
    end
    data = reshape(x, 1, :)
    return (data=data, dt_data=1.0, labels=["x(t)"])
end


"""
    mackey_glass_data(; tau=17, n_out=3000, discard=1000, y0=1.2,
                       alpha=0.2, beta=10.0, gamma=0.1, delta=0.1, subsample=10)

Mackey-Glass delay differential equation, discretized exactly as Jaeger
(2001) eqns (22)-(23) ("The 'echo state' approach...", GMD Report 148):

    ẏ(t) = α y(t-τ) / (1 + y(t-τ)^β) - γ y(t)                          (22)
    y(n+1) = y(n) + δ [α y(n-τ/δ) / (1 + y(n-τ/δ)^β) - γ y(n)]          (23)

with δ=0.1, then subsampled by 10 so that one *output* step corresponds
to one unit time interval of the continuous system -- matching Jaeger's
"one step from n to n+1 in the resulting sequences corresponds to a unit
time interval [t,t+1] of the original continuous system". τ=17 gives the
"mildly chaotic" attractor (Jaeger: "the majority of studies" use this
value); τ=30 gives the "wilder" one.

History for t<0 is the constant `y0=1.2` (the standard Mackey-Glass
starting condition). `discard` output steps are dropped after that
history region as burn-in before the first returned point, on top of
`n_out` -- generates `discard+n_out` output points internally and returns
only the last `n_out`.

Returns data in the *original* Mackey-Glass coordinates (not Jaeger's
[-1,1]-squashed network I/O convention -- see `mg_squash`/`mg_unsquash`
for that transform, applied separately where needed).
"""
function mackey_glass_data(;
    tau::Int = 17,
    n_out::Int = 3000,
    discard::Int = 1000,
    y0::Float64 = 1.2,
    alpha::Float64 = 0.2,
    beta::Float64 = 10.0,
    gamma::Float64 = 0.1,
    delta::Float64 = 0.1,
    subsample::Int = 10,
)
    tau_steps = round(Int, tau / delta)
    @assert isapprox(tau_steps * delta, tau; atol=1e-9) "tau/delta must be (close to) an integer number of fine steps"

    n_fine_needed = (discard + n_out) * subsample
    n_total_fine = tau_steps + n_fine_needed + 1

    y = fill(y0, n_total_fine)   # y[1:tau_steps] = constant history for t in [-tau, 0)

    for n in tau_steps:(n_total_fine - 1)
        y_delay = y[n - tau_steps + 1]
        y[n + 1] = y[n] + delta * (alpha * y_delay / (1 + y_delay^beta) - gamma * y[n])
    end

    fine_from_t0 = @view y[(tau_steps + 1):end]     # index 1 here is t=0
    subsampled = fine_from_t0[1:subsample:end]       # unit-time-step output series

    out = subsampled[(discard + 1):(discard + n_out)]
    data = reshape(collect(out), 1, :)

    return (data=data, dt_data=1.0, labels=["MG(t), τ=$tau"])
end

"""Jaeger's [-1,1] network I/O convention for Mackey-Glass: y ↦ tanh(y-1)."""
mg_squash(y) = tanh.(y .- 1)

"""Inverse of `mg_squash`, back to original Mackey-Glass coordinates."""
mg_unsquash(z) = atanh.(clamp.(z, -1 + 1e-12, 1 - 1e-12)) .+ 1

# ============================================================
# my_ABM_analysis.jl
# Generic ABM analysis utilities
# ============================================================
#
# This file contains generic summaries that can be applied to any model
# with positions and velocities: global order parameters, summary statistics,
# and cluster-level summaries.
# order_parameters_2d
# order_parameters_3d
# order_parameters
# order_parameters
# global_order_parameter_df
# mean_order_parameters
# cluster_components
# cluster_order_parameters

# Model-specific interpretations should stay in the relevant model folder.
#
# ============================================================
# ============================================================
# Generic speed summaries
# ============================================================
function mean_agent_speed(vel)
    isempty(vel) && return 0.0

    s = 0.0
    @inbounds for v in vel
        s += norm(v)
    end

    return s / length(vel)
end

function order_parameters_2d(
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    L::Float64;
    displacement
)
    N = length(pos)
    N == 0 && return (0.0, 0.0, 0.0, 0.0)

    mx = 0.0
    my = 0.0
    @inbounds for i in 1:N
        v̂ = unit(vel[i])
        mx += v̂.x
        my += v̂.y
    end
    polarisation = sqrt((mx / N)^2 + (my / N)^2)

    c = periodic_centre_of_mass_2d(pos, L)

    rot_sum = 0.0
    dil_sum = 0.0
    ang_sum = 0.0
    count = 0

    @inbounds for i in 1:N
        r = displacement(c, pos[i], L)
        ρ = norm(r)
        v = vel[i]
        vmag = norm(v)

        if ρ ≥ 1e-9 && vmag ≥ 1e-12
            r̂ = r / ρ
            t̂ = SVector2(-r̂.y, r̂.x)
            v̂ = unit(v)

            rot_sum += dot(v̂, t̂)
            dil_sum += dot(v̂, r̂)
            # Couzin (2002) eqn 5's r_ic x v_i term, using *raw* r (not r̂) --
            # summed as a signed quantity across agents before taking the
            # magnitude below, so opposite-handedness contributions can
            # cancel, matching eqn 5's |sum(...)| structure. r.x*v̂.y-r.y*v̂.x
            # is the z-component of r x v̂ (v_i is unit by the paper's own
            # notational convention -- see `rotation` below for why r itself
            # is not).
            ang_sum += r.x * v̂.y - r.y * v̂.x
            count += 1
        end
    end

    # Couzin (2002) eqn 5: m_group = (1/N)|sum_i(r_ic x v_i)|. The paper's own
    # text defines v_i as a *unit* direction vector, so v̂ above matches that
    # convention -- but the equation as typeset shows no such hat on r_ic, and
    # a literal raw-r reading would make m_group scale with the group's actual
    # physical radius rather than stay in the paper's own stated [0,1] bound.
    # This implementation normalizes r to r̂ to satisfy that stated bound; that
    # is a deliberate reading of an internally ambiguous equation, not a
    # verified transcription -- `abs_angular_momentum` below keeps the raw-r
    # reading instead, as a cross-check that isn't subject to the same
    # ambiguity resolution.
    #
    # Take the magnitude after summing so opposite milling directions do not
    # cancel during time or ensemble averaging.
    rotation = count > 0 ? abs(rot_sum / count) : 0.0
    dilation = count > 0 ? dil_sum / count : 0.0
    abs_angular_momentum = count > 0 ? abs(ang_sum / count) : 0.0

    return (dilation, rotation, polarisation, abs_angular_momentum)
end


function order_parameters_3d(
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    L::Float64;
    displacement
)
    N = length(pos)
    N == 0 && return (0.0, 0.0, 0.0, 0.0)

    mx = 0.0
    my = 0.0
    mz = 0.0
    @inbounds for i in 1:N
        v̂ = unit(vel[i])
        mx += v̂.x
        my += v̂.y
        mz += v̂.z
    end
    polarisation = sqrt((mx / N)^2 + (my / N)^2 + (mz / N)^2)

    c = periodic_centre_of_mass_3d(pos, L)

    rotx = 0.0
    roty = 0.0
    rotz = 0.0
    dil_sum = 0.0
    count = 0

    # Raw-r (not r̂) cross-product accumulator -- see order_parameters_2d's
    # `rotation` comment for the r vs r̂ ambiguity in Couzin (2002) eqn 5.
    # This field keeps the literal raw-r reading as a cross-check.
    ang_x = 0.0
    ang_y = 0.0
    ang_z = 0.0

    @inbounds for i in 1:N
        r = displacement(c, pos[i], L)
        ρ = norm(r)
        v = vel[i]
        vmag = norm(v)

        if ρ ≥ 1e-9 && vmag ≥ 1e-12
            r̂ = r / ρ
            v̂ = unit(v)

            dil_sum += dot(v̂, r̂)
            ℓ = cross(r̂, v̂)
            rotx += ℓ.x
            roty += ℓ.y
            rotz += ℓ.z

            # Summed as a vector (not per-agent magnitude) before the final
            # norm(), so opposite-handedness rotation cancels -- matching eqn
            # 5's |sum(...)| structure.
            m = cross(r, v̂)
            ang_x += m.x
            ang_y += m.y
            ang_z += m.z

            count += 1
        end
    end

    rotation = count > 0 ? sqrt((rotx / count)^2 + (roty / count)^2 + (rotz / count)^2) : 0.0
    dilation = count > 0 ? dil_sum / count : 0.0
    abs_angular_momentum = count > 0 ? sqrt((ang_x / count)^2 + (ang_y / count)^2 + (ang_z / count)^2) : 0.0

    return (dilation, rotation, polarisation, abs_angular_momentum)
end

function order_parameters(
    pos::Vector{SVector2},
    vel::Vector{SVector2},
    L::Float64;
    displacement
)
    return order_parameters_2d(pos, vel, L; displacement=displacement)
end

function order_parameters(
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    L::Float64;
    displacement
)
    return order_parameters_3d(pos, vel, L; displacement=displacement)
end


function global_order_parameter_df(out)
    DataFrame(
        time_index = 1:length(out.t),
        t = out.t,
        dilation_global = out.dilation,
        rotation_global = out.rotation,
        polarisation_global = out.polarisation,
        abs_angular_momentum_global = out.abs_angular_momentum,
        speed_global = out.speed,
    )
end

function mean_order_parameters(out; transient_steps::Int=0)
    @assert has_order(out) "Cannot compute averages: order parameters were not recorded."

    i0 = min(transient_steps + 1, length(out.t))

    return (
        dilation_mean               = mean(@view out.dilation[i0:end]),
        rotation_mean               = mean(@view out.rotation[i0:end]),
        polarisation_mean           = mean(@view out.polarisation[i0:end]),
        abs_angular_momentum_mean   = mean(@view out.abs_angular_momentum[i0:end]),
        speed_mean                  = mean(@view out.speed[i0:end]),

        dilation_std                = std(@view out.dilation[i0:end]),
        rotation_std                = std(@view out.rotation[i0:end]),
        polarisation_std            = std(@view out.polarisation[i0:end]),
        abs_angular_momentum_std    = std(@view out.abs_angular_momentum[i0:end]),
        speed_std                   = std(@view out.speed[i0:end]),

        transient_index             = i0,
        transient_time              = out.t[i0],
    )
end


"""
    order_parameter_windows(out; metric=:rotation, window=250, step=window)

Windowed mean/std of one order-parameter time series from a `simulate`
output `out` (`metric` in `:dilation`, `:rotation`, `:polarisation`,
`:abs_angular_momentum`, `:speed`). `step < window` gives overlapping
windows. Returns `(starts, t, mean, std)` -- `starts` are the window's
first-step indices into `out` (used by `estimate_settling_transient`
below to map a settled window back to a step count); `t`/`mean`/`std`
are each window's centre time and windowed mean/std, meant for plotting
or feeding into a settling check.
"""
function order_parameter_windows(out; metric::Symbol = :rotation, window::Int = 250, step::Int = window)
    @assert has_order(out) "Cannot compute windows: order parameters were not recorded."
    y = getfield(out, metric)
    T = length(y)
    @assert window <= T "window ($window) exceeds series length ($T)."

    starts = collect(1:step:(T - window + 1))
    t_centre = Float64[out.t[s + window ÷ 2] for s in starts]
    wmean = Float64[mean(@view y[s:(s + window - 1)]) for s in starts]
    wstd = Float64[std(@view y[s:(s + window - 1)]) for s in starts]
    return (starts = starts, t = t_centre, mean = wmean, std = wstd)
end

"""
    estimate_settling_transient(out; metric=:rotation, window=250, step=window÷2,
                                 tol=0.15, tail_fraction=0.25, min_tail_windows=6)

Estimate the settling transient from windowed order-parameter means.

A candidate is accepted only when every later window remains within `tol`
of a stable tail reference. Returns `settled=false` when the run is too
short, the tail is still changing, or no candidate is found. The returned
`windows` should be inspected for intermittent regimes.

Returns `(settled, transient_steps, transient_time, windows, tail_reference)`.
"""
function estimate_settling_transient(out;
    metric::Symbol = :rotation,
    window::Int = 250,
    step::Int = max(1, window ÷ 2),
    tol::Float64 = 0.15,
    tail_fraction::Float64 = 0.25,
    min_tail_windows::Int = 6,
)
    w = order_parameter_windows(out; metric = metric, window = window, step = step)
    nw = length(w.mean)
    n_tail = max(min_tail_windows, ceil(Int, tail_fraction * nw))
    @assert nw > n_tail "Not enough windows ($nw) to form a tail reference of $n_tail -- use a shorter window or a longer run."

    if nw < 2 * n_tail
        # Too short a run for the tail reference to be meaningfully distinct
        # from the whole series -- report "can't tell yet", not a guess.
        return (
            settled = false,
            transient_steps = nothing,
            transient_time = nothing,
            windows = w,
            tail_reference = missing,
        )
    end

    tail_window_means = w.mean[(nw - n_tail + 1):nw]
    tail_ref = mean(tail_window_means)

    # Linear trend across the tail (total drift start-to-end, via a simple
    # least-squares slope) -- `std` alone misses a *smooth* monotonic decline
    # (e.g. still sliding from ~0.8 toward the true ~0.0 asymptote), since a
    # gentle, low-noise slope has low std despite being nowhere near settled.
    n = length(tail_window_means)
    xs = 1:n
    xbar, ybar = mean(xs), tail_ref
    slope = sum((xs .- xbar) .* (tail_window_means .- ybar)) / sum((xs .- xbar) .^ 2)
    tail_drift = abs(slope) * (n - 1)

    if std(tail_window_means) > tol / 2 || tail_drift > tol / 2
        # The tail region itself is still trending/drifting (e.g. a run cut
        # off mid-transition, before the true asymptotic level is visible) --
        # it isn't a trustworthy reference to compare earlier windows against,
        # even though there were nominally "enough" windows for one. Treat
        # this the same as "can't tell yet" rather than anchoring on a
        # moving (or smoothly sliding) target.
        return (
            settled = false,
            transient_steps = nothing,
            transient_time = nothing,
            windows = w,
            tail_reference = tail_ref,
        )
    end

    settled_idx = nothing
    for i in 1:(nw - min_tail_windows + 1)
        if all(abs.(w.mean[i:end] .- tail_ref) .< tol)
            settled_idx = i
            break
        end
    end

    if settled_idx === nothing
        return (
            settled = false,
            transient_steps = nothing,
            transient_time = nothing,
            windows = w,
            tail_reference = tail_ref,
        )
    end

    step_idx = w.starts[settled_idx]
    return (
        settled = true,
        transient_steps = step_idx - 1,
        transient_time = out.t[step_idx],
        windows = w,
        tail_reference = tail_ref,
    )
end

"""
    auto_settling_transient(simulate_fn; metric=:rotation, initial_steps=2000,
                             growth_factor=2.0, max_steps=32000,
                             n_confirmations=1, kwargs...)

Repeat a simulation at increasing lengths until
`estimate_settling_transient` succeeds for the requested number of
confirmations or `max_steps` is reached.

`simulate_fn` must accept a step count and return a simulation result.
Returns `(result, out, steps_used, converged)`.
"""
function auto_settling_transient(simulate_fn::Function;
    metric::Symbol = :rotation,
    initial_steps::Int = 2000,
    growth_factor::Float64 = 2.0,
    max_steps::Int = 32000,
    n_confirmations::Int = 1,
    show_progress::Bool = false,
    window::Int = 250,
    step::Int = max(1, window ÷ 2),
    tol::Float64 = 0.15,
    tail_fraction::Float64 = 0.25,
    min_tail_windows::Int = 6,
)
    steps = initial_steps
    local out, result
    confirmations = 0
    while true
        out = simulate_fn(steps)
        result = estimate_settling_transient(out; metric = metric, window = window, step = step,
            tol = tol, tail_fraction = tail_fraction, min_tail_windows = min_tail_windows)

        confirmations = result.settled ? confirmations + 1 : 0

        if show_progress
            msg = "steps=$steps (t=$(round(out.t[end], digits=1)))  settled=$(result.settled)"
            result.settled && (msg *= "  transient_time=$(round(result.transient_time, digits=1))  confirmations=$confirmations/$(n_confirmations + 1)")
            println(msg)
        end

        (confirmations > n_confirmations || steps >= max_steps) && break
        steps = min(max_steps, ceil(Int, steps * growth_factor))
    end

    return (result = result, out = out, steps_used = steps, converged = confirmations > n_confirmations)
end

function cluster_components(
    pos,
    L::Float64;
    displacement,
    R_cluster::Float64,
    min_size::Int = 1,
)
    N = length(pos)
    visited = falses(N)
    components = Vector{Vector{Int}}()

    function neighbours(i)
        nbrs = Int[]
        pi = pos[i]
        @inbounds for j in 1:N
            j == i && continue
            d = displacement(pi, pos[j], L)
            if norm(d) <= R_cluster
                push!(nbrs, j)
            end
        end
        return nbrs
    end

    for i in 1:N
        visited[i] && continue

        comp = Int[]
        stack = [i]
        visited[i] = true

        while !isempty(stack)
            v = pop!(stack)
            push!(comp, v)

            for u in neighbours(v)
                if !visited[u]
                    visited[u] = true
                    push!(stack, u)
                end
            end
        end

        if length(comp) >= min_size
            push!(components, comp)
        end
    end

    return components
end


function cluster_order_parameters(pos, vel, L; displacement, R_cluster, order_parameters, min_size=1)
    comps = cluster_components(pos, L; displacement, R_cluster, min_size)

    rows = NamedTuple[]
    for (cid, comp) in enumerate(comps)
        vals = order_parameters(pos[comp], vel[comp], L; displacement)
        push!(rows, merge((cluster_id=cid, size=length(comp), members=comp), vals))
    end

    return rows
end

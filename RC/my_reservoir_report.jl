# my_reservoir_report.jl

using CairoMakie
using DataFrames
using LinearAlgebra

# ============================================================
# HELPERS
# ============================================================
function align_to_common_length(A::AbstractMatrix, B::AbstractMatrix)
    T = min(size(A,2), size(B,2))
    return @view(A[:, 1:T]), @view(B[:, 1:T]), T
end

function _save(fig, path::Union{Nothing,String})
    if path !== nothing
        save(path, fig)
        println("Saved: $path")
    end
end

rmse_per_step(Yhat::AbstractMatrix, Ytrue::AbstractMatrix) = begin
    @assert size(Yhat) == size(Ytrue)
    Ny, T = size(Ytrue)
    r = zeros(Float64, T)
    for t in 1:T
        s = 0.0
        for i in 1:Ny
            e = Yhat[i,t] - Ytrue[i,t]
            s += e*e
        end
        r[t] = sqrt(s / Ny)
    end
    r
end

divergence_index(Yhat::AbstractMatrix, Ytrue::AbstractMatrix; thresh::Float64=2.0) = begin
    r = rmse_per_step(Yhat, Ytrue)
    idx = findfirst(>(thresh), r)
    idx === nothing ? length(r) : idx
end
# ============================================================
# OUTPUTS FOLLOWING RESERVOIR TRAINING
# ============================================================
using CairoMakie

function reservoir_report(out;
    prefix::String="run",
    show::Bool=true,
    thresh::Float64=2.0
)
    wash = getproperty(out, :washout)
    dt   = getproperty(out, :dt_data)

    Ytrain = getproperty(out, :Ytrain)
    Ytest  = getproperty(out, :Ytest)

    Yhat_tf_full = getproperty(out, :Yhat_tf)
    Ygen_full    = getproperty(out, :Ygen)

    labels = hasproperty(out, :labels) ? getproperty(out, :labels) :
             ["y$(i)(t)" for i in 1:size(Ytrain,1)]

    Yhat_tf, Ytrain_al, Ttf = align_to_common_length(Yhat_tf_full, Ytrain)
    Ygen,    Ytest_al,  Tfr = align_to_common_length(Ygen_full,    Ytest)

    tf_start  = min(wash + 1, Ttf)
    Yhat_tf_w = @view Yhat_tf[:, tf_start:end]
    Ytrain_w  = @view Ytrain_al[:, tf_start:end]

    t_tf = (0:(size(Yhat_tf_w,2)-1)) .* dt
    t_fr = (0:(Tfr-1)) .* dt

    Ny = size(Ytrain, 1)

    # ============================================================
    # Combined figure: (Teacher-forced + Free-run) × (Time + Phase)
    # ============================================================
    figMain = Figure(size=(1400, 750))

    # --- Teacher-forced: time series (left) ---
    axTF_ts = Axis(figMain[1, 1];
        xlabel = "",
        ylabel = "value",
        title  = "Teacher-forced one-step (post-washout): time series"
    )
    for d in 1:Ny
        lines!(axTF_ts, t_tf, vec(Ytrain_w[d, :]); label="$(labels[d]) truth")
        lines!(axTF_ts, t_tf, vec(Yhat_tf_w[d, :]); label="$(labels[d]) pred", linestyle=:dash)
    end
    axislegend(axTF_ts, position=:rb)

    # --- Teacher-forced: phase space (right) ---
    axTF_ph = Axis(figMain[1, 2];
        xlabel = Ny >= 1 ? labels[1] : "y1",
        ylabel = Ny >= 2 ? labels[2] : "y2",
        title  = Ny >= 2 ? "Teacher-forced: phase space" : "Teacher-forced: phase space (Ny < 2)"
    )
    if Ny >= 2
        lines!(axTF_ph, vec(Ytrain_w[1, :]), vec(Ytrain_w[2, :]); label="truth")
        lines!(axTF_ph, vec(Yhat_tf_w[1, :]), vec(Yhat_tf_w[2, :]); label="pred", linestyle=:dash)
        axislegend(axTF_ph, position=:rb)
    end

    # --- Free-run: time series (left) ---
    axFR_ts = Axis(figMain[2, 1];
        xlabel = "time",
        ylabel = "value",
        title  = "Free-run rollout: time series"
    )
    for d in 1:Ny
        lines!(axFR_ts, t_fr, vec(Ytest_al[d, :]); label="$(labels[d]) truth")
        lines!(axFR_ts, t_fr, vec(Ygen[d, :]);     label="$(labels[d]) pred", linestyle=:dash)
    end
    axislegend(axFR_ts, position=:rb)

    # --- Free-run: phase space (right) ---
    axFR_ph = Axis(figMain[2, 2];
        xlabel = Ny >= 1 ? labels[1] : "y1",
        ylabel = Ny >= 2 ? labels[2] : "y2",
        title  = Ny >= 2 ? "Free-run: phase space" : "Free-run: phase space (Ny < 2)"
    )
    if Ny >= 2
        lines!(axFR_ph, vec(Ytest_al[1, :]), vec(Ytest_al[2, :]); label="truth")
        lines!(axFR_ph, vec(Ygen[1, :]),     vec(Ygen[2, :]);     label="pred", linestyle=:dash)
        axislegend(axFR_ph, position=:rb)
    end

    _save(figMain, "$(prefix)_main_timeseries_phase.png")
    show && display(figMain)

    # ----------------------------
    # RMSE curves (kept as-is)
    # ----------------------------
    rmse_tf = rmse_per_step(Yhat_tf_w, Ytrain_w)
    rmse_fr = rmse_per_step(Ygen, Ytest_al)

    div_fr = divergence_index(Ygen, Ytest_al; thresh=thresh)
    t_div  = t_fr[div_fr]

    figC = Figure(size=(1100, 420))
    axC = Axis(figC[1, 1], xlabel="time", ylabel="RMSE (per step)",
               title="Prediction error vs time (per-step RMSE)")
    lines!(axC, t_tf, rmse_tf, label="teacher-forced (post-washout)")
    lines!(axC, t_fr, rmse_fr, label="free-run")
    hlines!(axC, [thresh], linestyle=:dash, label="threshold")
    vlines!(axC, [t_div], linestyle=:dash, label="free-run divergence")
    axislegend(axC, position=:rt)
    _save(figC, "$(prefix)_rmse_vs_time.png")
    # show && display(figC)

    # Optional 3D attractor overlay (kept)
    figD = nothing
    if Ny >= 3
        figD = Figure(size=(900, 650))
        axD = Axis3(figD[1, 1], xlabel=labels[1], ylabel=labels[2], zlabel=labels[3],
                    title="Attractor overlay (free-run vs truth)")
        lines!(axD, vec(Ytest_al[1,:]), vec(Ytest_al[2,:]), vec(Ytest_al[3,:]), label="truth")
        lines!(axD, vec(Ygen[1,:]),     vec(Ygen[2,:]),     vec(Ygen[3,:]),     label="pred")
        axislegend(axD, position=:rt)
        _save(figD, "$(prefix)_attractor_free_run.png")
        # show && display(figD)
    end

    # Readout sanity
    Wout = getproperty(out, :Wout)
    w_norm = norm(Wout)
    w_max  = maximum(abs.(Wout))
    println("\nReadout sanity:")
    println("  ‖Wout‖₂ = $w_norm")
    println("  max|Wout| = $w_max")

    # Summary
    println("\nSummary:")
    if hasproperty(out, :mode)
        println("  reservoir mode: ", getproperty(out, :mode))
    end
    println("  washout: $wash, dt_data: $dt")
    println("  free-run divergence index (RMSE > $thresh): $div_fr  (t = $t_div)")
    if hasproperty(out, :m_tf)
        println("  teacher-forced: NRMSE=$(out.m_tf.nrmse)  R²=$(out.m_tf.r2)")
    end
    if hasproperty(out, :m_fr)
        println("  free-run:       NRMSE=$(out.m_fr.nrmse)  R²=$(out.m_fr.r2)")
    end

    return (
        figs = (Overview=figMain, RMSE=figC, Attractor=figD),
        rmse = (tf=rmse_tf, fr=rmse_fr),
        divergence = (idx=div_fr, t=t_div),
        readout = (norm=w_norm, maxabs=w_max)
    )
end


# ------------------------------------------------------------
# Generic diagnostic plot: works for TF or free-run
# ------------------------------------------------------------
"""
Single diagnostic figure for 2D outputs (Ny=2):

- Left column: y₁(t), y₂(t) truth vs prediction (2 rows)
- Right column: phase portrait (y₁ vs y₂) spanning both rows
- Optional marker at t = washout*dt (or any transient boundary)

Use for:
- teacher-forced one-step: (Ytrue=Ytrain, Yhat=Yhat_TF)
- free-run rollout:        (Ytrue=Ytest,  Yhat=Ygen)   or (Yhat=Yhat_FR)
"""
function plot_truth_vs_pred_2d(Ytrue::AbstractMatrix, Yhat::AbstractMatrix;
    washout::Int=0,
    dt::Real=1.0,
    labels::Vector{String}=["y₁","y₂"],
    title::String="Truth vs prediction",
    mark_washout::Bool=true
)
    @assert size(Ytrue, 1) == 2 "Expected Ny=2 for phase portrait."
    @assert size(Yhat,  1) == 2 "Expected Yhat to be 2×T."

    T = min(size(Ytrue, 2), size(Yhat, 2))
    Yt = @view Ytrue[:, 1:T]
    Yp = @view Yhat[:,  1:T]

    t = (0:T-1) .* dt
    t0_idx = min(washout + 1, T)  # start index after transient

    fig = Figure(size=(1200, 650))

    # --- Left column: time series (2 rows) ---
    ax1 = Axis(fig[1, 1];
        xlabel="",
        ylabel=labels[1],
        title=title
    )
    lines!(ax1, t, vec(Yt[1, :]); linewidth=3, label="truth")
    lines!(ax1, t, vec(Yp[1, :]); label="pred")
    if mark_washout && washout > 0
        vlines!(ax1, [t[t0_idx]]; linestyle=:dash, color=:red, label="transient")
    end
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[2, 1];
        xlabel="time",
        ylabel=labels[2]
    )
    lines!(ax2, t, vec(Yt[2, :]); linewidth=3, label="truth")
    lines!(ax2, t, vec(Yp[2, :]); label="pred")
    if mark_washout && washout > 0
        vlines!(ax2, [t[t0_idx]]; linestyle=:dash, color=:red)
    end

    # --- Right column: phase portrait spanning both rows ---
    axP = Axis(fig[1:2, 2];
        xlabel=labels[1],
        ylabel=labels[2],
        title="Phase space (post-transient)"
    )
    lines!(axP,
        vec(@view Yt[1, t0_idx:end]),
        vec(@view Yt[2, t0_idx:end]);
        linewidth=2,
        label="truth"
    )
    lines!(axP,
        vec(@view Yp[1, t0_idx:end]),
        vec(@view Yp[2, t0_idx:end]);
        color=(:orange, 0.8),
        label="pred"
    )
    axislegend(axP, position=:rb)

    colsize!(fig.layout, 1, Relative(0.52))
    colsize!(fig.layout, 2, Relative(0.48))

    return fig
end


function print_reservoir_diagnostics(out)
    println("\n================ Reservoir diagnostics ================")

    if hasproperty(out, :reservoir_diag) && out.reservoir_diag !== nothing
        dtr = out.reservoir_diag.features.train
        dfr = out.reservoir_diag.features.freerun

        println("\nFeature-space richness:")
        println("  train participation ratio = $(dtr.participation_ratio)")
        println("  train effective rank      = $(dtr.effective_rank)")
        println("  train dead feature frac   = $(dtr.dead_feature_fraction)")
        println("  train mean |corr|         = $(dtr.mean_abs_corr)")
        println("  train Gram cond           = $(dtr.gram_condition)")

        println("\nFree-run feature-space:")
        println("  free-run participation ratio = $(dfr.participation_ratio)")
        println("  free-run effective rank      = $(dfr.effective_rank)")
        println("  free-run dead feature frac   = $(dfr.dead_feature_fraction)")
        println("  free-run mean |corr|         = $(dfr.mean_abs_corr)")
    end

    if hasproperty(out, :memory_diag) && out.memory_diag !== nothing
        println("\nMemory:")
        println("  total memory capacity = $(out.memory_diag.mc)")
        kbest = argmax(out.memory_diag.r2)
        println("  best lag = $(out.memory_diag.lags[kbest]) with R² = $(out.memory_diag.r2[kbest])")
    end

    if hasproperty(out, :stability_diag) && out.stability_diag !== nothing
        println("\nStability:")
        println("  log-distance slope     = $(out.stability_diag.log_slope)")
        println("  median contraction ratio = $(out.stability_diag.contraction_ratio)")
    end

    if hasproperty(out, :separability_diag) && out.separability_diag !== nothing
        println("\nSeparability:")
        println("  mean pairwise distance = $(out.separability_diag.mean_pairwise_distance)")
        println("  min pairwise distance  = $(out.separability_diag.min_pairwise_distance)")
    end
end



function plot_memory_curve(memory_diag; title::String="Reservoir memory curve")

    lags = memory_diag.lags
    r2   = memory_diag.r2
    mc   = memory_diag.mc

    fig = Figure(size=(900, 400))
    ax = Axis(fig[1, 1],
        xlabel="lag",
        ylabel="R²",
        title="$title   (MC = $(round(mc, digits=2)))"
    )

    lines!(ax, lags, r2, linewidth=2)

    # Optional: mark where memory essentially vanishes
    # cutoff = findfirst(<(0.05), r2)
    # if cutoff !== nothing
    #     vlines!(ax, [lags[cutoff]], linestyle=:dash)
    # end

    # # annotate MC in the plot
    # text!(ax,
    #     maximum(lags)*0.75,
    #     0.9,
    #     text = "MC = $(round(mc, digits=2))",
    #     align = (:left, :center)
    # )

    band!(ax, lags, zeros(length(r2)), r2, transparency=true)

    return fig
end

function plot_stability_curve(stability_diag; dt::Real=1.0, title::String="Perturbation decay")
    t = (0:length(stability_diag.δ)-1) .* dt
    logδ = log10.(stability_diag.δ .+ 1e-16)

    fig = Figure(size=(900, 400))
    ax = Axis(fig[1, 1],
        xlabel="time",
        ylabel="log10 distance",
        title=title
    )
    lines!(ax, t, logδ, linewidth=2)
    return fig
end

function plot_separability_matrix(sep_diag; title="Pairwise sequence distances")

    D = sep_diag.pairwise_distances

    fig = Figure(size=(600,500))
    ax = Axis(fig[1,1],
        title=title,
        xlabel="sequence",
        ylabel="sequence"
    )

    hm = heatmap!(ax, D)

    Colorbar(fig[1,2], hm, label="distance")

    return fig
end

"""
    plot_ridge_lambda_tradeoff(Xtr, Ytr, Xte, Yte;
                                ridge_grid=10.0 .^ (-4:0.5:5))

Fit a ridge readout across `ridge_grid` and compare held-out correlation
with prediction roughness. Returns `(fig, results)`, where `results`
contains `λ`, `R` and `roughness`.
"""
function plot_ridge_lambda_tradeoff(Xtr::AbstractMatrix, Ytr::AbstractMatrix,
    Xte::AbstractMatrix, Yte::AbstractMatrix;
    ridge_grid = 10.0 .^ (-4:0.5:5),
    title::String = "Ridge λ trade-off: R vs. prediction roughness",
    fig_size::Tuple{Int,Int} = (800, 400),
)
    logλ = Float64[]
    Rs = Float64[]
    roughnesses = Float64[]
    for λ in ridge_grid
        Wout, μx, σx = train_readout_from_features(Xtr, Ytr; ridgeλ = Float64(λ), washout = 0)
        Yhat = apply_readout(Wout, Xte, μx, σx)
        push!(logλ, log10(λ))
        push!(Rs, cor(vec(Yhat), vec(Yte)))
        push!(roughnesses, std(diff(vec(Yhat))))
    end

    fig = Figure(size = fig_size)
    ax1 = Axis(fig[1, 1], xlabel = "log10(λ)", ylabel = "R (held-out correlation)", title = title)
    l1 = lines!(ax1, logλ, Rs, color = :steelblue)

    ax2 = Axis(fig[1, 1], ylabel = "prediction roughness (std of 1st diff)", yaxisposition = :right)
    hidespines!(ax2)
    hidexdecorations!(ax2)
    l2 = lines!(ax2, logλ, roughnesses, color = :firebrick)

    Legend(fig[1, 2], [l1, l2], ["R", "roughness"])

    results = DataFrame(λ = collect(ridge_grid), R = Rs, roughness = roughnesses)
    return fig, results
end

"""
    feature_conditioning_report(X; energy_threshold=0.99, rank_reltol=1e-8)

Summarise the singular-value spectrum of a feature matrix.

Returns its condition number, effective rank at `energy_threshold`,
numerical rank at `rank_reltol`, nominal rank ceiling and singular values.
"""
function feature_conditioning_report(X::AbstractMatrix; energy_threshold::Float64 = 0.99, rank_reltol::Float64 = 1e-8)
    sv = svdvals(X)
    energy = cumsum(sv .^ 2) ./ sum(sv .^ 2)
    eff_rank = findfirst(>=(energy_threshold), energy)
    num_rank = count(>=(rank_reltol * sv[1]), sv)
    return (
        condition_number = sv[1] / sv[end],
        numerical_rank = num_rank,
        effective_rank = eff_rank,
        nominal_size = min(size(X)...),
    )
end


# One semantic colour per metric, reused unchanged for both the 2D and 3D
# Couzin sweeps and for both heatmap and surface renderings.
const COUZIN_PHASE_COLORMAPS = (
    polarisation = :viridis,
    rotation = :plasma,
    abs_angular_momentum = :magma,
)

function Couzin_phase_metrics()
    return [
        phase_metric(:polarisation_mean;
            name = :pol,
            title = "Mean polarisation",
            label = "polarisation",
            zlabel = L"P_{\mathrm{group}}",
            colormap = COUZIN_PHASE_COLORMAPS.polarisation,
        ),

        phase_metric(:rotation_mean;
            name = :rot,
            title = "Mean rotation",
            label = "rotation",
            zlabel = L"m_{\mathrm{group}}",
            colormap = COUZIN_PHASE_COLORMAPS.rotation,
        ),

        phase_metric(:abs_angular_momentum_mean;
            name = :mabs,
            title = "Mean abs angular momentum",
            label = "abs angular momentum",
            zlabel = L"M_{\mathrm{abs}}",
            colormap = COUZIN_PHASE_COLORMAPS.abs_angular_momentum,
        ),
    ]
end


function summarise_Couzin_phase(df::DataFrame)
    return summarise_phase(
        df;
        xcol = :Δro,
        ycol = :Δra,
        metrics = Couzin_phase_metrics(),
    )
end


function Couzin_phase_plots(results::DataFrame;
    Δro_vals,
    Δra_vals,
    smooth::Bool = false,
    σ::Real = 1.0,
    interpolate::Bool = false,
    interpolation_factor::Int = 4,
)
    return phase_plots(
        results;
        xcol = :Δro,
        ycol = :Δra,
        xvals = Δro_vals,
        yvals = Δra_vals,
        metrics = Couzin_phase_metrics(),
        xlabel = L"\Delta r_o",
        ylabel = L"\Delta r_a",
        smooth = smooth,
        σ = σ,
        interpolate = interpolate,
        interpolation_factor = interpolation_factor,
    )
end


function phase_specification(::Val{:Couzin}, P_grid)
    return (
        xcol = :Zo,
        ycol = :Za,
        xvals = grid_values(P_grid, :Zo),
        yvals = grid_values(P_grid, :Za),
        metrics = Couzin_phase_metrics(),
        xlabel = L"Z_o",
        ylabel = L"Z_a",
    )
end

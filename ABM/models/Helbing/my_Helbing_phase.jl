function Helbing_phase_metrics()
    return [
        phase_metric(:speed_mean;
            name = :speed,
            title = "Mean speed",
            label = "speed",
            zlabel = "speed",
            colormap = :viridis,
        ),

        phase_metric(:polarisation_mean;
            name = :pol,
            title = "Mean polarisation",
            label = "polarisation",
            zlabel = "polarisation",
            colormap = :plasma,
        ),

        phase_metric(:abs_angular_momentum_mean;
            name = :mabs,
            title = "Mean abs angular momentum",
            label = "abs angular momentum",
            zlabel = "abs angular momentum",
            colormap = :magma,
        ),
    ]
end


function phase_specification(::Val{:Helbing}, P_grid)
    return (
        xcol = :tau,
        ycol = :A_agent,
        xvals = grid_values(P_grid, :tau),
        yvals = grid_values(P_grid, :A_agent),
        metrics = Helbing_phase_metrics(),
        xlabel = L"\tau",
        ylabel = L"A_{\mathrm{agent}}",
    )
end
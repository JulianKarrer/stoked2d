use convert_case::{Case, Casing};
use glam::DVec2;
use plotly::{Configuration, ImageFormat, Plot, Scatter, Surface};
use stoked2d::{gui::plot::standard_2d_plot, sph::KernelType, utils::linspace, KERNEL_SUPPORT};
use strum::IntoEnumIterator;

fn standard_3d_plot() -> Plot {
    let mut plot = Plot::new();
    plot.set_configuration(Configuration::new().responsive(true));
    plot
}

fn main() {
    plot_kernels(500);
    plot_kernel_derivatives(200);
}

/// Plot the kernel functions.
fn plot_kernels(resolution: usize) {
    let mut plot = standard_2d_plot();
    KernelType::iter().for_each(|kernel_type| {
        let xs = linspace(-KERNEL_SUPPORT, KERNEL_SUPPORT, resolution);
        let ys = xs
            .iter()
            .map(|x| kernel_type.w(&DVec2::ZERO, &DVec2::new(*x, 0.)))
            .collect::<Vec<f64>>();

        let trace = Scatter::new(xs, ys).name(format!("{:?}", kernel_type));
        plot.add_trace(trace)
    });
    plot.write_html("analysis/plot_kernels.html");
    plot.write_image("analysis/plot_kernels.png", ImageFormat::PNG, 800, 600, 1.0);
}

/// Plot the kernel derivatives.
fn plot_kernel_derivatives(resolution: usize) {
    KernelType::iter().for_each(|kernel_type| {
        let mut plot = standard_3d_plot();
        let xs = linspace(-KERNEL_SUPPORT, KERNEL_SUPPORT, resolution);
        let ys = linspace(-KERNEL_SUPPORT, KERNEL_SUPPORT, resolution);
        let values = xs
            .iter()
            .map(|x| {
                ys.iter()
                    .map(|y| kernel_type.dw(&DVec2::ZERO, &DVec2::new(*x, *y)).length())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        let trace =
            Surface::new(values)
                .x(xs)
                .y(ys)
                .color_scale(plotly::common::ColorScale::Palette(
                    plotly::common::ColorScalePalette::Reds,
                ));
        plot.add_trace(trace);
        plot.write_html(
            format!("analysis/plot_kernel_derivatives_{:?}.html", kernel_type).to_case(Case::Snake),
        );
    });
}

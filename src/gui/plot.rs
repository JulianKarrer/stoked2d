use plotly::{Configuration, Plot};

pub fn standard_2d_plot() -> Plot {
    let mut plot = Plot::new();
    plot.set_layout(
        plot.layout()
            .clone()
            .drag_mode(plotly::layout::DragMode::Pan),
    );
    plot.set_configuration(Configuration::new().responsive(true).scroll_zoom(true));
    plot
}

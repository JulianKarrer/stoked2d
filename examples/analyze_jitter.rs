use plotly::{ImageFormat, Scatter};
use stoked2d::{
    gui::plot::standard_2d_plot,
    simulation::run,
    utils::{get_timestamp, integrate_abs_error, integrate_squared_error, linspace},
    H, HISTORY, INITIAL_JITTER,
};

fn main() {
    let xs: Vec<f64> = linspace(0.0, 0.1, 25);
    let (ys_abs, ys_sqr): (Vec<f64>, Vec<f64>) = xs
        .iter()
        .map(|x| {
            println!("simulating jitter: {}*H", (*x));
            INITIAL_JITTER.store((*x) * H, atomic::Ordering::SeqCst);
            run(Some(10.));
            let abs_err = integrate_abs_error(&HISTORY.read().plot_density, 1.0);
            let sqr_err = integrate_squared_error(&HISTORY.read().plot_density, 1.0);
            println!("error abs: {}, error squared: {}", abs_err, sqr_err);
            (abs_err, sqr_err)
        })
        .unzip();
    let mut plot = standard_2d_plot();
    plot.add_trace(Scatter::new(xs.clone(), ys_abs).name("Absolute Density Error"));
    plot.add_trace(Scatter::new(xs, ys_sqr).name("Squared Density Error"));
    plot.write_html(format!(
        "analysis/plot_jitter_error_{}.html",
        get_timestamp()
    ));
    plot.write_image(
        format!("analysis/plot_jitter_error_{}.png", get_timestamp()),
        ImageFormat::PNG,
        800,
        600,
        1.0,
    );
}

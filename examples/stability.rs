use plotly::Contour;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use stoked2d::{
    gui::plot::standard_2d_plot,
    simulation::run,
    utils::{get_timestamp, integrate_abs_error, linspace},
    HISTORY, K, LAMBDA, NU, NU_2,
};

fn measure_k_nu_stability(
    nus: Vec<f64>,
    ks: Vec<f64>,
    sim_length: f64,
    from_time: f64,
    time_stamp: u64,
) {
    let zs = nus
        .iter()
        .map(|nu| {
            ks.iter()
                .map(|k| {
                    K.store(*k, atomic::Ordering::Relaxed);
                    NU.store(*nu, atomic::Ordering::Relaxed);
                    run(Some(sim_length), "setting_column_small.png");
                    // only look at kietic energies from `from_time` onwards
                    let data: Vec<[f64; 2]> = (*HISTORY
                        .read()
                        .plot_hamiltonian
                        .par_iter()
                        .filter(|[t, _e_kin]| *t > from_time)
                        .cloned()
                        .collect::<Vec<[f64; 2]>>())
                    .to_vec();
                    // find the peak average kinetic energy per particle
                    let peak_e_kin = *data
                        .par_iter()
                        .map(|[_t, e_kin]| e_kin)
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap();
                    // find the integral of the kinetic energy per particle
                    let int_e_kin = integrate_abs_error(&data, 0.0);
                    println!("k:{}, nu:{}, e_kin_peak:{}", k, nu, peak_e_kin);
                    (peak_e_kin, int_e_kin)
                })
                .collect::<Vec<(f64, f64)>>()
        })
        .collect::<Vec<Vec<(f64, f64)>>>();

    // unzip the data collected
    let zs_peak = zs
        .par_iter()
        .map(|row| row.iter().map(|entry| entry.0).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();
    let zs_int = zs
        .par_iter()
        .map(|row| row.iter().map(|entry| entry.1).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

    // plot the data
    let mut peak_plot = standard_2d_plot();
    peak_plot.add_trace(Contour::new(ks.clone(), nus.clone(), zs_peak).name("Peak Kinetic Energy"));
    peak_plot.write_html(format!("analysis/plot_stability_peak_{}.html", time_stamp));
    let mut int_plot = standard_2d_plot();
    int_plot.add_trace(Contour::new(ks, nus, zs_int).name("Time Integral of Kinetic Energy"));
    int_plot.write_html(format!("analysis/plot_stability_int_{}.html", time_stamp));
}

fn measure_k_lambda_stability(
    lambdas: Vec<f64>,
    ks: Vec<f64>,
    sim_length: f64,
    from_time: f64,
    time_stamp: u64,
) {
    let zs = lambdas
        .iter()
        .map(|lambda| {
            ks.iter()
                .map(|k| {
                    K.store(*k, atomic::Ordering::Relaxed);
                    LAMBDA.store(*lambda, atomic::Ordering::Relaxed);
                    run(Some(sim_length), "setting_column_small.png");
                    // only look at kietic energies from `from_time` onwards
                    let data: Vec<[f64; 2]> = (*HISTORY
                        .read()
                        .plot_hamiltonian
                        .par_iter()
                        .filter(|[t, _e_kin]| *t > from_time)
                        .cloned()
                        .collect::<Vec<[f64; 2]>>())
                    .to_vec();
                    // find the peak average kinetic energy per particle
                    let peak_e_kin = *data
                        .par_iter()
                        .map(|[_t, e_kin]| e_kin)
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap();
                    // find the integral of the kinetic energy per particle
                    let int_e_kin = integrate_abs_error(&data, 0.0);
                    println!("k:{}, lambda:{}, e_kin_peak:{}", k, lambda, peak_e_kin);
                    (peak_e_kin, int_e_kin)
                })
                .collect::<Vec<(f64, f64)>>()
        })
        .collect::<Vec<Vec<(f64, f64)>>>();

    // unzip the data collected
    let zs_peak = zs
        .par_iter()
        .map(|row| row.iter().map(|entry| entry.0).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();
    let zs_int = zs
        .par_iter()
        .map(|row| row.iter().map(|entry| entry.1).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

    // plot the data
    let mut peak_plot = standard_2d_plot();
    peak_plot
        .add_trace(Contour::new(ks.clone(), lambdas.clone(), zs_peak).name("Peak Kinetic Energy"));
    peak_plot.write_html(format!(
        "analysis/plot_stability_lambda_peak_{}.html",
        time_stamp
    ));
    let mut int_plot = standard_2d_plot();
    int_plot.add_trace(Contour::new(ks, lambdas, zs_int).name("Time Integral of Kinetic Energy"));
    int_plot.write_html(format!(
        "analysis/plot_stability_lambda_int_{}.html",
        time_stamp
    ));
}

fn main() {
    let res = 5;
    let sim_length = 20.;
    let from_time = 10.0;
    assert!(from_time < sim_length);
    NU_2.store(0., atomic::Ordering::Relaxed);

    let nus = linspace(0.0005, 0.002, res - 1);
    let ks = linspace(750., 1500., res - 1);
    let lambdas = linspace(0.01, 0.2, res - 1);

    let time_stamp = get_timestamp();

    // run a simulation for every nu and k specified
    LAMBDA.store(0.1, atomic::Ordering::Relaxed);
    measure_k_nu_stability(nus, ks.clone(), sim_length, from_time, time_stamp);

    // run a simulation for every lambda and k specified
    NU.store(0.001, atomic::Ordering::Relaxed);
    measure_k_lambda_stability(lambdas, ks, sim_length, from_time, time_stamp);
}

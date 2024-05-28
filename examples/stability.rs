use atomic_float::AtomicF64;
use plotly::Contour;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use stoked2d::{
    gui::plot::standard_2d_plot,
    simulation::{run, Solver},
    utils::{get_timestamp, integrate_abs_error, linspace},
    HISTORY, K, LAMBDA, NU, NU_2, SOLVER,
};
use strum::IntoEnumIterator;

fn measure_stability(
    ys: Vec<f64>,
    xs: Vec<f64>,
    sim_length: f64,
    from_time: f64,
    x_ref: &AtomicF64,
    y_ref: &AtomicF64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let zs = ys
        .iter()
        .map(|y| {
            xs.iter()
                .map(|x| {
                    x_ref.store(*x, atomic::Ordering::Relaxed);
                    y_ref.store(*y, atomic::Ordering::Relaxed);
                    run(Some(sim_length as f32), "scenes/setting_column_small.png");
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
                    println!("x:{}, y:{}, e_kin_peak:{}", x, y, peak_e_kin);
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

    (zs_peak, zs_int)
}

fn plot_data(xs: &[f64], ys: &[f64], zs: (Vec<Vec<f64>>, Vec<Vec<f64>>), name: &str) {
    // plot the data
    let mut peak_plot = standard_2d_plot();
    peak_plot.add_trace(Contour::new(xs.to_vec(), ys.to_vec(), zs.0).name("Peak Kinetic Energy"));
    peak_plot.write_html(format!("analysis/plot_stability_peak_{}.html", name));
    let mut int_plot = standard_2d_plot();
    int_plot.add_trace(
        Contour::new(xs.to_vec(), ys.to_vec(), zs.1).name("Time Integral of Kinetic Energy"),
    );
    int_plot.write_html(format!("analysis/plot_stability_int_{}.html", name));
}

fn main() {
    let res = 5;
    let sim_length = 20.;
    let from_time = 10.0;
    assert!(from_time < sim_length);
    NU_2.store(0., atomic::Ordering::Relaxed);

    let nus = linspace(0.0001, 0.002, res - 1);
    let ks = linspace(500., 1500., res - 1);
    let lambdas = linspace(0.1, 1.0, res - 1);

    // run a simulation for every nu and k specified
    for solver in Solver::iter() {
        SOLVER.store(solver, atomic::Ordering::Relaxed);

        LAMBDA.store(0.1, atomic::Ordering::Relaxed);
        let (zs_peak, zs_int) =
            measure_stability(nus.clone(), ks.clone(), sim_length, from_time, &K, &NU);
        plot_data(
            &ks,
            &nus,
            (zs_peak, zs_int),
            &format!("k_nu_{:?}_{}", solver, get_timestamp()),
        );

        // run a simulation for every lambda and k specified
        NU.store(0.0001, atomic::Ordering::Relaxed);
        let (zs_peak, zs_int) = measure_stability(
            lambdas.clone(),
            ks.clone(),
            sim_length,
            from_time,
            &K,
            &LAMBDA,
        );
        plot_data(
            &ks,
            &lambdas,
            (zs_peak, zs_int),
            &format!("k_lambda_{:?}_{}", solver, get_timestamp()),
        );
    }
}

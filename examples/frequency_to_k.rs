use plotly::Scatter;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustfft::{num_complex::Complex64, FftPlanner};
use stoked2d::{
    gui::plot::standard_2d_plot,
    simulation::run,
    utils::{get_timestamp, linspace},
    FIXED_DT, HISTORY, K, USE_FIXED_DT,
};

fn main() {
    // SETTINGS
    let ks_resolution = 9 * 4;
    let nu_res = 1;
    let seconds_sampled = 10.0;
    let sim_length = 12.0;
    // use a fixed dt and corresponding sample size
    let samples_hz: f64 = 5_000.0;

    // set up fixed timestep
    assert!(sim_length > seconds_sampled);
    let fixed_dt = 1. / samples_hz;
    USE_FIXED_DT.store(true, atomic::Ordering::Relaxed);
    FIXED_DT.store(fixed_dt, atomic::Ordering::Relaxed);

    let ks = linspace(100., 1000., ks_resolution);
    // create plots
    let mut power_spectra = standard_2d_plot();
    let mut k_to_peak = standard_2d_plot();
    let mut k_to_err = standard_2d_plot();

    linspace(0.001, 0.1, nu_res).iter().for_each(|nu| {
        // collect data
        let (peak_freq, err): (Vec<f64>, Vec<f64>) = ks
            .iter()
            .map(|k| {
                // run the simulation
                K.store(*k, atomic::Ordering::Relaxed);
                run(Some(sim_length), "setting_column_small.png");
                // create a complex valued buffer from the density plot
                let mut buff: Vec<Complex64> = HISTORY
                    .read()
                    .plot_density
                    .par_iter()
                    .map(|[_t, den]| Complex64::new(*den, 0.))
                    .collect();
                // aply fast fourier transform on the buffer
                FftPlanner::new()
                    .plan_fft_forward(buff.len())
                    .process(&mut buff);

                // calculate power spectrum and corresponding frequencies
                let powers: Vec<f64> = buff.par_iter().map(|c| c.norm_sqr()).collect();
                let n = powers.len();
                let powers_half: Vec<f64> =
                    powers.iter().skip(1).take(n / 2 + 1).cloned().collect();
                let freqencies: Vec<f64> = (1..(n / 2 + 1))
                    .map(|i| ((i as f64) * samples_hz) / (n as f64))
                    .collect();
                let peak = freqencies[powers_half
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap()];
                // add the power spectrum to the plot of all spectra
                power_spectra
                    .add_trace(Scatter::new(freqencies, powers_half).name(format!("k={}", k)));
                println!("k:{},  peak:{}, error:{}", k, peak, powers[0]);
                (peak, powers[0])
            })
            .unzip();
        k_to_peak.add_trace(
            Scatter::new(ks.clone(), peak_freq).name(format!("Peak Frequency, ν={}", nu)),
        );
        k_to_err.add_trace(Scatter::new(ks.clone(), err).name(format!("Zero Frequency, ν={}", nu)));
    });

    // write out files
    let time_stamp = get_timestamp();
    power_spectra.write_html(format!(
        "analysis/plot_ks_power_spectra_{}.html",
        time_stamp
    ));
    k_to_peak.write_html(format!("analysis/plot_ks_to_freq_{}.html", time_stamp));
    k_to_err.write_html(format!("analysis/plot_ks_to_err_{}.html", time_stamp));
}

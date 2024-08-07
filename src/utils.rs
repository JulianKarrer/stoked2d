use std::time::{SystemTime, UNIX_EPOCH};

use core::sync::atomic::Ordering::Relaxed;
use glam::DVec2;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{HISTORY_FRAME_TIME, SIM_FPS};

/// Efficiently calculates the next multiple of 'multiple_of' which is greater or
/// equal to 'n'. Equivalent to the rounded up result of division.
///
/// As seen on [StackOverflow](https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c)
pub fn next_multiple(n: usize, multiple_of: usize) -> usize {
    (1 + ((n - 1) / multiple_of)) * multiple_of
}

/// Compute the average value of a slice of values
pub fn average_val(arr: &[f64]) -> f64 {
    arr.par_iter().sum::<f64>() / arr.len() as f64
}

/// Compute the maximum norm of a slice of values
pub fn max_length(values: &[DVec2]) -> f64 {
    values
        .par_iter()
        .map(|elem| elem.length())
        .reduce_with(|a, b| a.max(b))
        .unwrap()
}

/// Get the current second of the unix epoch.
pub fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Unzips a single vector of arrays of size two of floats into two vectors of floats.
pub fn unzip_f64_2(values: &[[f64; 2]]) -> (Vec<f64>, Vec<f64>) {
    values.par_iter().map(|v| (v[0], v[1])).unzip()
}

/// Give a random 'DVec2' within a given square range $[-range;range]^2$
pub fn random_vec2(range: f64) -> DVec2 {
    DVec2::new(
        thread_rng().gen_range(-range..=range),
        thread_rng().gen_range(-range..=range),
    )
}

/// Yield `resolution + 1` many floats equally spaced between `min` and `max` (both inclusive)
pub fn linspace(min: f64, max: f64, resolution: usize) -> Vec<f64> {
    assert!(max > min);
    let step = (max - min) / (resolution as f64);
    let mut res = vec![];
    for i in 0..=resolution {
        res.push(i as f64 * step + min)
    }
    res
}

/// Given `data`, which is a slice of `[x,y]` arrays, integrate the area between
/// the y-values and the `should_be` value (i.e. the absolute error)
/// using the trapezoidal rule.
pub fn integrate_abs_error(data: &[[f64; 2]], should_be: f64) -> f64 {
    assert!(
        data.len() > 1,
        "At least two datapoints must be given to integrate an error."
    );
    // loop over pairs of subsequent datapoints
    data.iter()
        .zip(data.iter().skip(1))
        .map(|(a, b)| {
            // implement the trapezoidal rule for the interval from a to b
            let err_a = (a[1] - should_be).abs();
            let err_b = (b[1] - should_be).abs();
            (b[0] - a[0]) * (err_a + err_b) * 0.5
        })
        .sum()
}

/// Given `data`, which is a slice of `[x,y]` arrays, integrate the squared error
/// between the y-values and the `should_be` value (i.e. the squared residuals)
/// using the trapezoidal rule.
pub fn integrate_squared_error(data: &[[f64; 2]], should_be: f64) -> f64 {
    assert!(
        data.len() > 1,
        "At least two datapoints must be given to integrate an error."
    );
    // loop over pairs of subsequent datapoints
    data.iter()
        .zip(data.iter().skip(1))
        .map(|(a, b)| {
            // implement the trapezoidal rule for the interval from a to b
            let err_a = (a[1] - should_be) * (a[1] - should_be);
            let err_b = (b[1] - should_be) * (b[1] - should_be);
            (b[0] - a[0]) * (err_a + err_b) * 0.5
        })
        .sum()
}

/// For values of RGBA in bytes, determine if the colour represented is black.
pub fn is_black(r: u8, g: u8, b: u8, a: u8) -> bool {
    a > 100 && r < 10 && g < 10 && b < 10
}
/// For values of RGBA in bytes, determine if the colour represented is blue.
pub fn is_blue(r: u8, g: u8, b: u8, a: u8) -> bool {
    a > 100 && b > (r + 10) && b > (g + 10)
}

// Create a progressbar
pub fn create_progressbar(run_for_t: Option<f32>) -> Option<ProgressBar> {
    if run_for_t.is_some() {
        Some({
            let progress =
                ProgressBar::new((run_for_t.unwrap() / HISTORY_FRAME_TIME).ceil() as u64);
            progress.set_message(format!("{} ITERS/S", SIM_FPS.load(Relaxed)));
            progress.set_style(
      ProgressStyle::with_template("{msg} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} (ETA {eta})").unwrap()
    );
            progress
        })
    } else {
        None
    }
}

/// Get the current timestamp in microseconds
pub fn timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Error getting current time")
        .as_micros()
}
/// Convert microseconds to seconds
pub fn micros_to_seconds(timespan: u128) -> f64 {
    0.000_001 * timespan as f64
}
/// Convert seconds to microseconds
pub fn seconds_to_micros(timespan: f64) -> u128 {
    (1_000_000.0 * timespan).round() as u128
}

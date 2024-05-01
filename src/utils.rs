use std::time::{SystemTime, UNIX_EPOCH};

use glam::DVec2;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::GRAVITY;

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

pub fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn unzip_f64_2(values: &[[f64; 2]]) -> (Vec<f64>, Vec<f64>) {
    values.par_iter().map(|v| (v[0], v[1])).unzip()
}

/// Compute the Hamiltonian of the system
pub fn hamiltonian(pos: &[DVec2], vel: &[DVec2], mass: f64, min_height: f64) -> f64 {
    let g = -GRAVITY.load(atomic::Ordering::Relaxed);
    let lowest = mass * g * min_height * pos.len() as f64;
    let ham: f64 = pos
        .par_iter()
        .zip(vel)
        .map(|(p, v)| {
            // the gravitational potential of the particle
            let pot_grav = mass * g * p.y;
            let pot_kinetic = 0.5 * mass * v.length_squared();
            pot_grav + pot_kinetic
        })
        .sum();
    (ham - lowest) / lowest.abs()
}

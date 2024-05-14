use stoked2d::{simulation::Boundary, sph::KernelType, GAMMA_1, GAMMA_2};

fn main() {
    Boundary::calculate_gammas(&KernelType::GaussSpline3);
    println!(
        "For a Cubic Spline Kernel: \n gamma 1: {}, gamma 2: {}",
        GAMMA_1.load(atomic::Ordering::Relaxed),
        GAMMA_2.load(atomic::Ordering::Relaxed)
    )
}

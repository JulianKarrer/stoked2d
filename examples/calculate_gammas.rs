use glam::DVec2;
use stoked2d::{sph::KernelType, H};

fn main() {
    // assuming a kernel support of 2:
    // calculate the setting of a particle at the origin surrounded by fluid with
    // ideal sampling on one layer of boundary particles
    let h = H;
    let mut x = -3. * h;
    let eps = 10e-12;
    let mut pos: Vec<DVec2> = vec![];
    while x <= 3. * h + eps {
        pos.push(DVec2::new(x, 3. * h));
        pos.push(DVec2::new(x, 2. * h));
        pos.push(DVec2::new(x, 1. * h));
        pos.push(DVec2::new(x, 0.));
        x += h;
    }
    let bdy: Vec<DVec2> = (-2..=2).map(|i| DVec2::new((i as f64) * H, -H)).collect();
    println!("pos: {:?}", pos);
    println!("bdy: {:?}", bdy);

    let knl = KernelType::GaussSpline3;
    let x_i = DVec2::ZERO;
    let fluid_sum: f64 = pos.iter().map(|x_j| knl.w(&x_i, x_j)).sum();
    let bdy_sum: f64 = bdy.iter().map(|x_j| knl.w(&x_i, x_j)).sum();
    let gamma_1 = ((1. / (H * H)) - fluid_sum) / bdy_sum;
    println!(
        "gamma 1: {}, fluidsum: {}, bdysum:{}",
        gamma_1, fluid_sum, bdy_sum
    );

    let grad_fluid_sum: DVec2 = pos.iter().map(|x_j| knl.dw(&x_i, x_j)).sum::<DVec2>();
    let grad_bdy_sum: DVec2 = bdy.iter().map(|x_j| knl.dw(&x_i, x_j)).sum::<DVec2>();
    let gamma_2 = (grad_fluid_sum.dot(grad_bdy_sum)) / (grad_bdy_sum.dot(grad_bdy_sum));
    println!(
        "gamma 2: {}, grad_fluidsum: {}, grad_bdysum:{}",
        gamma_2, grad_fluid_sum, grad_bdy_sum
    );
}

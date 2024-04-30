use crate::*;
use std::f64::consts::TAU;
use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::Debug;
use strum_macros::EnumIter;

pub const KERNEL_CUBIC_NORMALIZE: f64 = 5.0 / (14.0 * PI * H * H);

#[derive(Clone, Copy, EnumIter, PartialEq)]
pub enum KernelType {
    GaussSpline3,
    Spiky,
    DoubleCosine,
}

impl Debug for KernelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussSpline3 => write!(f, "Cubic Gauss Spline"),
            Self::Spiky => write!(f, "Spiky Kernel"),
            Self::DoubleCosine => write!(f, "Double Cosine"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct SphKernel {
    pub density: KernelType,
    pub pressure: KernelType,
    pub viscosity: KernelType,
}

impl Default for SphKernel {
    fn default() -> Self {
        Self {
            density: KernelType::DoubleCosine,
            pressure: KernelType::DoubleCosine,
            viscosity: KernelType::DoubleCosine,
        }
    }
}

impl KernelType {
    pub fn w(&self, x_i: &DVec2, x_j: &DVec2) -> f64 {
        match self {
            KernelType::GaussSpline3 => {
                const ALPHA: f64 = 5.0 / (14.0 * PI * H * H);
                let q = x_i.distance(*x_j) / H;
                let t1 = (1.0 - q).max(0.0);
                let t2 = (2.0 - q).max(0.0);
                ALPHA * (t2 * t2 * t2 - 4.0 * t1 * t1 * t1)
            }
            KernelType::Spiky => {
                const KS_5: f64 = (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT);
                const ALPHA: f64 = 10.0 / PI * KS_5;
                let r = x_i.distance(*x_j);
                let t1 = (KERNEL_SUPPORT - r).max(0.0);
                ALPHA * t1 * t1 * t1
            }
            KernelType::DoubleCosine => {
                const ALPHA: f64 = PI / ((3. * PI * PI - 16.) * KERNEL_SUPPORT * KERNEL_SUPPORT);
                let r = x_i.distance(*x_j);
                let s = r / H;
                if r <= KERNEL_SUPPORT {
                    ALPHA * (4. * (FRAC_PI_2 * s).cos() + (PI * s).cos() + 3.)
                } else {
                    0.
                }
            }
        }
    }

    pub fn dw(&self, x_i: &DVec2, x_j: &DVec2) -> DVec2 {
        match self {
            KernelType::GaussSpline3 => {
                const ALPHA: f64 = 5.0 / (14.0 * PI * H * H);
                let q = x_i.distance(*x_j) / H;
                let t1 = (1.0 - q).max(0.0);
                let t2 = (2.0 - q).max(0.0);
                (*x_i - *x_j).normalize_or_zero() * ALPHA * (-3.0 * t2 * t2 + 12.0 * t1 * t1)
            }
            KernelType::Spiky => {
                const KS_5: f64 = (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT)
                    * (1. / KERNEL_SUPPORT);
                const ALPHA: f64 = -10.0 / PI * KS_5;
                let r = x_i.distance(*x_j);
                let t2 = (KERNEL_SUPPORT - r).max(0.0);
                ALPHA * (*x_i - *x_j).normalize_or_zero() * t2 * t2
            }
            KernelType::DoubleCosine => {
                const ALPHA: f64 = PI / ((3. * PI * PI - 16.) * KERNEL_SUPPORT * KERNEL_SUPPORT);
                let r = x_i.distance(*x_j);
                let s = r / H;
                if r <= KERNEL_SUPPORT {
                    ALPHA
                        * (*x_i - *x_j).normalize_or_zero()
                        * (-2. * TAU / KERNEL_SUPPORT * (FRAC_PI_2 * s).sin()
                            - TAU / KERNEL_SUPPORT * (PI * s).sin())
                } else {
                    DVec2::ZERO
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
    use test_case::test_case;

    const TEST_RANGE: f64 = 2.1 * H;
    const ASSUMED_RANGE: f64 = KERNEL_SUPPORT;
    const MC_TEST_RUNS: usize = 100_000_000;
    const RIEMANN_INTEGRAL_RES: usize = 10_000;
    const RIEMANN_INTEGRAL_ACCEPTED_ERROR: f64 = 10e-8;
    const MC_TEST_RUNS_ACCEPTED_REL_ERROR: f64 = 10e-3;
    const IDEAL_SAMPLING_ACCEPTED_ERROR: f64 = 10e-3;
    const F64_EQUALITY_EPSILON: f64 = 10e-12;

    fn get_ideal_sampled_pos_around_origin() -> Vec<DVec2> {
        let max_extend = ((KERNEL_SUPPORT / H).ceil() + 1.) as i32;
        (-max_extend..=max_extend)
            .map(|x| {
                (-max_extend..=max_extend)
                    .map(|y| DVec2::new(x as f64 * H, y as f64 * H))
                    .collect::<Vec<DVec2>>()
            })
            .flatten()
            .collect::<Vec<DVec2>>()
    }

    fn random_vec2(range: f64) -> DVec2 {
        DVec2::new(
            thread_rng().gen_range(-range..range),
            thread_rng().gen_range(-range..range),
        )
    }

    fn randspace(min: f64, max: f64, resolution: usize) -> Vec<f64> {
        assert!(max > min);
        (0..resolution)
            .map(|_| thread_rng().gen_range(min..max))
            .collect::<Vec<f64>>()
    }

    fn linspace(min: f64, max: f64, resolution: usize) -> Vec<f64> {
        assert!(max > min);
        let step = (max - min) / (resolution as f64);
        let res_half = (resolution / 2) as i64;
        (-res_half..=res_half)
            .map(|i| i as f64 * step)
            .collect::<Vec<f64>>()
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_positivity(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            assert!(knl.w(&random_vec2(TEST_RANGE), &random_vec2(TEST_RANGE)) >= 0.0)
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_positivity(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            if a.distance(b) <= ASSUMED_RANGE - F64_EQUALITY_EPSILON {
                assert!(
                    a.distance(b) < F64_EQUALITY_EPSILON || knl.dw(&a, &b).length_squared() >= 0.0,
                    "a:{} b:{} dw:{} distance:{}",
                    a,
                    b,
                    knl.dw(&a, &b),
                    a.distance(b)
                )
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_compact(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            let (a, b) = (
                random_vec2(2. * ASSUMED_RANGE),
                random_vec2(2. * ASSUMED_RANGE),
            );
            if a.distance(b) > ASSUMED_RANGE {
                assert!(knl.w(&a, &b).abs() < F64_EQUALITY_EPSILON)
            } else {
                assert!(knl.w(&a, &b) >= 0.0)
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_compact(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            if a.distance(b) > ASSUMED_RANGE {
                assert!(knl.dw(&a, &b).length_squared() < F64_EQUALITY_EPSILON)
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_symmetry(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            assert_eq!(knl.w(&a, &b), knl.w(&b, &a))
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_antisymmetry(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            assert_eq!(knl.dw(&a, &b), -knl.dw(&b, &a))
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_integral_mc(knl: &KernelType) {
        let o = DVec2::ZERO;
        let mut sum = 0.0;
        for _ in 0..MC_TEST_RUNS {
            sum += knl.w(&o, &random_vec2(ASSUMED_RANGE));
        }
        sum /= MC_TEST_RUNS as f64;
        sum *= (ASSUMED_RANGE * 2.0).powi(2);
        let relative_error = sum / 1.0 - 1.0;
        assert!(
            relative_error.abs() < MC_TEST_RUNS_ACCEPTED_REL_ERROR,
            "{}",
            sum
        );
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_integral_mc(knl: &KernelType) {
        let o = DVec2::ZERO;
        let mut sum = DVec2::ZERO;
        for _ in 0..MC_TEST_RUNS {
            sum += knl.dw(&o, &random_vec2(ASSUMED_RANGE));
        }
        sum /= MC_TEST_RUNS as f64;
        sum *= (ASSUMED_RANGE * 2.0).powi(2);
        let max_error_component = sum.max_element();
        assert!(
            max_error_component < MC_TEST_RUNS_ACCEPTED_REL_ERROR,
            "{}",
            sum
        );
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_riemann_integral_(knl: &KernelType) {
        let xs = linspace(-ASSUMED_RANGE, ASSUMED_RANGE, RIEMANN_INTEGRAL_RES);
        let ys = linspace(-ASSUMED_RANGE, ASSUMED_RANGE, RIEMANN_INTEGRAL_RES);
        let mut sum = 0.;
        for x in &xs {
            for y in &ys {
                sum += knl.w(&DVec2::ZERO, &DVec2::new(*x, *y))
            }
        }
        sum /= (RIEMANN_INTEGRAL_RES * RIEMANN_INTEGRAL_RES) as f64;
        sum *= (ASSUMED_RANGE * 2.).powi(2);
        assert!(
            (sum - 1.).abs() < RIEMANN_INTEGRAL_ACCEPTED_ERROR,
            "sum: {}",
            sum
        )
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_and_dw_consistency(knl: &KernelType) {
        // https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
        let range = 0.90 * KERNEL_SUPPORT;
        let resolution = 1000;
        let xs = linspace(-range, range, resolution);
        let ys = linspace(-range, range, resolution);
        let delta = H / 64.;
        for x in &xs {
            for y in &ys {
                let f_xp2d = knl.w(&DVec2::ZERO, &DVec2::new(*x + 2. * delta, *y));
                let f_xpd = knl.w(&DVec2::ZERO, &DVec2::new(*x + delta, *y));
                let f_xmd = knl.w(&DVec2::ZERO, &DVec2::new(*x - delta, *y));
                let f_xm2d = knl.w(&DVec2::ZERO, &DVec2::new(*x - 2. * delta, *y));
                let f_yp2d = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y + 2. * delta));
                let f_ypd = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y + delta));
                let f_ymd = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y - delta));
                let f_ym2d = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y - 2. * delta));
                let dx = -(-f_xp2d + 8. * f_xpd - 8. * f_xmd + f_xm2d) / (12. * delta);
                let dy = -(-f_yp2d + 8. * f_ypd - 8. * f_ymd + f_ym2d) / (12. * delta);
                let dw = knl.dw(&DVec2::ZERO, &DVec2::new(*x, *y));
                assert!(
                    dx.abs() < 0.1 || (dw.x / dx - 1.0).abs() < 10e-2,
                    "x failed! dx:{} dy:{} dw:{} at:[{},{}]",
                    dx,
                    dy,
                    dw,
                    x,
                    y
                );
                assert!(
                    dy.abs() < 0.1 || (dw.y / dy - 1.0).abs() < 10e-2,
                    "y failed! dx:{} dy:{} dw:{} at:[{},{}]",
                    dx,
                    dy,
                    dw,
                    x,
                    y
                );
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_and_dw_consistency_random(knl: &KernelType) {
        // https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
        let range = 0.90 * KERNEL_SUPPORT;
        let resolution = 1000;
        let xs = randspace(-range, range, resolution);
        let ys = randspace(-range, range, resolution);
        let delta = H / 1024.;
        for x in &xs {
            for y in &ys {
                let f_xp2d = knl.w(&DVec2::ZERO, &DVec2::new(*x + 2. * delta, *y));
                let f_xpd = knl.w(&DVec2::ZERO, &DVec2::new(*x + delta, *y));
                let f_xmd = knl.w(&DVec2::ZERO, &DVec2::new(*x - delta, *y));
                let f_xm2d = knl.w(&DVec2::ZERO, &DVec2::new(*x - 2. * delta, *y));
                let f_yp2d = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y + 2. * delta));
                let f_ypd = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y + delta));
                let f_ymd = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y - delta));
                let f_ym2d = knl.w(&DVec2::ZERO, &DVec2::new(*x, *y - 2. * delta));
                let dx = -(-f_xp2d + 8. * f_xpd - 8. * f_xmd + f_xm2d) / (12. * delta);
                let dy = -(-f_yp2d + 8. * f_ypd - 8. * f_ymd + f_ym2d) / (12. * delta);
                let dw = knl.dw(&DVec2::ZERO, &DVec2::new(*x, *y));
                assert!(
                    dx.abs() < 0.1 || (dw.x / dx - 1.0).abs() < 10e-2,
                    "x failed! dx:{} dy:{} dw:{} at:[{},{}]",
                    dx,
                    dy,
                    dw,
                    x,
                    y
                );
                assert!(
                    dy.abs() < 0.1 || (dw.y / dy - 1.0).abs() < 10e-2,
                    "y failed! dx:{} dy:{} dw:{} at:[{},{}]",
                    dx,
                    dy,
                    dw,
                    x,
                    y
                );
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_riemann_integral_(knl: &KernelType) {
        let xs = linspace(-ASSUMED_RANGE, ASSUMED_RANGE, RIEMANN_INTEGRAL_RES);
        let ys = linspace(-ASSUMED_RANGE, ASSUMED_RANGE, RIEMANN_INTEGRAL_RES);
        let mut sum = DVec2::ZERO;
        for x in &xs {
            for y in &ys {
                sum += knl.dw(&DVec2::ZERO, &DVec2::new(*x, *y))
            }
        }
        sum /= (RIEMANN_INTEGRAL_RES * RIEMANN_INTEGRAL_RES) as f64;
        sum *= (ASSUMED_RANGE * 2.).powi(2);
        assert!(
            sum.length() < RIEMANN_INTEGRAL_ACCEPTED_ERROR,
            "sum: {}",
            sum
        )
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_a_to_a_is_zero(knl: &KernelType) {
        for _ in 0..MC_TEST_RUNS {
            let x = random_vec2(TEST_RANGE);
            assert_eq!(knl.dw(&x, &x), DVec2::ZERO)
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn w_ideal_volume(knl: &KernelType) {
        let pos = get_ideal_sampled_pos_around_origin();
        let x_i = DVec2::ZERO;
        let sum: f64 = pos.par_iter().map(|x_j| knl.w(&x_i, x_j)).sum();
        let relative_error = sum / (1.0 / (H * H)) - 1.0;
        assert!(
            relative_error.abs() < IDEAL_SAMPLING_ACCEPTED_ERROR,
            "sum: {}\texpected: {}\trelative error:{}\tin units of h^2: {}",
            sum,
            (1.0 / (H * H)),
            relative_error,
            (sum - (1.0 / (H * H))).abs() / (H * H)
        )
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    fn dw_ideal_sum_is_zero(knl: &KernelType) {
        let pos = get_ideal_sampled_pos_around_origin();
        let x_i = DVec2::ZERO;
        let sum: DVec2 = pos.par_iter().map(|x_j| knl.dw(&x_i, x_j)).sum();
        assert!(sum.length() < MC_TEST_RUNS_ACCEPTED_REL_ERROR)
    }

    // #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    // #[test_case(&KernelType::Spiky ; "Spiky")]
    // #[test_case(&KernelType::DoubleCosine ; "Double Cosine")]
    // /// Assert $$sum_j{ (x_i - x_j) \bigotimes \nabla W_{ij} = -\frac{1}{V_i}\cdot \mathbf{I}}$$
    // fn dw_ideal_projected_density(knl: &KernelType) {
    //     let pos = get_ideal_sampled_pos_around_origin();
    //     let x_i = DVec2::ZERO;
    //     let sum: DVec2 = pos
    //         .par_iter()
    //         .map(|x_j| knl.dw(&x_i, x_j) * (x_i - *x_j))
    //         .sum();
    //     let expected = -1.0 / (H * H) * DVec2::ONE;
    //     assert!(
    //         sum.distance(expected) < TEST_RUNS_ACCEPTED_REL_ERROR,
    //         "sum: {} expected: {}",
    //         sum,
    //         expected
    //     );
    //     // also show that the kernel derivative is reversed correctly
    //     assert!(sum.x <= 0.0);
    //     assert!(sum.y <= 0.0);
    // }
}

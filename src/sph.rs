use crate::*;
use std::f64::consts::PI;

pub const KERNEL_CUBIC_NORMALIZE: f64 = 5.0 / (14.0 * PI * H * H);

#[derive(Clone, Copy)]
pub enum KernelType {
    GaussSpline3,
    Spiky,
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
            density: KernelType::GaussSpline3,
            pressure: KernelType::Spiky,
            viscosity: KernelType::Spiky,
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
        }
    }

    pub fn dw(&self, x_i: &DVec2, x_j: &DVec2) -> DVec2 {
        match self {
            KernelType::GaussSpline3 => {
                const ALPHA: f64 = 5.0 / (14.0 * PI * H * H);
                let q = x_i.distance(*x_j) / H;
                let t1 = (1.0 - q).max(0.0);
                let t2 = (2.0 - q).max(0.0);
                let magnitude = ALPHA / (q * H) * (-3.0 * t2 * t2 + 12.0 * t1 * t1);
                if magnitude.is_finite() {
                    (*x_i - *x_j) * magnitude
                } else {
                    DVec2::ZERO
                }
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
    const ASSUMED_RANGE: f64 = 2.0 * H;
    const TEST_RUNS: usize = 100_000_000;
    const TEST_RUNS_ACCEPTED_REL_ERROR: f64 = 10e-3;
    const IDEAL_SAMPLING_ACCEPTED_ERROR: f64 = 10e-3;
    const F64_EQUALITY_EPSILON: f64 = 10e-9;

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

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn w_positivity(knl: &KernelType) {
        for _ in 0..TEST_RUNS {
            assert!(knl.w(&random_vec2(TEST_RANGE), &random_vec2(TEST_RANGE)) >= 0.0)
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn w_compact(knl: &KernelType) {
        for _ in 0..TEST_RUNS {
            let (a, b) = (
                random_vec2(2. * ASSUMED_RANGE),
                random_vec2(2. * ASSUMED_RANGE),
            );
            if a.distance(b) > ASSUMED_RANGE {
                assert!(knl.w(&a, &b).abs() < F64_EQUALITY_EPSILON)
            } else {
                assert!(knl.w(&a, &b) > 0.0)
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn dw_compact(knl: &KernelType) {
        for _ in 0..TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            if a.distance(b) > ASSUMED_RANGE {
                assert!(knl.dw(&a, &b).length_squared() < F64_EQUALITY_EPSILON)
            } else {
                assert!(a.distance(b) == 0. || knl.dw(&a, &b).length_squared() != 0.0)
            }
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn w_symmetry(knl: &KernelType) {
        for _ in 0..TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            assert_eq!(knl.w(&a, &b), knl.w(&b, &a))
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn dw_antisymmetry(knl: &KernelType) {
        for _ in 0..TEST_RUNS {
            let (a, b) = (random_vec2(TEST_RANGE), random_vec2(TEST_RANGE));
            assert_eq!(knl.dw(&a, &b), -knl.dw(&b, &a))
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn w_integral(knl: &KernelType) {
        let o = DVec2::ZERO;
        let mut sum = 0.0;
        for _ in 0..TEST_RUNS {
            sum += knl.w(&o, &random_vec2(ASSUMED_RANGE));
        }
        sum /= TEST_RUNS as f64;
        sum *= (ASSUMED_RANGE * 2.0).powi(2);
        let relative_error = sum / 1.0 - 1.0;
        assert!(
            relative_error.abs() < TEST_RUNS_ACCEPTED_REL_ERROR,
            "{}",
            sum
        );
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn dw_integral(knl: &KernelType) {
        let o = DVec2::ZERO;
        let mut sum = DVec2::ZERO;
        for _ in 0..TEST_RUNS {
            sum += knl.dw(&o, &random_vec2(ASSUMED_RANGE));
        }
        sum /= TEST_RUNS as f64;
        sum *= (ASSUMED_RANGE * 2.0).powi(2);
        let error = sum.length();
        assert!(error < TEST_RUNS_ACCEPTED_REL_ERROR, "{}", sum);
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
    fn dw_a_to_a_is_zero(knl: &KernelType) {
        for _ in 0..TEST_RUNS {
            let x = random_vec2(TEST_RANGE);
            assert_eq!(knl.dw(&x, &x), DVec2::ZERO)
        }
    }

    #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    #[test_case(&KernelType::Spiky ; "Spiky")]
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
    fn dw_ideal_sum_is_zero(knl: &KernelType) {
        let pos = get_ideal_sampled_pos_around_origin();
        let x_i = DVec2::ZERO;
        let sum: DVec2 = pos.par_iter().map(|x_j| knl.dw(&x_i, x_j)).sum();
        assert!(sum.length() < TEST_RUNS_ACCEPTED_REL_ERROR)
    }

    // #[test_case(&KernelType::GaussSpline3 ; "Cubic Spline")]
    // #[test_case(&KernelType::Spiky ; "Spiky")]
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

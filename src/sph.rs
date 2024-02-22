use std::f64::consts::PI;
use crate::*;

pub const KERNEL_CUBIC_NORMALIZE: f64 = 5.0/(14.0*PI*H*H);

pub fn kernel(x_i:&DVec2, x_j:&DVec2)->f64{
  let q = x_i.distance(*x_j)/H;
  let t1 = 1.0-q;
  let t2 = 2.0-q;
  KERNEL_CUBIC_NORMALIZE * match q {
      _ if (0.0..1.0).contains(&q) => t2*t2*t2 - 4.0*t1*t1*t1,
      _ if (1.0..2.0).contains(&q) => t2*t2*t2 ,
      _ => 0.0
  }
  // let t1 = (1.0-q).max(0.0);
  // let t2 = (2.0-q).max(0.0);
  // KERNEL_CUBIC_NORMALIZE * (t2*t2*t2 - 4.0*t1*t1*t1)
}

pub fn kernel_derivative(x_i:&DVec2, x_j:&DVec2)->DVec2{
  let q = x_i.distance(*x_j)/H;
  let t1 = 1.0-q;
  let t2 = 2.0-q;
  let magnitude = KERNEL_CUBIC_NORMALIZE/H * match q {
    _ if (0.0..1.0).contains(&q) => -3.0*t2*t2 + 12.0*t1*t1,
    _ if (1.0..2.0).contains(&q) => -3.0*t2*t2,
    _ => 0.0
  };
  if magnitude.is_finite() {(*x_i-*x_j)*magnitude} else {DVec2::ZERO}

  // let t1 = (1.0-q).max(0.0);
  // let t2 = (2.0-q).max(0.0);
  // let magnitude = KERNEL_CUBIC_NORMALIZE  / (q*H) * (-3.0*t2*t2 + 12.0*t1*t1);
  // if magnitude.is_finite() {(*x_i-*x_j)*magnitude} else {DVec2::ZERO}
}


#[cfg(test)]
mod tests {
  use approx::assert_relative_eq;
  use rand::{thread_rng, Rng};
  use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
  use super::*;

  const TEST_RANGE:f64 = 4.0*H;
  const ASSUMED_RANGE:f64 = 2.0*H;
  const TEST_RUNS:usize = 1_000_000;
  const TEST_RUNS_ACCEPTED_REL_ERROR:f64 = 0.01;
  const IDEAL_SAMPLING_ACCEPTED_ERROR:f64 = 0.01;
  const F64_EPSILON:f64 = f64::EPSILON*64.0;
  
  fn get_ideal_sampled_pos_around_origin()->Vec<DVec2>{
    let mut pos = vec![];
    let (mut x, mut y) = (-3.0*H, -3.0*H);
    while x <= 3.0*H {
      while y <= 3.0*H {
        pos.push(DVec2::new(x, y));
        y += H;
      }
      y = -3.0*H;
      x += H;
    }
    pos
  }

  fn random_vec2(range:f64)->DVec2{
    DVec2::new(
      thread_rng().gen_range(-range..range),
      thread_rng().gen_range(-range..range),
    )
  }

  #[test]
  fn kernel_positivity() {
    for _ in 0..TEST_RUNS{
      assert!(kernel(&random_vec2(TEST_RANGE), &random_vec2(TEST_RANGE))>=0.0)
    }
  }

  #[test]
  fn kernel_compact() {
    for _ in 0..TEST_RUNS{
      let (a,b) = (random_vec2(TEST_RANGE),random_vec2(TEST_RANGE));
      if a.distance(b)>ASSUMED_RANGE {
        assert_relative_eq!(kernel(&a, &b), 0.0)
      }
    }
  }

  #[test]
  fn kernel_symmetry() {
    for _ in 0..TEST_RUNS{
      let (a,b) = (random_vec2(TEST_RANGE),random_vec2(TEST_RANGE));
      assert_eq!(kernel(&a, &b), kernel(&b, &a))
    }
  }
  
  #[test]
  fn kernel_integral() {
    let o = DVec2::ZERO;
    let mut sum = 0.0;
    for _ in 0..TEST_RUNS{
      sum += kernel(&o, &random_vec2(ASSUMED_RANGE));
    }
    sum /= TEST_RUNS as f64;
    sum *= (ASSUMED_RANGE*2.0).powi(2);
    let relative_error = sum/1.0-1.0;
    assert!(relative_error.abs() < TEST_RUNS_ACCEPTED_REL_ERROR, "{}", sum);
  }

  #[test]
  fn kernel_ideal_sampling_volume() {
    let pos = get_ideal_sampled_pos_around_origin();
    let x_i = DVec2::ZERO;
    let sum:f64 = pos.par_iter().map(|x_j| kernel(&x_i, x_j)).sum();
    let relative_error = sum/(1.0/(H*H))-1.0;
    assert!(relative_error.abs() < IDEAL_SAMPLING_ACCEPTED_ERROR)
  }

  #[test]
  fn kernel_derivative_integral() {
    let o = DVec2::ZERO;
    let mut sum = 0.0;
    for _ in 0..TEST_RUNS{
      sum += kernel(&o, &random_vec2(ASSUMED_RANGE));
    }
    sum /= TEST_RUNS as f64;
    sum *= (ASSUMED_RANGE*2.0).powi(2);
    let relative_error = sum/1.0-1.0;
    assert!(relative_error.abs() < TEST_RUNS_ACCEPTED_REL_ERROR, "{}", sum);
  }

  #[test]
  fn kernel_derivative_antisymmetry() {
    for _ in 0..TEST_RUNS{
      let (a,b) = (random_vec2(TEST_RANGE),random_vec2(TEST_RANGE));
      assert_eq!(kernel_derivative(&a, &b), -kernel_derivative(&b, &a))
    }
  }
  
  #[test]
  fn kernel_derivative_a_to_a_is_zero() {
    for _ in 0..TEST_RUNS{
      let x = random_vec2(TEST_RANGE);
      assert_eq!(kernel_derivative(&x, &x), DVec2::ZERO)
    }
  }
  
  #[test]
  fn kernel_derivative_ideal_sampling_sum_is_zero() {
    let pos = get_ideal_sampled_pos_around_origin();
    let x_i = DVec2::ZERO;
    let sum:DVec2 = pos.par_iter().map(|x_j| kernel_derivative(&x_i, x_j)).sum();
    assert!(sum.length() < F64_EPSILON)
  }

  // #[test]
  // /// Assert $sum_j{ (x_i - x_j) \bigotimes \nabla W_{ij} = -\frac{1}{V_i}\cdot \mathbf{I}}$
  // fn kernel_derivative_ideal_sampling_identity() {
  //   let pos = get_ideal_sampled_pos_around_origin();
  //   let x_i = DVec2::ZERO;
  //   let sum:DVec2 = pos.par_iter().map(|x_j| { kernel_derivative(&x_i, x_j) * (x_i-*x_j)}).sum();
  //   let expected = -1.0/(H*H) * DVec2::ONE;
  //   assert!(relative_eq!(sum.distance(expected), 0.0), "sum: {} expected: {}", sum, expected);
  //   // also show that the kernel derivative is reversed correctly
  //   assert!(sum.x <= 0.0);
  //   assert!(sum.y <= 0.0);
  // }


}
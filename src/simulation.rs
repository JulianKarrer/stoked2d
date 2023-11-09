use crate::{*, sph::{kernel, kernel_derivative}, datastructure::Grid};
use std::{time::Duration, fmt::Debug};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use spin_sleep::sleep;
use atomic_enum::atomic_enum;

// MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn run(){
  let mut state = Attributes::new();
  let mut grid = Grid::new(state.pos.len());
  grid.update_grid(&state.pos, KERNEL_SUPPORT);
  let mut current_t = 0.0;
  let mut since_resort = 0;
  // reset history and add first timestep
  update_densities(&state.pos, &mut state.den, &grid);
  {  HISTORY.write().reset_and_add(&state, &grid, current_t); }
  let mut last_update_time = timestamp();
  while !REQUEST_RESTART.fetch_and(false, Relaxed) {
    // throttle
    sleep(Duration::from_micros(SIMULATION_THROTTLE_MICROS.load(Relaxed)));

    // update the datastructure and potentially resort particle attributes
    if since_resort > RESORT_ATTRIBUTES_EVERY_N.load(Relaxed) {
      state.resort(&grid);
      since_resort = 0;
    } else {since_resort += 1;}
    grid.update_grid(&state.pos, KERNEL_SUPPORT);

    // perform an update step using the selected fluid solver
    SOLVER.load(Relaxed).solve(&mut state, &mut grid, &mut current_t);

    // enforce rudimentary boundary conditions
    enforce_boundary_conditions(&mut state.pos, &mut state.vel, &mut state.acc);

    // write back the positions to the global buffer for visualization and update the FPS count
    update_densities(&state.pos, &mut state.den, &grid);
    update_fps(&mut last_update_time);
    {  HISTORY.write().add_step(&state, &grid, current_t); }
  }
}


// SOLVERS AVAILABLE USED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Perform a simulation update step using the basic SESPH solver
fn sesph(state: &mut Attributes, grid: &Grid, current_t: &mut f64){
  // update densities and pressures
  update_densities(&state.pos, &mut state.den, grid);
  update_pressures(&state.den, &mut state.prs);
  // apply external forces
  apply_gravity_and_viscosity(&state.pos, &state.vel, &mut state.acc, &state.den, grid);
  // apply pressure forces
  add_pressure_accelerations(&state.pos, &state.den, &state.prs, &mut state.acc, grid);
  // enforce rudimentary boundary conditions
  enforce_boundary_conditions(&mut state.pos, &mut state.vel, &mut state.acc);
  // perform a time step
  let dt = update_dt(&state.vel, current_t);
  time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}

/// Perform a simulation update step using a SESPH solver with splitting
fn ssesph(state: &mut Attributes, grid: &mut Grid, current_t: &mut f64){
  // apply external forces
  update_densities(&state.pos, &mut state.den, grid);
  apply_gravity_and_viscosity(&state.pos, &state.vel, &mut state.acc, &state.den, grid);
  // predict velocities and positions based on non-pressure accelerations
  let dt = update_dt(&state.vel, current_t);
  state.pos_pred = state.pos.clone();
  time_step_euler_cromer(&mut state.pos_pred, &mut state.vel, &state.acc, dt);
  enforce_boundary_conditions(&mut state.pos_pred, &mut state.vel, &mut state.acc);
  // update the grid to account for predicted positions
  grid.update_grid(&state.pos_pred, KERNEL_SUPPORT);
  // compute densities at predicted positions
  update_densities(&state.pos_pred, &mut state.den, grid);
  update_pressures(&state.den, &mut state.prs);
  // overwrite! the non-pressure accelerations with pressure accelerations
  overwrite_pressure_accelerations(&state.pos_pred, &state.den, &state.prs, &mut state.acc, grid);
  // perform another timestep
  time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}

// FUNCTIONS USED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Determines the size of the next time step in seconds using the maximum velocity of a particle
/// in the previous step, the particle spacing H and a factor LAMBDA, updating the current time and 
/// returning the dt for numerical time integration
/// 
/// This correpsponds to the Courant-Friedrichs-Lewy condition
fn update_dt(vel: &[DVec2], current_t: &mut f64)->f64{
  let max_dt = MAX_DT.load(Relaxed);
  let v_max = vel.par_iter().map(|v|v.length()).reduce_with(|a,b| a.max(b)).unwrap();
  let mut dt = (LAMBDA.load(Relaxed) * H / v_max).min(max_dt);
  if !dt.is_normal() {dt = max_dt}
  *current_t += dt;
  dt
}

/// Update the FPS counter based on the previous iterations timestamp
fn update_fps(previous_timestamp: &mut u128){
  let now = timestamp();
  SIM_FPS.fetch_update(Relaxed, Relaxed, |fps| Some(
    fps*FPS_SMOOTING + 1.0/micros_to_seconds( now - *previous_timestamp) * (1.0-FPS_SMOOTING)
  )).unwrap();
  *previous_timestamp = now;
}

/// Apply external forces such as gravity and viscosity, overwriting the accelerations vec
fn apply_gravity_and_viscosity(pos: &[DVec2], vel: &[DVec2], acc: &mut[DVec2], den: &[f64], grid: &Grid){
  // account for gravity
  let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
  // calculate viscosity
  let nu = NU.load(Relaxed);
  acc.par_iter_mut().enumerate().zip(pos).zip(vel).for_each(|(((i, a), p), v)|{
    let vis:DVec2 = nu * 2.0 * grid.query_index(i).iter().map(|j|{
      let x_i_j = *p-pos[*j];
      let v_i_j = *v-vel[*j];
      M/den[*j] * (v_i_j).dot(x_i_j)/(x_i_j.length_squared() + 0.01*H*H) * kernel_derivative(p, &pos[*j])
    }).reduce(|a,b|a+b).unwrap_or(DVec2::ZERO);
    assert!(vis.is_finite());
    *a = acc_g + vis
  });
}

/// Update the densities at each particle
fn update_densities(pos: &[DVec2], den: &mut[f64], grid: &Grid){
  pos.par_iter().enumerate().zip(den).for_each(|((i, x_i), rho_i)|{
    *rho_i = grid.query_index(i).iter().map(|x_j| M*kernel(x_i, &pos[*x_j])).sum::<f64>();
  });
}

/// Update the pressures at each particle using densities
fn update_pressures(den: &[f64], prs:&mut[f64]){
  let k = K.load(Relaxed);
  let rho_0 = RHO_ZERO.load(Relaxed);
  let eq = PRESSURE_EQ.load(Relaxed);
  prs.par_iter_mut().zip(den).for_each(|(p_i, rho_i)|{
    *p_i = match eq {
      PressureEquation::Absolute => k*(rho_i - rho_0),
      PressureEquation::Relative => k*(rho_i / rho_0 - 1.0),
      PressureEquation::ClampedRelative => (k*(rho_i / rho_0 - 1.0)).max(0.0),
      PressureEquation::Compressible => k*((rho_i / rho_0).powi(7) - 1.0),
      PressureEquation::ClampedCompressible => (k*((rho_i / rho_0).powi(7) - 1.0)).max(0.0),
    }
  });
}

/// Compute pressure accelerations from the momentum-preserving SPH approximation of the density gradient,
/// adding the result to the current accelerations
fn add_pressure_accelerations(pos: &[DVec2], den: &[f64], prs: &[f64], acc: &mut[DVec2], grid: &Grid){
  pos.par_iter().enumerate().zip(prs).zip(den).zip(acc)
  .for_each(|((((i, x_i), p_i), rho_i), acc)|{
    *acc += - grid.query_index(i).iter().map(|j| 
      M*(p_i/(rho_i*rho_i) + prs[*j]/(den[*j]*den[*j])) * kernel_derivative(x_i, &pos[*j])
    ).sum::<DVec2>();
  })
}

/// Compute pressure accelerations from the momentum-preserving SPH approximation of the density gradient,
/// overwriting the current accelerations
fn overwrite_pressure_accelerations(pos: &[DVec2], den: &[f64], prs: &[f64], acc: &mut[DVec2], grid: &Grid){
  pos.par_iter().enumerate().zip(prs).zip(den).zip(acc)
  .for_each(|((((i, x_i), p_i), rho_i), acc)|{
    *acc = - grid.query_index(i).iter().map(|j| 
      M*(p_i/(rho_i*rho_i) + prs[*j]/(den[*j]*den[*j])) * kernel_derivative(x_i, &pos[*j])
    ).sum::<DVec2>();
  })
}

/// Perform a numerical time integration step using the symplectic Euler or Eurler-Cromer integration scheme.
fn time_step_euler_cromer(pos: &mut[DVec2], vel: &mut[DVec2], acc: &[DVec2], dt:f64){
  pos.par_iter_mut().zip(vel).zip(acc).for_each(|((p,v),a)|{
    *v += *a * dt;
    *p += *v * dt;
  })
}

pub fn average_density(den: &[f64])->f64{
  den.par_iter().sum::<f64>()/den.len() as f64
}

/// Harshly enforce rudimentary boundary conditions by simply setting velocity and acceleration of 
/// particles penetrating the boundary to zero while adjusting ther positions in a direction 
/// orthogonal to the boundary to be within bounds.
fn enforce_boundary_conditions(pos: &mut[DVec2], vel: &mut[DVec2], acc: &mut[DVec2]){
  pos.par_iter_mut().zip(vel).zip(acc).for_each(|((p, v), a)| {
    if p.x < BOUNDARY[0].x {p.x = BOUNDARY[0].x; a.x=0.0; v.x=0.0;} 
    if p.y < BOUNDARY[0].y {p.y = BOUNDARY[0].y; a.y=0.0; v.y=0.0;} 
    if p.x > BOUNDARY[1].x {p.x = BOUNDARY[1].x; a.x=0.0; v.x=0.0;} 
    if p.y > BOUNDARY[1].y {p.y = BOUNDARY[1].y; a.y=0.0; v.y=0.0;} 
  })
}


// STRUCTURE DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[atomic_enum]
#[derive(PartialEq)]
/// A fluid solver, implementing a single simulation step of some SPH method.
pub enum Solver{
  SESPH,
  SSESPH
}

impl Solver{
  /// Perform a single simulation step using some SPH method.
  /// 
  /// The solver can rely on current particle positions and velocities to be accurate.
  /// Accelerations, densities and pressures etc. are not resorted and must be assumed to be
  /// uninitialized garbage. 
  /// 
  /// The grid can also be assumed to be accurate.
  fn solve(&self, state: &mut Attributes, grid: &mut Grid, current_t: &mut f64){
    match self {
      Solver::SESPH => sesph(state, grid, current_t),
      Solver::SSESPH => ssesph(state, grid, current_t),
    }
  }
}

#[atomic_enum]
#[derive(PartialEq)]
/// An equation relating measured density deviation to pressure, or stress to strain.
/// While Clamped equations produce only pressures > 0 counteracting compression, the respective
/// unclamped version can lead to negative pressures, creating attraction between particles in
/// low density regions. The global static stiffness K may factor into these equations.
pub enum PressureEquation{
  Absolute,
  Relative,
  ClampedRelative,
  Compressible,
  ClampedCompressible,
}

impl std::fmt::Display for PressureEquation{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", match self{
      PressureEquation::Absolute => "k·(ρᵢ-ρ₀)",
      PressureEquation::Relative => "k·(ρᵢ/ρ₀-1)",
      PressureEquation::ClampedRelative => "k·(ρᵢ/ρ₀-1).max(0)",
      PressureEquation::Compressible => "k·((ρᵢ/ρ₀)⁷-1)",
      PressureEquation::ClampedCompressible => "k·((ρᵢ/ρ₀)⁷-1)",
    })
  }
}

/// Holds all particle data as a struct of arrays
pub struct Attributes{
  pub pos:Vec<DVec2>, 
  pub vel:Vec<DVec2>, 
  pub acc:Vec<DVec2>, 
  pub prs:Vec<f64>,
  pub den:Vec<f64>,
  pub pos_pred:Vec<DVec2>, 
}

impl Attributes{
  /// Initialize a new set of particles attributes, filling the area within the
  /// box given by FLUID with particles using spacing H
  fn new()->Self{
    // estimate the number of particles beforehand for allocation
    let n: usize = ((FLUID[1].x-FLUID[0].x)/(H) + 1.0).ceil() as usize * ((FLUID[1].y-FLUID[0].y)/(H) + 1.0).ceil() as usize;
    println!("{}",n);
    // allocation
    let mut pos:Vec<DVec2> = Vec::with_capacity(n); 
    let mut vel:Vec<DVec2> = Vec::with_capacity(n); 
    let mut acc:Vec<DVec2> = Vec::with_capacity(n); 
    let pos_pred:Vec<DVec2> = Vec::with_capacity(n); 
    // initialization
    let mut x = FLUID[0].x;
    let mut y = FLUID[0].y;
    while x <= FLUID[1].x {
      while y <= FLUID[1].y{
        pos.push(DVec2::new(x, y));
        vel.push(DVec2::ZERO);
        acc.push(DVec2::ZERO);
        y += H;
      }
      y = FLUID[0].y;
      x += H;
    }
    let prs = vec![0.0;pos.len()];
    let den = vec![M/(H*H);pos.len()];
    // return new set of attributes
    Self { pos, vel, acc, prs, den, pos_pred }
  }

  /// Resort all particle attributes according to some given order, which must be
  /// a permutation of (0..NUMBER_OF_PARTICLES). This is meant to eg. improve cache-hit-rates
  /// by employing the same sorting as the acceleration datastructure for neighbourhood queries.
  fn resort(&mut self, grid: &Grid){
    // extract the order of the attributes according to cell-wise z-ordering
    let order:Vec<usize> = grid.handles.par_iter().map(|h|h.index).collect();
    debug_assert!(order.len() == self.pos.len());
    // re-order relevatn particle attributes in accordance with the order
    self.pos = order.par_iter().map(|i| self.pos[*i]).collect();
    self.vel = order.par_iter().map(|i| self.vel[*i]).collect();
  }
}

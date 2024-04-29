use crate::{*, sph::KernelType, datastructure::Grid};
use std::{fmt::Debug, time::Duration};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use atomic_enum::atomic_enum;

use self::{gui::gui::{REQUEST_RESTART, SIMULATION_TOGGLE}, utils::average_val};

// MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn run()->bool{
  let mut state = Attributes::new();
  let mut grid = Grid::new(state.pos.len());
  grid.update_grid(&state.pos, KERNEL_SUPPORT);
  // state.resort(&grid);
  let boundary = Boundary::new(BOUNDARY_LAYER_COUNT);
  let mut current_t = 0.0;
  let mut since_resort = 0;
  // reset history and add first timestep
  update_densities(&state.pos, &mut state.den, &grid, &boundary, &KernelType::GaussSpline3);
  {  HISTORY.write().reset_and_add(&state, &grid, &boundary.pos, current_t); }
  let mut last_update_time = timestamp();
  let mut last_gui_update_t = 0.0f64;
  while !*REQUEST_RESTART.read() {
    // wait if requested
    while *SIMULATION_TOGGLE.read() { thread::sleep(Duration::from_millis(10)); }
    // update the datastructure and potentially resort particle attributes
    if since_resort > {*RESORT_ATTRIBUTES_EVERY_N.read()} {
      state.resort(&grid);
      since_resort = 0;
    } else {since_resort += 1;}
    grid.update_grid(&state.pos, KERNEL_SUPPORT);

    // perform an update step using the selected fluid solver
    let kernels = {*SPH_KERNELS.read()};
    SOLVER.load(Relaxed).solve(&mut state, &grid, &mut current_t, &boundary, &kernels);
    enforce_boundary_conditions(&mut state.pos, &mut state.vel, &mut state.acc);

    // write back the positions to the global buffer for visualization and update the FPS count
    update_fps(&mut last_update_time);
    if current_t - last_gui_update_t > FRAME_TIME.into(){
      last_gui_update_t = current_t;
      {  HISTORY.write().add_step(&state, &grid, current_t); }
    }
  }
  *REQUEST_RESTART.write() = false;
  true
}


// SOLVERS AVAILABLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Perform a simulation update step using the basic SESPH solver
fn sesph(state: &mut Attributes, grid: &Grid, current_t: &mut f64, boundary: &Boundary, knls:&SphKernel){
  // update densities and pressures
  update_densities(&state.pos, &mut state.den, grid, boundary, &knls.density);
  update_pressures(&state.den, &mut state.prs);
  // apply external forces
  apply_gravity_and_viscosity(&state.pos, &state.vel, &mut state.acc, &state.den, grid, &knls.viscosity);
  // apply pressure forces
  add_pressure_accelerations(&state.pos, &state.den, &state.prs, &mut state.acc, grid, boundary, &knls.pressure);
  // perform a time step
  let dt = update_dt(&state.vel, current_t);
  time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}

/// Perform a simulation update step using a SESPH solver with splitting
fn ssesph(state: &mut Attributes, grid: &Grid, current_t: &mut f64, boundary: &Boundary, knls:&SphKernel){
  // apply external forces
  apply_gravity_and_viscosity(&state.pos, &state.vel, &mut state.acc, &state.den, grid, &knls.viscosity);
  // predict velocities and densities based on non-pressure accelerations
  let dt = update_dt(&state.vel, current_t);
  time_step_explicit_euler_one_quantity(&mut state.vel, &state.acc, dt);
  predict_densities(&mut state.den, &state.vel, &state.pos, grid, boundary, dt, &knls.density);
  // calculate pressure forces/accelerations from the predicted densities
  update_pressures(&state.den, &mut state.prs);
  overwrite_pressure_accelerations(&state.pos, &state.den, &state.prs, &mut state.acc, grid, boundary, &knls.pressure);
  // refine the predicted velocity using this pressure acceleration and update positions
  time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}


/// Perform a simulation update step using a SESPH solver with splitting
fn isesph(state: &mut Attributes, grid: &Grid, current_t: &mut f64, boundary: &Boundary, knls:&SphKernel){
  // apply external forces
  apply_gravity_and_viscosity(&state.pos, &state.vel, &mut state.acc, &state.den, grid, &knls.viscosity);
  // predict velocities and densities based on non-pressure accelerations
  let dt = update_dt(&state.vel, current_t);
  time_step_explicit_euler_one_quantity(&mut state.vel, &state.acc, dt);
  let rho_zero = RHO_ZERO.load(Relaxed);
  let max_rho_dev = MAX_RHO_DEVIATION.load(Relaxed);
  loop{
    predict_densities(&mut state.den, &state.vel, &state.pos, grid, boundary, dt, &knls.density);
    // calculate pressure forces/accelerations from the predicted densities
    update_pressures(&state.den, &mut state.prs);
    overwrite_pressure_accelerations(&state.pos, &state.den, &state.prs, &mut state.acc, grid, boundary, &knls.pressure);
    // refine the velocity prediction using the predicted pressure accelerations
    time_step_explicit_euler_one_quantity(&mut state.vel, &state.acc, dt);
    if average_val(&state.den)/rho_zero-1.0 < max_rho_dev || *REQUEST_RESTART.read() {
      break;
    }
  }
  // refine the predicted velocity using this pressure acceleration and update positions
  time_step_explicit_euler_one_quantity(&mut state.pos, &state.vel, dt);
  // time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}

// FUNCTIONS USED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Determines the size of the next time step in seconds using the maximum velocity of a particle
/// in the previous step, the particle spacing H and a factor LAMBDA, updating the current time and 
/// returning the dt for numerical time integration
/// 
/// This correpsponds to the Courant-Friedrichs-Lewy condition
fn update_dt(vel: &[DVec2], current_t: &mut f64)->f64{
  let v_max = vel.par_iter().map(|v|v.length()).reduce_with(|a,b| a.max(b)).unwrap();
  let mut dt = (LAMBDA.load(Relaxed) * H / v_max).min(MAX_DT.load(Relaxed));
  if v_max < VELOCITY_EPSILON || !dt.is_normal() {dt = INITIAL_DT.load(Relaxed)}
  *current_t += dt;
  dt
}

/// Update the FPS counter based on the previous iterations timestamp
pub fn update_fps(previous_timestamp: &mut u128){
  let now = timestamp();
  SIM_FPS.fetch_update(Relaxed, Relaxed, |fps| Some(
    fps*FPS_SMOOTING + 1.0/micros_to_seconds( now - *previous_timestamp) * (1.0-FPS_SMOOTING)
  )).unwrap();
  *previous_timestamp = now;
}

/// Apply external forces such as gravity and viscosity, overwriting the accelerations vec
fn apply_gravity_and_viscosity(pos: &[DVec2], vel: &[DVec2], acc: &mut[DVec2], den: &[f64], grid: &Grid, knl:&KernelType){
  // account for gravity
  let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
  // calculate viscosity
  let nu = NU.load(Relaxed);
  acc.par_iter_mut().enumerate().zip(pos).zip(vel).for_each(|(((i, a), p), v)|{
    let vis:DVec2 = nu * 2.0 * grid.query_index(i).iter().map(|j|{
      let x_i_j = *p-pos[*j];
      let v_i_j = *v-vel[*j];
      M/den[*j] * (v_i_j).dot(x_i_j)/(x_i_j.length_squared() + 0.01*H*H) * knl.dw(p, &pos[*j])
    }).reduce(|a,b|a+b).unwrap_or(DVec2::ZERO);
    assert!(vis.is_finite());
    *a = acc_g + vis
  });
}

/// Update the densities at each particle
fn update_densities(pos: &[DVec2], den: &mut[f64], grid: &Grid, boundary: &Boundary, knl:&KernelType){
  pos.par_iter().enumerate().zip(den).for_each(|((i, x_i), rho_i)|{
    *rho_i = 
      M * grid.query_index(i).iter().map(|j| 
        knl.w(x_i, &pos[*j])
      ).sum::<f64>() + 
      M * boundary.grid.query_radius(x_i, &boundary.pos, KERNEL_SUPPORT).iter().map(|j| 
        knl.w(x_i, &boundary.pos[*j])
      ).sum::<f64>();
  });
}

/// Predict densities based on current positions and predicted velocities
fn predict_densities(den: &mut[f64], v_star: &[DVec2], pos: &[DVec2], grid: &Grid, boundary: &Boundary, dt:f64, knl:&KernelType){
  den.par_iter_mut().enumerate().for_each(|(i, rho_i)|{
    let fluid_neighbours = grid.query_index(i);
    let boundary_neighbours = boundary.grid.query_radius(&pos[i], &boundary.pos, KERNEL_SUPPORT);
    *rho_i = 
      // density from current fluid neighbours
      M *fluid_neighbours.iter().map(|j|
        knl.w(&pos[i], &pos[*j])
      ).sum::<f64>()
      // density from predicted fluid neighbours
      + dt * M * fluid_neighbours.iter().map(|j|
        (v_star[i] - v_star[*j]).dot(knl.dw(&pos[i], &pos[*j]))
      ).sum::<f64>()
      
      // density from current boundary neighbours
      + M * boundary_neighbours.iter().map(|j|
        knl.w(&pos[i], &boundary.pos[*j])
      ).sum::<f64>()
      // density from predicted boundary neighbours
      + dt * M * boundary_neighbours.iter().map(|j|
        (v_star[i]).dot(knl.dw(&pos[i], &boundary.pos[*j]))
      ).sum::<f64>();
  })
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
fn add_pressure_accelerations(pos: &[DVec2], den: &[f64], prs: &[f64], acc: &mut[DVec2], grid: &Grid, boundary: &Boundary, knl:&KernelType){
  let rho_0 = RHO_ZERO.load(Relaxed);
  let one_over_rho_0_squared = 1.0/(rho_0*rho_0);
  pos.par_iter().enumerate().zip(prs).zip(den).zip(acc)
  .for_each(|((((i, x_i), p_i), rho_i), acc)|{
    let p_i_over_rho_i_squared = p_i/(rho_i*rho_i);
    *acc += 
      -M * grid.query_index(i).iter().map(|j| 
        (p_i_over_rho_i_squared + prs[*j]/(den[*j]*den[*j])) * knl.dw(x_i, &pos[*j])
      ).sum::<DVec2>()
      -M * boundary.grid.query_radius(x_i, &boundary.pos, KERNEL_SUPPORT).iter().map(|j| 
        (p_i_over_rho_i_squared + p_i*one_over_rho_0_squared) * knl.dw(x_i, &boundary.pos[*j])
      ).sum::<DVec2>();
  })
}

/// Compute pressure accelerations from the momentum-preserving SPH approximation of the density gradient,
/// overwriting the current accelerations
fn overwrite_pressure_accelerations(pos: &[DVec2], den: &[f64], prs: &[f64], acc: &mut[DVec2], grid: &Grid, boundary: &Boundary, knl:&KernelType){
  let rho_0 = RHO_ZERO.load(Relaxed);
  let one_over_rho_0_squared = 1.0/rho_0*rho_0;
  pos.par_iter().enumerate().zip(prs).zip(den).zip(acc)
  .for_each(|((((i, x_i), p_i), rho_i), acc)|{
    let p_i_over_rho_i_squared = p_i/(rho_i*rho_i);
    *acc = 
      -M * grid.query_index(i).iter().map(|j| 
        (p_i_over_rho_i_squared + prs[*j]/(den[*j]*den[*j])) * knl.dw(x_i, &pos[*j])
      ).sum::<DVec2>()
      -M * boundary.grid.query_radius(x_i, &boundary.pos, KERNEL_SUPPORT).iter().map(|j| 
        (p_i_over_rho_i_squared + p_i*one_over_rho_0_squared) * knl.dw(x_i, &boundary.pos[*j])
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

/// Perform an explicit Euler numerical time integration step for a single quantity.
/// This simply means that x := x' * dt
fn time_step_explicit_euler_one_quantity(quantity: &mut[DVec2], derivative: &[DVec2], dt:f64){
  quantity.par_iter_mut().zip(derivative).for_each(|(v,a)|{
    *v += *a * dt;
  })
}

/// Harshly enforce rudimentary boundary conditions by simply setting velocity and acceleration of 
/// particles penetrating the boundary to zero while adjusting ther positions in a direction 
/// orthogonal to the boundary to be within bounds.
fn enforce_boundary_conditions(pos: &mut[DVec2], vel: &mut[DVec2], acc: &mut[DVec2]){
  pos.par_iter_mut().zip(vel).zip(acc).for_each(|((p, v), a)| {
    if p.x < HARD_BOUNDARY[0][0] as f64 {p.x = HARD_BOUNDARY[0][0] as f64; a.x=0.0; v.x=0.0;} 
    if p.y < HARD_BOUNDARY[0][1] as f64 {p.y = HARD_BOUNDARY[0][1] as f64; a.y=0.0; v.y=0.0;} 
    if p.x > HARD_BOUNDARY[1][0] as f64 {p.x = HARD_BOUNDARY[1][0] as f64; a.x=0.0; v.x=0.0;} 
    if p.y > HARD_BOUNDARY[1][1] as f64 {p.y = HARD_BOUNDARY[1][1] as f64; a.y=0.0; v.y=0.0;} 
  })
}


// STRUCTURE DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[atomic_enum]
#[derive(PartialEq)]
/// A fluid solver, implementing a single simulation step of some SPH method.
pub enum Solver{
  SESPH,
  SSESPH,
  ISESPH,
}

impl Solver{
  /// Perform a single simulation step using some SPH method.
  /// 
  /// The solver can rely on current particle positions and velocities to be accurate.
  /// Accelerations, densities and pressures etc. are not resorted and must be assumed to be
  /// uninitialized garbage. 
  /// 
  /// The grid can also be assumed to be accurate.
  fn solve(&self, state: &mut Attributes, grid: &Grid, current_t: &mut f64, boundary: &Boundary, knls:&SphKernel){
    match self {
      Solver::SESPH => sesph(state, grid, current_t, boundary, knls),
      Solver::SSESPH => ssesph(state, grid, current_t, boundary, knls),
      Solver::ISESPH => isesph(state, grid, current_t, boundary, knls),
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
    Self { pos, vel, acc, prs, den }
  }

  /// Resort all particle attributes according to some given order, which must be
  /// a permutation of (0..NUMBER_OF_PARTICLES). This is meant to eg. improve cache-hit-rates
  /// by employing the same sorting as the acceleration datastructure provides for neighbourhood queries.
  fn resort(&mut self, grid: &Grid){
    // extract the order of the attributes according to cell-wise z-ordering
    let order:Vec<usize> = grid.handles.par_iter().map(|h|h.index).collect();
    debug_assert!(order.len() == self.pos.len());
    // re-order relevant particle attributes in accordance with the order
    self.pos = order.par_iter().map(|i| self.pos[*i]).collect();
    self.vel = order.par_iter().map(|i| self.vel[*i]).collect();
  }
}

/// A struct representing a set of boundary particles and a respective data structure
/// for querying their positions which can be used to mirror pressure forces, creating
/// static boundaries for the simulation that use the SPH pressure solver to enforce
/// impenetrability of the boundary.
pub struct Boundary{
  pub pos:Vec<DVec2>,
  pub grid: Grid,
}

impl Boundary{
  /// Creates a new set of boundary particles in the pos Vec, with an accompanying
  /// grid to query for boundary neighbours. 
  /// The initialization creates layers of particles around the rectangle specified by 
  /// the static BOUNDARY. 
  /// 
  /// The internal grid is not meant to be updated, since the boundary is static.
  /// Boundary structs can and should therefore always be immutable.
  fn new(layers: usize)->Self{
    // initialize boundary particle positions
    let mut pos = vec![];
    for i in 0..layers{
      let mut x = BOUNDARY[0].x+H;
      while x <= BOUNDARY[1].x{
        pos.push(DVec2::new(x, BOUNDARY[0].y-i as f64*H));
        pos.push(DVec2::new(x, BOUNDARY[1].y+i as f64*H));
        x += H
      }
    }
    for i in 0..layers{
      let mut y = BOUNDARY[0].y-(layers-1) as f64*H;
      while y < BOUNDARY[1].y+(layers) as f64*H{
        pos.push(DVec2::new(BOUNDARY[0].x-i as f64*H, y));
        pos.push(DVec2::new(BOUNDARY[1].x+i as f64*H, y));
        y += H
      }
    }
    // create a grid with the boundary particles
    let mut grid = Grid::new(pos.len());
    grid.update_grid(&pos, KERNEL_SUPPORT);
    // immediately resort the positions vector for spatial locality
    let order:Vec<usize> = grid.handles.par_iter().map(|h|h.index).collect();
    pos = order.par_iter().map(|i| pos[*i]).collect();
    grid.update_grid(&pos, KERNEL_SUPPORT);

    // return the boundary struct
    Self{pos, grid}
  }
}
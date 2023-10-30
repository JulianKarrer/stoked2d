use crate::{*, sph::{kernel, kernel_derivative}, datastructure::Grid};
use std::time::Duration;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use spin_sleep::sleep;

// MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn run(){
  let mut state = Attributes::new();
  let mut grid = Grid::new(state.pos.len());
  let mut last_update_time = timestamp();
  let mut current_t = 0.0;
  let mut since_resort = 0;
  loop {
    // restart the simulation if requested
    if REQUEST_RESTART.fetch_and(false, SeqCst){
      state = Attributes::new();
      grid = Grid::new(state.pos.len());
      *(HISTORY.write()) = History::default();
      current_t = 0.0;
      last_update_time = timestamp();
    }
    // throttle
    sleep(Duration::from_micros(SIMULATION_THROTTLE_MICROS.load(Relaxed)));

    // update the datastructure and potentially resort particle attributes
    grid.update_grid(&state.pos, KERNEL_SUPPORT);
    if since_resort > RESORT_ATTRIBUTES_EVERY_N.load(Relaxed) {
      state.resort(&mut grid);
      since_resort = 0;
    } else {since_resort += 1;}

    // update densities and pressures
    update_density_pressure(&state.pos, &mut state.prs, &mut state.den, &grid);

    // apply external forces
    external_forces(&state.pos, &state.vel, &mut state.acc, &state.den, &grid);

    // apply pressure forces
    add_pressure_accelerations(&state.pos, &state.den, &state.prs, &mut state.acc, &grid);

    // enforce rudimentary boundary conditions
    enforce_boundary_conditions(&mut state.pos, &mut state.vel, &mut state.acc);

    // perform a time step
    let dt = update_dt(&state.vel, &mut current_t);
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &mut state.acc, dt);

    // write back the positions to the global buffer for visualization and update the FPS count
    update_fps(&mut last_update_time);
    {  HISTORY.write().add_step(&state, &grid, current_t); }
  }
}



// FUNCTIONS USED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Determines the size of the next time step in seconds using the maximum velocity of a particle
/// in the previous step, the particle spacing H and a factor LAMBDA, updating the current time and 
/// returning the dt for numerical time integration
/// 
/// This correpsponds to the Courant-Friedrichs-Lewy condition
fn update_dt(vel: &[DVec2], current_t: &mut f64)->f64{
  let v_max = vel.par_iter().map(|v|v.length()).reduce_with(|a,b| a.max(b)).unwrap();
  let mut dt = LAMBDA.load(Relaxed) * H / v_max;
  if !dt.is_normal() {dt = DEFAULT_DT}
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
fn external_forces(pos: &[DVec2], vel: &[DVec2], acc: &mut[DVec2], den: &[f64], grid: &Grid){
  // account for gravity
  let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
  // calculate viscosity
  let nu = NU.load(Relaxed);
  acc.par_iter_mut().zip(pos).zip(vel).for_each(|((a, p), v)|{
    let vis:DVec2 = nu * 2.0 * grid.query(p, KERNEL_SUPPORT).iter().map(|j|{
      let x_i_j = *p-pos[*j];
      let v_i_j = *v-vel[*j];
      M/den[*j] * (v_i_j).dot(x_i_j)/(x_i_j.length_squared() + 0.01*H*H) * kernel_derivative(p, &pos[*j])
    }).reduce(|a,b|a+b).unwrap_or(DVec2::ZERO);
    assert!(vis.is_finite());
    *a = acc_g + vis
  });
}

/// Update the densities and pressures at each particle
fn update_density_pressure(pos: &[DVec2], prs: &mut[f64], den: &mut[f64], grid: &Grid){
  let k = K.load(Relaxed);
  let rho_0 = RHO_ZERO.load(Relaxed);
  pos.par_iter().zip(prs).zip(den).for_each(|((x_i, p_i), d)|{
    let rho_i = grid.query(x_i, KERNEL_SUPPORT).par_iter().map(|x_j| M*kernel(x_i, &pos[*x_j])).sum::<f64>();
    *d = rho_i;
    *p_i = k*(rho_i / rho_0 - 1.0)
  });
}

/// Compute pressure accelerations from the momentum-preserving SPH approximation of the density gradient
fn add_pressure_accelerations(pos: &[DVec2], den: &[f64], prs: &[f64], acc: &mut[DVec2], grid: &Grid){
  pos.par_iter().zip(prs).zip(den).zip(acc).for_each(|(((x_i, p_i), rho_i), acc)|{
    *acc += - grid.query(x_i, KERNEL_SUPPORT).par_iter().map(|j| 
      M*(p_i/(rho_i*rho_i) + prs[*j]/(den[*j]*den[*j])) * kernel_derivative(x_i, &pos[*j])
    ).sum::<DVec2>();
  })
}

/// Perform a numerical time integration step using the symplectic Euler or Eurler-Cromer integration scheme.
fn time_step_euler_cromer(pos: &mut[DVec2], vel: &mut[DVec2], acc: &mut[DVec2], dt:f64){
  pos.par_iter_mut().zip(vel).zip(acc).for_each(|((p,v),a)|{
    *v += *a * dt;
    *p += *v * dt;
  })
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
    let den = vec![0.0;pos.len()];
    // return new set of attributes
    Self { pos, vel, acc, prs, den }
  }

  /// Resort all particle attributes according to some given order, which must be
  /// a permutation of (0..NUMBER_OF_PARTICLES). This is meant to eg. improve cache-hit-rates
  /// by employing the same sorting as the acceleration datastructure for neighbourhood queries.
  fn resort(&mut self, grid: &mut Grid){
    // extract the order of the attributes according to cell-wise z-ordering
    let order:Vec<usize> = grid.handles.par_iter().map(|h|h.index).collect();
    debug_assert!(order.len() == self.pos.len());
    // re-order all particle attributes in accordance with the order
    self.pos = order.par_iter().map(|i| self.pos[*i]).collect();
    self.vel = order.par_iter().map(|i| self.vel[*i]).collect();
    self.acc = order.par_iter().map(|i| self.acc[*i]).collect();
    self.prs = order.par_iter().map(|i| self.prs[*i]).collect();
    self.den = order.par_iter().map(|i| self.den[*i]).collect();

    // update the handles to point to the newly re-sorted attributes by simply setting them to
    // (0..NUMBER_OF_PARTICLES)
    grid.handles.par_iter_mut().enumerate().for_each(|(i,h)| h.index = i);
  }
}

use std::time::Duration;

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use spin_sleep::sleep;

use crate::{*, sph::{kernel, kernel_derivative}};


// INITIALIZATION
pub fn reset(pos: &mut Vec<DVec2>, vel: &mut Vec<DVec2>, acc: &mut Vec<DVec2>, pre: &mut Vec<f64>, den: &mut Vec<f64>){
  *(HISTORY.write()) = vec![(vec![], 0.0)];
  pos.clear();
  vel.clear();
  acc.clear();
  pre.clear();
  den.clear();
}

pub fn init(pos: &mut Vec<DVec2>, vel: &mut Vec<DVec2>, acc: &mut Vec<DVec2>, pre: &mut Vec<f64>, den: &mut Vec<f64>){
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
  *pre = vec![0.0;pos.len()];
  *den = vec![0.0;pos.len()];
}

// MAIN SIMULATION LOOP
pub fn run(){
  let mut pos:Vec<DVec2> = vec![]; 
  let mut vel:Vec<DVec2> = vec![]; 
  let mut acc:Vec<DVec2> = vec![]; 
  let mut prs:Vec<f64> = vec![];
  let mut den:Vec<f64> = vec![];
  init(&mut pos, &mut vel, &mut acc, &mut prs, &mut den);
  let mut last_update_time = timestamp();
  let mut current_t = 0.0;
  loop {
    // restart the simulation if requested
    if REQUEST_RESTART.fetch_and(false, SeqCst){
      reset(&mut pos, &mut vel, &mut acc, &mut prs, &mut den);
      init(&mut pos, &mut vel, &mut acc, &mut prs, &mut den);
    }
    // throttle
    sleep(Duration::from_micros(SIMULATION_THROTTLE_MICROS.load(Relaxed)));

    // apply external forces
    external_forces(&mut acc);

    // apply pressure forces
    update_density_pressure(&pos, &mut prs, &mut den);
    add_pressure_accelerations(&pos, &den, &prs, &mut acc);

    // enforce rudimentary boundary conditions
    enforce_boundary_conditions(&mut pos, &mut vel, &mut acc);

    // perform a time step
    let dt = micros_to_seconds(timestamp() - last_update_time);
    current_t += dt;
    last_update_time = timestamp();
    SIM_FPS.fetch_update(Relaxed, Relaxed, |fps| Some(fps*FPS_SMOOTING + 1.0/dt * (1.0-FPS_SMOOTING))).unwrap();
    time_step_euler_cromer(&mut pos, &mut vel, &mut acc, dt);

    // write back the positions to the global buffer for visualization and update timestamp
    {
      HISTORY.write().push((pos.clone(), current_t));
      // visualize the normalized speed of each particle
      let min = 0.8;
      let max = 1.2;
      *(COLOUR.write()) = den.par_iter().map(|x| 1.0- (x.min(max)-min)/(max-min)).collect();
    }
  }
}

/// Apply external forces such as gravity to the fluid
fn external_forces(acc: &mut[DVec2]){
  let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
  acc.par_iter_mut().for_each(|a|*a = acc_g);
}

/// Update the densities and pressures at each particle
fn update_density_pressure(pos: &[DVec2], prs: &mut[f64], den: &mut[f64]){
  let k = K.load(Relaxed);
  let rho_0 = RHO_ZERO.load(Relaxed);
  pos.par_iter().zip(prs).zip(den).for_each(|((x_i, p_i), d)|{
    let rho_i = pos.par_iter().map(|x_j| M*kernel(x_i, x_j)).sum::<f64>();
    *d = rho_i;
    *p_i = k*(rho_i / rho_0 - 1.0)
  });
}

fn add_pressure_accelerations(pos: &[DVec2], den: &[f64], prs: &[f64], acc: &mut[DVec2]){
  pos.par_iter().zip(prs).zip(den).zip(acc).for_each(|(((x_i, p_i), rho_i), acc)|{
    *acc += - pos.par_iter().enumerate().map(|(j, x_j)| 
      M*(p_i/(rho_i*rho_i) + prs[j]/(den[j]*den[j])) * kernel_derivative(x_i, x_j)
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

//
fn enforce_boundary_conditions(pos: &mut[DVec2], vel: &mut[DVec2], acc: &mut[DVec2]){
  pos.par_iter_mut().zip(vel).zip(acc).for_each(|((p, v), a)| {
    if p.x < BOUNDARY[0].x {p.x = BOUNDARY[0].x; a.x=0.0; v.x=0.0;} 
    if p.y < BOUNDARY[0].y {p.y = BOUNDARY[0].y; a.y=0.0; v.y=0.0;} 
    if p.x > BOUNDARY[1].x {p.x = BOUNDARY[1].x; a.x=0.0; v.x=0.0;} 
    if p.y > BOUNDARY[1].y {p.y = BOUNDARY[1].y; a.y=0.0; v.y=0.0;} 
  })
}
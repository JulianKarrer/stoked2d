use std::{thread::sleep, time::Duration};

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};

use crate::*;


// INITIALIZATION
pub fn reset(pos: &mut Vec<DVec2>, vel: &mut Vec<DVec2>, acc: &mut Vec<DVec2>){
  pos.clear();
  vel.clear();
  acc.clear();
}

pub fn init(pos: &mut Vec<DVec2>, vel: &mut Vec<DVec2>, acc: &mut Vec<DVec2>){
  for x in -5..=5{
    for y in 1..= 5{
      pos.push(DVec2::new(x as f64, y as f64));
      vel.push(DVec2::ZERO);
      acc.push(DVec2::ZERO);
    }
  }
}

// MAIN SIMULATION LOOP
pub fn run(){
  let mut pos:Vec<DVec2> = vec![]; 
  let mut vel:Vec<DVec2> = vec![]; 
  let mut acc:Vec<DVec2> = vec![]; 
  let mut last_update_time = timestamp();
  loop {
    // restart the simulation if requested
    if REQUEST_RESTART.load(Relaxed){
      reset(&mut pos, &mut vel, &mut acc);
      init(&mut pos, &mut vel, &mut acc);
      REQUEST_RESTART.store(false, Relaxed);
    }
    // throttle
    sleep(Duration::from_nanos(10));
    // update dt and fps
    let dt = timestamp() - last_update_time;
    SIM_FPS.store(SIM_FPS.load(Relaxed)*FPS_SMOOTING + 1.0/dt * (1.0-FPS_SMOOTING), Relaxed);

    // apply external forces
    external_forces(&mut acc);

    // enforce rudimentary boundary conditions
    enforce_boundary_conditions(&mut pos, &mut vel, &mut acc);

    // perform a time step
    time_step_euler_cromer(&mut pos, &mut vel, &mut acc, dt);

    // write back the positions to the global buffer for visualization and update timestamp
    {
      *(POSITIONS.write()) = pos.clone();
      // visualize the normalized speed of each particle
      let min = 0.0;
      let max = 20.0;
      *(COLOUR.write()) = vel.par_iter().map(|v| 1.0- (v.length().min(max)-min)/(max-min)).collect();
    }
    last_update_time = timestamp();
  }
}

/// Apply external forces such as gravity to the fluid
fn external_forces(acc: &mut[DVec2]){
  let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
  acc.par_iter_mut().for_each(|a|*a = acc_g);
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
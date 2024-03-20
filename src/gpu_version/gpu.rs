extern crate ocl;

use std::time::Duration;

use ocl::{prm::{Float, Float2, Uint2}, ProQue};
use rayon::iter::{IntoParallelRefIterator,  ParallelIterator};
use crate::{gpu_version::{buffers::GpuBuffers, kernels::Kernels}, gui::gui::SIMULATION_TOGGLE, simulation::update_fps, *};


pub fn run(){
  let n_est: usize = ((FLUID[1].x-FLUID[0].x)/(H) + 1.0).ceil() as usize * ((FLUID[1].y-FLUID[0].y)/(H) + 1.0).ceil() as usize;
  
  // buffer allocation and initialization
  let mut pos:Vec<Float2> = Vec::with_capacity(n_est); 
  let mut vel:Vec<Float2> = Vec::with_capacity(n_est); 
  let n = GpuBuffers::init_cpu_side(&mut pos, &mut vel);
  let pro_que: ProQue = ProQue::builder()
    .src(include_str!("kernels.cl"))
    .dims(n)
    .build().unwrap();
  println!("Number of particles: {}",n);
  let b: GpuBuffers = GpuBuffers::new(&pro_que, n);
  b.init_gpu_side(n, &pos, &vel);
  let mut den: Vec<Float> = vec![Float::new((M/(H*H)) as f32);pos.len()];
  let mut handles:Vec<Uint2> = (0..pos.len()).map(|i| Uint2::new(i as u32, i as u32)).collect();

  // define time step size
  let mut current_t = 0.0f32;
  let mut last_gui_update_t = 0.0f32;
  let mut dt = INITIAL_DT.load(Relaxed) as f32;

  // build relevant kernels
  let k = Kernels::new(&b, pro_que, pos.len() as u32, dt);
  
  // intitial history timestep for visualization
  let vmax = update_dt_gpu(&vel, &mut current_t, &mut dt);
  let handle_indices = handles.par_iter().map(|h|h[0]).collect::<Vec<u32>>();
  {  HISTORY.write().gpu_reset_and_add(&pos, &vel, &handle_indices, &den, current_t.into(), vmax); }

  // MAIN LOOP
  let mut last_update_time = timestamp();
  let mut since_resort = 0;
  while !REQUEST_RESTART.fetch_and(false, Relaxed) {
    // wait if requested
    while *SIMULATION_TOGGLE.read() { thread::sleep(Duration::from_millis(100)); }
    // update atomics to kernel programs
    k.update_atomics();

    // update grid
    if since_resort > {*RESORT_ATTRIBUTES_EVERY_N.read()} {
      unsafe { k.resort_pos_vel.enq().unwrap(); }
      unsafe { k.resort_pos_vel_b.enq().unwrap(); }
      since_resort = 0;
    } else {since_resort += 1}
    b.pos.read(&mut pos).enq().unwrap();
    let min: Float2 = pos.par_iter().cloned().reduce(
      || Float2::new(f32::MAX, f32::MAX), 
      |a,b|Float2::new(a[0].min(b[0]), a[1].min(b[1]))) 
      - Float2::new(2.0* KERNEL_SUPPORT as f32, 2.0*KERNEL_SUPPORT as f32);
    k.compute_neighbours.set_arg(0, &min).unwrap();
    k.compute_cell_keys.set_arg(0, &min).unwrap();
    unsafe { k.compute_cell_keys.enq().unwrap(); }
    k.sort_handles(&b);
    unsafe { k.compute_neighbours.enq().unwrap(); }

    
    // update densities and pressures
    unsafe { k.densitiy_pressure.enq().unwrap(); }
    // apply gravity and viscosity
    unsafe { k.gravity_viscosity.enq().unwrap(); }
    // add pressure accelerations
    unsafe { k.pressure_acceleration.enq().unwrap(); }

    // integrate accelerations to position updates
    b.vel.read(&mut vel).enq().unwrap();
    let vmax = update_dt_gpu(&vel, &mut current_t, &mut dt);
    k.euler_cromer.set_arg(3, dt).unwrap();
    unsafe { k.euler_cromer.enq().unwrap(); }
    unsafe { k.boundary.enq().unwrap(); }
    
    // write back the positions to the global buffer for visualization and update the FPS count
    update_fps(&mut last_update_time);
    if current_t - last_gui_update_t > 0.017 {
      b.den.read(&mut den).enq().unwrap();
      b.handles.read(&mut handles).enq().unwrap();
      last_gui_update_t = current_t;
      let handle_indices = handles.par_iter().map(|h|h[1]).collect::<Vec<u32>>();
      {  HISTORY.write().gpu_add_step(&pos, &vel, &handle_indices, &den, current_t.into(), vmax); }
    }
    
  }
}


/// Calculate the length of a Float2
pub fn len_float2(x:&Float2)->f64{
  let xc: f64 = x[0] as f64;
  let yc: f64 = x[1] as f64;
  ((xc*xc)+(yc*yc)).sqrt()
}

/// Update the current time step size in accordance with the CFL condition,
/// respecting the atomics for initial and maximum time step sizes and accumulate into the current total time `current_t`
fn update_dt_gpu(vel: &[Float2], current_t: &mut f32, dt: &mut f32)->f64{
  let v_max = vel.par_iter().map(len_float2).reduce_with(|a,b| a.max(b)).unwrap();
  let mut new_dt = ((LAMBDA.load(Relaxed) * H / v_max)
    .min(MAX_DT.load(Relaxed))) as f32;
  if v_max < VELOCITY_EPSILON || !new_dt.is_normal() {new_dt = INITIAL_DT.load(Relaxed) as f32}
  *current_t += new_dt;
  *dt = new_dt;
  v_max.max(VELOCITY_EPSILON)
}


#[cfg(test)]
mod tests {
  extern crate test;
  use super::*;
  use rand::Rng;
  use test::Bencher;
  

  #[test]
  fn gpu_sorting_radix_small(){
    for _ in 0..20{ 
      gpu_sorting_radix(rand::thread_rng().gen_range(2..262_144)) 
    }
  }

  #[test]
  fn gpu_sorting_radix_big(){
    for _ in 0..10{ 
      gpu_sorting_radix(rand::thread_rng().gen_range(262_144..1_000_000)) 
    }
  }

  fn rand_uint()->u32{rand::thread_rng().gen_range(u32::MIN..u32::MAX)}
  fn gpu_sorting_radix(n:usize){
    let pro_que: ProQue = ProQue::builder()
      .src(include_str!("kernels.cl")).dims(n).build().unwrap();
    let b: GpuBuffers = GpuBuffers::new(&pro_que, n);
    let mut handles:Vec<Uint2> = (0..n).map(|i| 
      Uint2::new(rand_uint(), i as u32)
    ).collect();
    b.handles.write(&handles).enq().unwrap();

    let k = Kernels::new(&b, pro_que, n as u32, 0.);
    k.sort_handles(&b);
    b.handles.read(&mut handles).enq().unwrap();
    handles.iter().enumerate().skip(1).for_each(|(i,h)| 
      assert!(h[0]>=handles[i-1][0])
    )
  }

  #[bench]
  fn gpu_radix_bench(b: &mut Bencher){
    let n = 30_000;
    let pro_que: ProQue = ProQue::builder()
      .src(include_str!("kernels.cl")).dims(n).build().unwrap();
    let buffers: GpuBuffers = GpuBuffers::new(&pro_que, n);
    let kernels = Kernels::new(&buffers, pro_que, n as u32, 0.);
    let mut handles:Vec<Uint2> = (0..n).map(|i| 
      Uint2::new(rand_uint(), i as u32)
    ).collect();

    b.iter(||{
      buffers.handles.write(&handles).enq().unwrap();
      kernels.sort_handles(&buffers);
    });
   
    buffers.handles.read(&mut handles).enq().unwrap();
  }

}

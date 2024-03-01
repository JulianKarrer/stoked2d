extern crate ocl;

use std::time::Duration;

use ocl::{prm::{Float, Float2, Uint2}, ProQue};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use voracious_radix_sort::{RadixSort, Radixable};
use crate::{gpu_version::{buffers::GpuBuffers, kernels::Kernels}, gui::SIMULATION_TOGGLE, simulation::update_fps, *};

#[derive(Copy, Clone, Debug, Default)]
struct GpuHandle(Uint2);
impl GpuHandle {
  fn new(cell: u32, key:u32)->Self{Self(Uint2::new(cell, key))}
  fn cell(&self)->u32{self.0[0]}
  fn set_cell(&mut self, cell:u32){self.0[0] = cell}
  fn index(&self)->u32{self.0[1]}
}
impl PartialOrd for GpuHandle{
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {self.cell().partial_cmp(&other.cell())}
}
impl PartialEq for GpuHandle{
  fn eq(&self, other: &Self) -> bool {self.cell() == other.cell() }
}
impl Radixable<u32> for GpuHandle{
  type Key = u32; fn key(&self) -> Self::Key {self.cell()}
}

pub fn run(){
  // upper bound on the number of particles before initialization
  let n: usize = ((FLUID[1].x-FLUID[0].x)/(H) + 1.0).ceil() as usize * ((FLUID[1].y-FLUID[0].y)/(H) + 1.0).ceil() as usize;

  let pro_que: ProQue = ProQue::builder()
    .src(include_str!("kernels.cl"))
    .dims(n)
    .build().unwrap();
  println!("Number of particles: {}",n);
  
  // buffer allocation and initialization
  let mut b: GpuBuffers = GpuBuffers::new(&pro_que, n);
  let mut pos:Vec<Float2> = Vec::with_capacity(n); 
  let mut vel:Vec<Float2> = Vec::with_capacity(n); 
  b.init_fluid_pos_vel(&mut pos, &mut vel, n);
  let mut den: Vec<Float> = vec![Float::new((M/(H*H)) as f32);pos.len()];
  let mut handles:Vec<GpuHandle> = (0..pos.len()).map(|i| GpuHandle::new(i as u32, 0)).collect();
  let mut handles_cpu:Vec<Uint2> = vec![Uint2::new(0,0); pos.len()];

  // define time step size
  let mut current_t = 0.0f32;
  let mut dt = INITIAL_DT.load(Relaxed) as f32;

  // build relevant kernels
  let k = Kernels::new(&b, pro_que, pos.len() as u32, dt);
  
  // MAIN LOOPddddsdfsdsADFASDFASDFASDF
  let mut last_update_time = timestamp();
  let mut since_resort = 0;
  {  HISTORY.write().gpu_reset_and_add(&pos, &handles.par_iter().map(|h|h.0[1]).collect::<Vec<u32>>(), &den, current_t.into()); }
  while !REQUEST_RESTART.fetch_and(false, Relaxed) {
    // wait if requested
    while *SIMULATION_TOGGLE.read() { thread::sleep(Duration::from_millis(100)); }
    // update atomics to kernel programs
    k.update_atomics();

    // copy pos to host for cpu side computations
      // potentially resort particle data to restore spatial locality
    b.load_and_resort_pos_vel(&mut since_resort, &mut pos, &mut vel);
    // update grid
    // println!("before {} {}", pos[0], pos[30000]);
    update_grid(&k, &b, &mut handles, &mut handles_cpu, &pos);
    // println!("after {} {}", pos[0], pos[30000]);
    
    // update densities and pressures
    unsafe { k.densitiy_pressure_kernel.enq().unwrap(); }
    // apply gravity and viscosity
    unsafe { k.gravity_viscosity_kernel.enq().unwrap(); }
    // add pressure accelerations
    unsafe { k.pressure_acceleration_kernel.enq().unwrap(); }

    // integrate accelerations to position updates
    b.vel.read(&mut vel).enq().unwrap();
    dt = update_dt_gpu(&vel, &mut current_t);
    k.euler_cromer_kernel.set_arg(3, dt).unwrap();
    unsafe { k.euler_cromer_kernel.enq().unwrap(); }
    unsafe { k.boundary_kernel.enq().unwrap(); }
    
    // write back the positions to the global buffer for visualization and update the FPS count
    update_fps(&mut last_update_time);
    b.den.read(&mut den).enq().unwrap();
    {  HISTORY.write().gpu_add_step(&pos, 
      &(handles.par_iter().map(|h|h.0[1]).collect::<Vec<u32>>()), 
      &den, current_t.into()); }
  }
}

/// Update all cell indices in the array of handles. Returns the minimum point from which the 
/// space filling XY-curve starts.
fn update_cell_keys(pos:&[Float2], handles:&mut [GpuHandle])->Float2{
  let min: Float2 = pos.par_iter().cloned().reduce(
    || Float2::new(f32::MAX, f32::MAX), 
    |a,b|Float2::new(a[0].min(b[0]), a[1].min(b[1]))) 
    - Float2::new(2.0* KERNEL_SUPPORT as f32, 2.0*KERNEL_SUPPORT as f32);
  handles.par_iter_mut().for_each(|c| c.set_cell(cell_key(&pos[c.index() as usize], &min)));
  min
}

/// Calculate cell index from position and minimum point of the domain extent
fn cell_key(pos:&Float2, min:&Float2) -> u32{ 
  let x: u16 = ((pos[0]-min[0])/KERNEL_SUPPORT as f32).floor() as u16;
  let y: u16 = ((pos[1]-min[1])/KERNEL_SUPPORT as f32).floor() as u16;
  ((y as u32) << 16) + (x as u32)
}

fn len_float2(x:&Float2)->f64{
  let xc = x[0] as f64;
  let yc = x[1] as f64;
  ((xc*xc)+(yc*yc)).sqrt()
}

/// Update the current time step size in accordance with the CFL condition,
/// respecting the atomics for initial and maximum time step sizes and accumulate into the current total time `current_t`
fn update_dt_gpu(vel: &[Float2], current_t: &mut f32)->f32{
  let v_max = vel.par_iter().map(len_float2).reduce_with(|a,b| a.max(b)).unwrap();
  let mut dt = ((LAMBDA.load(Relaxed) * H / v_max)
    .min(MAX_DT.load(Relaxed))) as f32;
  if v_max < VELOCITY_EPSILON || !dt.is_normal() {dt = INITIAL_DT.load(Relaxed) as f32}
  *current_t += dt;
  dt
}

/// Update the uniform grid.
/// - Find the point of minimum extent
/// - Compute cell keys for each Particle
/// - Sort the array of handles on the CPU
/// - transfer back the sorted cell- and index- lists to GPU
/// - compute neighbour sets from the sorted array on the GPU
fn update_grid(k: &Kernels, b:&GpuBuffers, handles:&mut [GpuHandle], handles_cpu: &mut Vec<Uint2>,pos: &[Float2]){
    // perform cpu side computations
  let min = update_cell_keys(pos, handles);
  k.compute_neighbours_kernel.set_arg(0, min).unwrap();

  handles.voracious_mt_sort(*THREADS);
    // transfer sorted handles back to gpu and create neighbour lists
  handles.par_iter().map(|h|h.0).collect_into_vec(handles_cpu);
  b.handles.write(&*handles_cpu).enq().unwrap();
    // compute neighbour sets on the GPU 
  unsafe { k.compute_neighbours_kernel.enq().unwrap(); }
}


#[cfg(test)]
mod tests {
  use super::*;
  extern crate test;

  #[test]
  fn gpu_cell_keys_xyz(){
    let min = Float2::new(0.,0.);
    let pos:Vec<Float2> = (0..5).zip(0..5).map(|(x,y)| Float2::new(x as f32, y as f32)).collect();
    for i in 1..pos.len(){
      assert!(cell_key(&pos[i], &min) > cell_key(&pos[i-1], &min))
    }
    // assert that changes in y dominate changes in x
    assert!(cell_key(&Float2::new(4.,3.), &min) < cell_key(&Float2::new(3.,4.), &min));
    assert!(cell_key(&Float2::new(8.,2.), &min) < cell_key(&Float2::new(2.,8.), &min));
    // assert that an increase in x by the grid size increments the cell key once
    assert!(
      cell_key(&Float2::new(KERNEL_SUPPORT as f32, KERNEL_SUPPORT as f32), &min)+1 ==
      cell_key(&Float2::new(2.*KERNEL_SUPPORT as f32, KERNEL_SUPPORT as f32), &min)
    );
    // assert that any two positions in the same cell have the same key 
    assert!(
      cell_key(&Float2::new(KERNEL_SUPPORT as f32, KERNEL_SUPPORT as f32), &min) ==
      cell_key(&Float2::new(1.9*KERNEL_SUPPORT as f32, 1.9*KERNEL_SUPPORT as f32), &min)
    );
    assert!(
      cell_key(&Float2::new(KERNEL_SUPPORT as f32, KERNEL_SUPPORT as f32), &min) !=
      cell_key(&Float2::new(2.1*KERNEL_SUPPORT as f32, 2.1*KERNEL_SUPPORT as f32), &min)
    );
  }
}

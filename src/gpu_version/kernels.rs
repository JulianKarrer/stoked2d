use ocl::{prm::Float2, Kernel, ProQue};

use crate::{sph::KERNEL_CUBIC_NORMALIZE, utils::ceil_div, BOUNDARY, GRAVITY, H, K, KERNEL_SUPPORT, M, NU, RHO_ZERO};
use std::sync::atomic::Ordering::Relaxed;
use super::buffers::GpuBuffers;


pub struct Kernels{
  pub euler_cromer:Kernel,
  pub boundary:Kernel,
  pub densitiy_pressure:Kernel,
  pub gravity_viscosity:Kernel,
  pub pressure_acceleration:Kernel,
  pub compute_neighbours:Kernel,
  pub compute_cell_keys:Kernel,
  pub resort_pos_vel:Kernel,
  pub resort_pos_vel_b:Kernel,
  radix_sort_histograms:Kernel,
  radix_sort_prefixsum:Kernel,
  radix_sort_reorder:Kernel,
}

impl Kernels{
  /// Initialize a new set of kernel functions, compiling all
  /// relevant programs and setting their parameters using constants,
  /// atomics and the corresponsing buffers in a `GpuBuffer`.
  pub fn new(b: &GpuBuffers, pro_que:ProQue, n:u32, dt:f32)->Self{
    let euler_cromer_kernel = pro_que.kernel_builder("eulercromerstep")
      .arg(&b.pos)
      .arg(&b.vel)
      .arg(&b.acc)
      .arg(dt)
      .build().unwrap();
    let boundary_kernel = pro_que.kernel_builder("enforce_boundary")
      .arg(&b.pos)
      .arg(&b.vel)
      .arg(&b.acc)
      .arg(Float2::new(BOUNDARY[0][0] as f32, BOUNDARY[0][1] as f32))
      .arg(Float2::new(BOUNDARY[1][0] as f32, BOUNDARY[1][1] as f32))
      .build().unwrap();
    let densitiy_pressure_kernel = pro_que.kernel_builder("update_densities_pressures")
      .arg(K.load(Relaxed) as f32)
      .arg(RHO_ZERO.load(Relaxed) as f32)
      .arg(&b.den)
      .arg(&b.prs)
      .arg(&b.pos)
      .arg(&b.handles)
      .arg(&b.neighbours)
      .arg(M as f32)
      .arg(KERNEL_CUBIC_NORMALIZE as f32)
      .arg(H as f32)
      .arg(n)
      .build().unwrap();
    let gravity_viscosity_kernel = pro_que.kernel_builder("apply_gravity_viscosity")
      .arg(NU.load(Relaxed) as f32)
      .arg(GRAVITY.load(Relaxed) as f32)
      .arg(&b.acc)
      .arg(&b.vel)
      .arg(&b.pos)
      .arg(&b.den)
      .arg(&b.handles)
      .arg(&b.neighbours)
      .arg(M as f32)
      .arg(KERNEL_CUBIC_NORMALIZE as f32)
      .arg(H as f32)
      .arg(n)
      .build().unwrap();
    let pressure_acceleration_kernel = pro_que.kernel_builder("add_pressure_acceleration")
      .arg(RHO_ZERO.load(Relaxed) as f32)
      .arg(&b.pos)
      .arg(&b.acc)
      .arg(&b.den)
      .arg(&b.prs)
      .arg(&b.handles)
      .arg(&b.neighbours)
      .arg(M as f32)
      .arg(KERNEL_CUBIC_NORMALIZE as f32)
      .arg(H as f32)
      .arg(n)
      .build().unwrap();
    let min_extend = Float2::new(
      (BOUNDARY[0][0] - KERNEL_SUPPORT) as f32, (BOUNDARY[0][1]  - KERNEL_SUPPORT) as f32, 
    );
    let compute_neighbours_kernel = pro_que.kernel_builder("compute_neighbours")
      .arg(&min_extend)
      .arg(&b.pos)
      .arg(&b.handles)
      .arg(&b.neighbours)
      .arg(KERNEL_SUPPORT as f32)
      .arg(n)
      .build().unwrap();
    let compute_cell_keys_kernel = pro_que.kernel_builder("compute_cell_keys")
      .arg(&min_extend)
      .arg(KERNEL_SUPPORT as f32)
      .arg(&b.pos)
      .arg(&b.handles)
      .build().unwrap();
    let resort_pos_vel = pro_que.kernel_builder("resort_data_a")
      .arg(&b.handles)
      .arg(&b.pos)
      .arg(&b.vel)
      .arg(&b.pos_resort)
      .arg(&b.vel_resort)
      .arg(n)
      .build().unwrap();
    let resort_pos_vel_b = pro_que.kernel_builder("resort_data_b")
      .arg(&b.handles)
      .arg(&b.pos)
      .arg(&b.vel)
      .arg(&b.pos_resort)
      .arg(&b.vel_resort)
      .arg(n)
      .build().unwrap();

    let workgroup_size = 256;
    let n_256 = ceil_div(n as usize, workgroup_size);
    let radix_sort_histograms = pro_que.kernel_builder("radix_sort_histograms")
      .arg(&b.handles)
      .arg(&b.handles_temp)
      .arg(&b.histograms)
      .arg(&b.counts)
      .arg_local::<u32>(workgroup_size)
      .arg(0u32)
      .arg(n)
      .local_work_size(workgroup_size)
      .global_work_size(n_256) 
      .build().unwrap();
    let radix_sort_prefixsum = pro_que.kernel_builder("radix_sort_prefixsum")
      .arg(&b.histograms)
      .arg(&b.counts)
      .local_work_size(workgroup_size)
      .global_work_size(n_256) 
      .build().unwrap();
    let radix_sort_reorder = pro_que.kernel_builder("radix_sort_reorder")
      .arg(&b.handles)
      .arg(&b.handles_temp)
      .arg(&b.histograms)
      .arg(&b.counts)
      .arg_local::<u32>(workgroup_size)
      .arg(0u32)
      .arg(n)
      .local_work_size(workgroup_size)
      .global_work_size(n_256) 
      .build().unwrap();
    
    Self { 
      euler_cromer: euler_cromer_kernel, 
      boundary: boundary_kernel, 
      densitiy_pressure: densitiy_pressure_kernel, 
      gravity_viscosity: gravity_viscosity_kernel, 
      pressure_acceleration: pressure_acceleration_kernel, 
      compute_neighbours: compute_neighbours_kernel,
      compute_cell_keys: compute_cell_keys_kernel, 
      resort_pos_vel,
      resort_pos_vel_b,
      radix_sort_histograms,
      radix_sort_prefixsum,
      radix_sort_reorder,
    }
  }

  /// Update kernel arguments that are adjusted in the gui
  /// thorugh the use of atomics.
  pub fn update_atomics(&self){
    self.densitiy_pressure.set_arg(0, K.load(Relaxed) as f32).unwrap();
    self.densitiy_pressure.set_arg(1, RHO_ZERO.load(Relaxed) as f32).unwrap();
    self.gravity_viscosity.set_arg(0, NU.load(Relaxed) as f32).unwrap();
    self.gravity_viscosity.set_arg(1, GRAVITY.load(Relaxed) as f32).unwrap();
    self.pressure_acceleration.set_arg(0, RHO_ZERO.load(Relaxed) as f32).unwrap();
  }

  pub fn sort_handles(&self, b: &GpuBuffers){
    for (i, shift) in (0u32..32).step_by(8).enumerate(){
      // set bitshift argument
      self.radix_sort_histograms.set_arg(5, shift).unwrap();
      self.radix_sort_reorder.set_arg(5, shift).unwrap();
      // swap input and output buffers
      self.radix_sort_histograms.set_arg(0, 
        if i%2==0 {&b.handles} else {&b.handles_temp}
      ).unwrap();
      self.radix_sort_histograms.set_arg(1, 
        if i%2==0 {&b.handles_temp} else {&b.handles}
      ).unwrap();
      self.radix_sort_reorder.set_arg(0, 
        if i%2==0 {&b.handles} else {&b.handles_temp}
      ).unwrap();
      self.radix_sort_reorder.set_arg(1, 
        if i%2==0 {&b.handles_temp} else {&b.handles}
      ).unwrap();
      // reset histograms
      b.counts.write(&b.counts_zeros).enq().unwrap();
      b.histograms.write(&b.hist_zeros).enq().unwrap();
      //enqueue kernels
      unsafe {self.radix_sort_histograms.enq().unwrap();}
      unsafe {self.radix_sort_prefixsum.enq().unwrap();}
      unsafe {self.radix_sort_reorder.enq().unwrap();}
    }
  }

}

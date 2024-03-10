use ocl::{prm::{Float2, Uint2}, Kernel, ProQue};

use crate::{sph::KERNEL_CUBIC_NORMALIZE, BOUNDARY, GRAVITY, H, K, KERNEL_SUPPORT, M, NU, RHO_ZERO};
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
  pub sort_handles:Kernel,
  pub sort_handles_b:Kernel,
  pub resort_pos_vel:Kernel,
  pub resort_pos_vel_b:Kernel,
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
    let workgroup_size = 32;
    let sort_kernel = pro_que.kernel_builder("sort_handles_simple")
        .arg(&b.handles)
        .arg(&b.handles_sorted)
        .arg(n)
        .arg_local::<Uint2>(workgroup_size)
        .local_work_size(workgroup_size)
        .build().unwrap();
    let copy_handles = pro_que.kernel_builder("copy_handles")
      .arg(&b.handles_sorted)
      .arg(&b.handles)
      .arg(n)
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
    Self { 
      euler_cromer: euler_cromer_kernel, 
      boundary: boundary_kernel, 
      densitiy_pressure: densitiy_pressure_kernel, 
      gravity_viscosity: gravity_viscosity_kernel, 
      pressure_acceleration: pressure_acceleration_kernel, 
      compute_neighbours: compute_neighbours_kernel,
      compute_cell_keys: compute_cell_keys_kernel, 
      sort_handles: sort_kernel,
      sort_handles_b: copy_handles,
      resort_pos_vel,
      resort_pos_vel_b,
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

}

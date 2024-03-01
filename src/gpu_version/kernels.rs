use ocl::{prm::Float2, Kernel, ProQue};

use crate::{sph::KERNEL_CUBIC_NORMALIZE, BOUNDARY, GRAVITY, H, K, KERNEL_SUPPORT, M, NU, RHO_ZERO};
use std::sync::atomic::Ordering::Relaxed;
use super::buffers::GpuBuffers;


pub struct Kernels{
  pub euler_cromer_kernel:Kernel,
  pub boundary_kernel:Kernel,
  pub densitiy_pressure_kernel:Kernel,
  pub gravity_viscosity_kernel:Kernel,
  pub pressure_acceleration_kernel:Kernel,
  pub compute_neighbours_kernel:Kernel,
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
  let compute_neighbours_kernel = pro_que.kernel_builder("compute_neighbours")
    .arg(Float2::new(0.0,0.0))
    .arg(&b.pos)
    .arg(&b.handles)
    .arg(&b.neighbours)
    .arg(KERNEL_SUPPORT as f32)
    .arg(n)
    .build().unwrap();

    Self { 
      euler_cromer_kernel, 
      boundary_kernel, 
      densitiy_pressure_kernel, 
      gravity_viscosity_kernel, 
      pressure_acceleration_kernel, 
      compute_neighbours_kernel 
    }
  }

  /// Update kernel arguments that are adjusted in the gui
  /// thorugh the use of atomics.
  pub fn update_atomics(&self){
    self.densitiy_pressure_kernel.set_arg(0, K.load(Relaxed) as f32).unwrap();
    self.densitiy_pressure_kernel.set_arg(1, RHO_ZERO.load(Relaxed) as f32).unwrap();
    self.gravity_viscosity_kernel.set_arg(0, NU.load(Relaxed) as f32).unwrap();
    self.gravity_viscosity_kernel.set_arg(1, GRAVITY.load(Relaxed) as f32).unwrap();
    self.pressure_acceleration_kernel.set_arg(0, RHO_ZERO.load(Relaxed) as f32).unwrap();
  }

}

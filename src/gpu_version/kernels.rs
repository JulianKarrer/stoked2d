use ocl::{prm::Float2, Kernel, ProQue};

use crate::{sph::KERNEL_CUBIC_NORMALIZE, utils::next_multiple, BOUNDARY, GRAVITY, H, INITIAL_DT, K, KERNEL_SUPPORT, LAMBDA, M, MAX_DT, NU, RHO_ZERO, VELOCITY_EPSILON, WARP, WORKGROUP_SIZE};
use std::sync::atomic::Ordering::Relaxed;
use super::buffers::GpuBuffers;

pub struct KernelGroup{
  pub group: Vec<Kernel>
}
impl KernelGroup{
  pub unsafe fn enq(&self)->ocl::Result<()>{
    for kernel in &self.group{unsafe { kernel.enq()? }}
    Ok(())
  }
  fn new(vec:Vec<Kernel>)->Self{
    Self { group: vec }
  }
}

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
  radix_sort:KernelGroup,
  reduce_min_pos:Kernel,
  reduce_min_vel_compute_dt:KernelGroup,
}

impl Kernels{
  /// Initialize a new set of kernel functions, compiling all
  /// relevant programs and setting their parameters using constants,
  /// atomics and the corresponsing buffers in a `GpuBuffer`.
  pub fn new(b: &GpuBuffers, pro_que:ProQue, n:u32)->Self{
    let euler_cromer_kernel = pro_que.kernel_builder("eulercromerstep")
      .arg(&b.pos)
      .arg(&b.vel)
      .arg(&b.acc)
      .arg(&b.dt)
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
      .arg(&b.pos_min)
      .arg(&b.pos)
      .arg(&b.handles)
      .arg(&b.neighbours)
      .arg(KERNEL_SUPPORT as f32)
      .arg(n)
      .build().unwrap();
    let compute_cell_keys_kernel = pro_que.kernel_builder("compute_cell_keys")
      .arg(&b.pos_min)
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

    let n_256 = next_multiple(n as usize, WORKGROUP_SIZE);
    let n_groups = n_256/256;
    let splinters = 1 + ((n_groups - 1) / WARP);
  
    let radix_a = pro_que.kernel_builder("radix_sort_histograms")
      .arg(&b.handles)
      .arg(&b.handles_temp)
      .arg(&b.histograms)
      .arg(&b.counts)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(0u32)
      .arg(n as u32)
      .local_work_size(WORKGROUP_SIZE)
      .global_work_size(n_256) 
      .build().unwrap();
    let n_groups = n_256/256;
    let next_power_of_two = (2u32).pow((n_groups as f64).log2().ceil() as u32) as usize;
    let radix_b_small = pro_que.kernel_builder("radix_sort_prefixsum_small")
      .arg(&b.histograms)
      .arg_local::<u32>(next_power_of_two)
      .arg(&b.counts)
      .arg(n_groups as u32)
      .local_work_size(next_power_of_two)
      .global_work_size(next_power_of_two*256) 
      .build().unwrap();
    let radix_b = pro_que.kernel_builder("radix_sort_prefixsum_a")
      .arg(&b.histograms)
      .arg_local::<u32>(WARP)
      .arg(&b.counts)
      .arg(&b.counts_b)
      .arg(n_groups as u32)
      .arg(splinters as u32)
      .local_work_size(WARP)
      .global_work_size((WARP*splinters)*256) 
      .build().unwrap();
    let radix_c = pro_que.kernel_builder("radix_sort_prefixsum_b")
      .arg(&b.histograms)
      .arg(&b.counts_b)
      .arg(n_groups as u32)
      .arg(splinters as u32)
      .global_work_size(256) 
      .build().unwrap();
    let radix_d = pro_que.kernel_builder("radix_sort_prefixsum_c")
      .arg(&b.histograms)
      .arg(&b.counts_b)
      .arg(n_groups as u32)
      .arg(splinters as u32)
      .local_work_size(WARP)
      .global_work_size((WARP*splinters)*256) 
      .build().unwrap();
    let radix_e = pro_que.kernel_builder("radix_sort_reorder")
      .arg(&b.handles)
      .arg(&b.handles_temp)
      .arg(&b.histograms)
      .arg(&b.counts)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(0u32)
      .arg(n as u32)
      .local_work_size(WORKGROUP_SIZE)
      .global_work_size(n_256) 
      .build().unwrap();
    let reduce_min_pos = pro_que.kernel_builder("reduce_min")
      .arg(&b.pos)
      .arg(&b.pos_min)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(n as u32)
      .arg(1u32)
      .local_work_size(WORKGROUP_SIZE)
      .build().unwrap();
    let reduce_max_vel = pro_que.kernel_builder("reduce_max_magnitude")
      .arg(&b.vel)
      .arg(&b.vel_max)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(n as u32)
      .arg(1u32)
      .local_work_size(WORKGROUP_SIZE)
      .build().unwrap();
    let update_dt = pro_que.kernel_builder("update_dt")
      .arg(&b.dt)
      .arg(&b.current_t)
      .arg(&b.vel_max)
      .arg(LAMBDA.load(Relaxed) as f32)
      .arg(MAX_DT.load(Relaxed) as f32)
      .arg(INITIAL_DT.load(Relaxed) as f32)
      .arg(VELOCITY_EPSILON as f32)
      .arg(H as f32)
      .global_work_size(1)
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
      radix_sort: if n<262_144 {
        KernelGroup::new(vec![radix_a, radix_b_small, radix_e])

      } else {
        KernelGroup::new(vec![radix_a, radix_b, radix_c, radix_d, radix_e])
      },
      reduce_min_pos: reduce_min_pos,
      reduce_min_vel_compute_dt: KernelGroup::new(vec![reduce_max_vel, update_dt]),
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
    self.reduce_min_vel_compute_dt.group[1].set_arg(3, LAMBDA.load(Relaxed) as f32).unwrap();
    self.reduce_min_vel_compute_dt.group[1].set_arg(4, MAX_DT.load(Relaxed) as f32).unwrap();
    self.reduce_min_vel_compute_dt.group[1].set_arg(5, INITIAL_DT.load(Relaxed) as f32).unwrap();
  }

  pub fn sort_handles(&self, b: &GpuBuffers){
    for (i, shift) in (0u32..32).step_by(8).enumerate(){
      // set bitshift argument
      self.radix_sort.group.first().unwrap().set_arg(5, shift).unwrap();
      self.radix_sort.group.last().unwrap().set_arg(5, shift).unwrap();
      // swap input and output buffers
      self.radix_sort.group.first().unwrap().set_arg(0, 
        if i%2==0 {&b.handles} else {&b.handles_temp}
      ).unwrap();
      self.radix_sort.group.first().unwrap().set_arg(1, 
        if i%2==0 {&b.handles_temp} else {&b.handles}
      ).unwrap();
      self.radix_sort.group.last().unwrap().set_arg(0, 
        if i%2==0 {&b.handles} else {&b.handles_temp}
      ).unwrap();
      self.radix_sort.group.last().unwrap().set_arg(1, 
        if i%2==0 {&b.handles_temp} else {&b.handles}
      ).unwrap();
      // reset histograms
      b.counts.write(&b.counts_zeros).enq().unwrap();
      b.counts_b.write(&b.counts_b_zeros).enq().unwrap();
      b.histograms.write(&b.hist_zeros).enq().unwrap();
      //enqueue kernels
      unsafe { self.radix_sort.enq().unwrap() }
    }
  }

  /// Calculate the minimum position in `b.pos`, saving the result to `b.pos_min[0]`
  pub fn reduce_pos_min(&self, b: &GpuBuffers){
    let mut cur_size = b.n;
    self.reduce_min_pos.set_arg(4, 1u32).unwrap();
    while cur_size > 1 {
      self.reduce_min_pos.set_arg(3, cur_size as u32).unwrap();
      unsafe { self.reduce_min_pos.cmd()
        .global_work_size(next_multiple(cur_size, WORKGROUP_SIZE))
        .enq().unwrap() }
        self.reduce_min_pos.set_arg(4, 0u32).unwrap();
      cur_size = next_multiple(cur_size, WORKGROUP_SIZE) / WORKGROUP_SIZE;
    }
  }

  /// Calculate the updated timestep size according to the CFL condition
  /// by reducing `b.vel` to find the magnitude of the maximum velocity.
  pub fn update_dt_reduce_vel(&self, b: &GpuBuffers){
    // reduce the array of velocities to find the maximum magnitude
    let mut cur_size = b.n;
    self.reduce_min_vel_compute_dt.group[0].set_arg(4, 1u32).unwrap();
    while cur_size > 1 {
      self.reduce_min_vel_compute_dt.group[0].set_arg(3, cur_size as u32).unwrap();
      unsafe { self.reduce_min_vel_compute_dt.group[0].cmd()
        .global_work_size(next_multiple(cur_size, WORKGROUP_SIZE))
        .enq().unwrap() }
        self.reduce_min_vel_compute_dt.group[0].set_arg(4, 0u32).unwrap();
      cur_size = next_multiple(cur_size, WORKGROUP_SIZE) / WORKGROUP_SIZE;
    }
    // compute updated dt
    unsafe { self.reduce_min_vel_compute_dt.group[1].enq().unwrap() }
  }
}

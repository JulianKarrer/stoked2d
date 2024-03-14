extern crate ocl;

use std::time::Duration;

use ocl::{prm::{Float, Float2, Uint2}, ProQue};
use rayon::iter::{IntoParallelRefIterator,  ParallelIterator};
use crate::{gpu_version::{buffers::GpuBuffers, kernels::Kernels}, gui::SIMULATION_TOGGLE, simulation::update_fps, *};


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
  
  // MAIN LOOP
  let mut last_update_time = timestamp();
  let mut since_resort = 0;
  {  HISTORY.write().gpu_reset_and_add(&pos, &handles.par_iter().map(|h|h[0]).collect::<Vec<u32>>(), &den, current_t.into()); }
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
    dt = update_dt_gpu(&vel, &mut current_t);
    k.euler_cromer.set_arg(3, dt).unwrap();
    unsafe { k.euler_cromer.enq().unwrap(); }
    unsafe { k.boundary.enq().unwrap(); }
    
    // write back the positions to the global buffer for visualization and update the FPS count
    update_fps(&mut last_update_time);
    if current_t - last_gui_update_t > 0.017 {
      b.den.read(&mut den).enq().unwrap();
      b.handles.read(&mut handles).enq().unwrap();
        last_gui_update_t = current_t;
      {  HISTORY.write().gpu_add_step(&pos, &handles.par_iter().map(|h|h[1]).collect::<Vec<u32>>(), &den, current_t.into()); }
    }
    
  }
}


/// Calculate the length of a Float2
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


#[cfg(test)]
mod tests {
  use ocl::{Buffer, MemFlags};
  use rand::Rng;
  use test::Bencher;
  use crate::utils::ceil_div;

use super::*;
  extern crate test;
  fn rand_uint()->u32{rand::thread_rng().gen_range(u32::MIN..u32::MAX)}

  #[test]
  fn repeated_gpu_sorting_radix(){
    for _ in 0..100{
      gpu_sorting_radix();
    }
  }

  #[test]
  fn gpu_sorting_radix(){
    let src = include_str!("kernels.cl");
  
    const WORKGROUP_SIZE: usize = 256; 
    let n: usize = rand::thread_rng().gen_range(2..5_000_000); 
    println!("{}",n);
    let n_256 = ceil_div(n, 256);

    let n_groups = n_256/256;
    let warp = 256;
    let splinters = 1 + ((n_groups - 1) / warp);
  
    let initial_state = (0..n_256)
      // .map(|i| Uint2::new((n-i-1) as u32, i as u32))
      .map(|i| Uint2::new(rand_uint(), i as u32))
      .collect::<Vec<Uint2>>();
  
    let pro_que = ProQue::builder()
      .src(src)
      .dims(n_256)
      .build().unwrap();


    let input = pro_que.create_buffer::<Uint2>().unwrap();
    let output = pro_que.create_buffer::<Uint2>().unwrap();
    let histo_zeros  =  vec![0u32; n_256];
    let histograms = Buffer::builder()
      .queue(pro_que.queue().clone())
      .flags(MemFlags::new().read_write())
      .len(n_256)
      .fill_val(0u32)
      .build().unwrap();
    let count_zeros = vec![0u32; 256];
    let counts = Buffer::builder()
      .queue(pro_que.queue().clone())
      .flags(MemFlags::new().read_write())
      .len(256)
      .fill_val(0u32)
      .build().unwrap();
    let count_b_zeros = vec![0u32; splinters*256];
    let counts_b = Buffer::builder()
      .queue(pro_que.queue().clone())
      .flags(MemFlags::new().read_write())
      .len(splinters*256)
      .fill_val(0u32)
      .build().unwrap();
  
    let radix_a = pro_que.kernel_builder("radix_sort_histograms")
      .arg(&input)
      .arg(&output)
      .arg(&histograms)
      .arg(&counts)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(0u32)
      .arg(n as u32)
      .local_work_size(WORKGROUP_SIZE)
      .global_work_size(n_256) 
      .build().unwrap();
    let radix_b = pro_que.kernel_builder("radix_sort_prefixsum_a")
      .arg(&histograms)
      .arg_local::<u32>(warp)
      .arg(&counts)
      .arg(&counts_b)
      .arg(n_groups as u32)
      .local_work_size(warp)
      .global_work_size((warp*splinters)*256) 
      .build().unwrap();
    let radix_c = pro_que.kernel_builder("radix_sort_prefixsum_b")
      .arg(&histograms)
      .arg_local::<u32>(warp)
      .arg(&counts)
      .arg(&counts_b)
      .arg(n_groups as u32)
      .global_work_size(256) 
      .build().unwrap();
    let radix_d = pro_que.kernel_builder("radix_sort_prefixsum_c")
      .arg(&histograms)
      .arg_local::<u32>(warp)
      .arg(&counts)
      .arg(&counts_b)
      .arg(n_groups as u32)
      .local_work_size(warp)
      .global_work_size((warp*splinters)*256) 
      .build().unwrap();
    let radix_e = pro_que.kernel_builder("radix_sort_reorder")
      .arg(&input)
      .arg(&output)
      .arg(&histograms)
      .arg(&counts)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(0u32)
      .arg(n as u32)
      .local_work_size(WORKGROUP_SIZE)
      .global_work_size(n_256) 
      .build().unwrap();

    input.write(&initial_state).enq().unwrap();
    output.write(&initial_state).enq().unwrap();
  
    // execute the kernel passes
    for (i, shift) in (0u32..32).step_by(8).enumerate(){
      // set bitshift argument
      radix_a.set_arg(5, shift).unwrap();
      radix_e.set_arg(5, shift).unwrap();
      // swap input and output buffers
      radix_a.set_arg(0, if i%2==0 {&input} else {&output}).unwrap();
      radix_a.set_arg(1, if i%2==0 {&output} else {&input}).unwrap();
      radix_e.set_arg(0, if i%2==0 {&input} else {&output}).unwrap();
      radix_e.set_arg(1, if i%2==0 {&output} else {&input}).unwrap();
      // reset histograms
      counts_b.write(&count_b_zeros).enq().unwrap();
      counts.write(&count_zeros).enq().unwrap();
      histograms.write(&histo_zeros).enq().unwrap();
      //enqueue kernels
      unsafe {radix_a.enq().unwrap();}
      unsafe {radix_b.enq().unwrap();}
      unsafe {radix_c.enq().unwrap();}
      unsafe {radix_d.enq().unwrap();}
      unsafe {radix_e.enq().unwrap();}
    }
   
    // show the result
    let mut res_in = initial_state.clone();
    let mut res_out = initial_state.clone();
    let mut res_hist = vec![0;n_256];
    let mut res_counts = vec![0;256];
    input.read(&mut res_in).enq().unwrap();
    output.read(&mut res_out).enq().unwrap();
    histograms.read(&mut res_hist).enq().unwrap();
    counts.read(&mut res_counts).enq().unwrap();
  
    let assert_all_true:Vec<bool> = (1..n).map(|i|res_in[i][0]>= res_in[i-1][0]).collect();
    assert!(assert_all_true.iter().all(|x|*x),
    "in\n{:?}\n\nout\n{:?}\n\nhist\n{:?}\n\ncounts\n{:?}\n\nn:{}", 
    res_in, res_out, res_hist,res_counts,n
    );
  }

  #[bench]
  fn gpu_radix_bench(b: &mut Bencher){
    let src = include_str!("kernels.cl");
  
    const WORKGROUP_SIZE: usize = 256; 
    let n: usize = 1_000_000;//rand::thread_rng().gen_range(2..5_000_000); 
    let n_256 = ceil_div(n, 256);

    let n_groups = n_256/256;
    let warp = 256;
    let splinters = 1 + ((n_groups - 1) / warp);
    
    let initial_state = (0..n_256)
      // .map(|i| Uint2::new((n-i-1) as u32, i as u32))
      .map(|i| Uint2::new(rand_uint(), i as u32))
      .collect::<Vec<Uint2>>();
  
    let pro_que = ProQue::builder()
      .src(src)
      .dims(n_256)
      .build().unwrap();

    let input = pro_que.create_buffer::<Uint2>().unwrap();
    let output = pro_que.create_buffer::<Uint2>().unwrap();
    let histo_zeros  =  vec![0u32; n_256];
    let histograms = Buffer::builder()
      .queue(pro_que.queue().clone())
      .flags(MemFlags::new().read_write())
      .len(n_256)
      .fill_val(0u32)
      .build().unwrap();
    let count_zeros = vec![0u32; 256];
    let counts = Buffer::builder()
      .queue(pro_que.queue().clone())
      .flags(MemFlags::new().read_write())
      .len(256)
      .fill_val(0u32)
      .build().unwrap();
    let counts_b = Buffer::builder()
      .queue(pro_que.queue().clone())
      .flags(MemFlags::new().read_write())
      .len(splinters*256)
      .fill_val(0u32)
      .build().unwrap();
  
    let radix_a = pro_que.kernel_builder("radix_sort_histograms")
      .arg(&input)
      .arg(&output)
      .arg(&histograms)
      .arg(&counts)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(0u32)
      .arg(n as u32)
      .local_work_size(WORKGROUP_SIZE)
      .global_work_size(n_256) 
      .build().unwrap();
    let radix_b_a = pro_que.kernel_builder("radix_sort_prefixsum_a")
      .arg(&histograms)
      .arg_local::<u32>(warp)
      .arg(&counts)
      .arg(&counts_b)
      .arg(n_groups as u32)
      .local_work_size(warp)
      .global_work_size((warp*splinters)*256) 
      .build().unwrap();
    let radix_b_b = pro_que.kernel_builder("radix_sort_prefixsum_b")
      .arg(&histograms)
      .arg_local::<u32>(warp)
      .arg(&counts)
      .arg(&counts_b)
      .arg(n_groups as u32)
      .global_work_size(256) 
      .build().unwrap();
    let radix_b_c = pro_que.kernel_builder("radix_sort_prefixsum_c")
      .arg(&histograms)
      .arg_local::<u32>(warp)
      .arg(&counts)
      .arg(&counts_b)
      .arg(n_groups as u32)
      .local_work_size(warp)
      .global_work_size((warp*splinters)*256) 
      .build().unwrap();
    let radix_c = pro_que.kernel_builder("radix_sort_reorder")
      .arg(&input)
      .arg(&output)
      .arg(&histograms)
      .arg(&counts)
      .arg_local::<u32>(WORKGROUP_SIZE)
      .arg(0u32)
      .arg(n as u32)
      .local_work_size(WORKGROUP_SIZE)
      .global_work_size(n_256) 
      .build().unwrap();

    b.iter(|| {
      input.write(&initial_state).enq().unwrap();
      output.write(&initial_state).enq().unwrap();
    
      // execute the kernel passes
      for (i, shift) in (0u32..32).step_by(8).enumerate(){
        // set bitshift argument
        radix_a.set_arg(5, shift).unwrap();
        radix_c.set_arg(5, shift).unwrap();
        // swap input and output buffers
        radix_a.set_arg(0, if i%2==0 {&input} else {&output}).unwrap();
        radix_a.set_arg(1, if i%2==0 {&output} else {&input}).unwrap();
        radix_c.set_arg(0, if i%2==0 {&input} else {&output}).unwrap();
        radix_c.set_arg(1, if i%2==0 {&output} else {&input}).unwrap();
        // reset histograms
        counts.write(&count_zeros).enq().unwrap();
        histograms.write(&histo_zeros).enq().unwrap();
        //enqueue kernels
        unsafe {radix_a.enq().unwrap();}
        // unsafe {radix_b_counts.enq().unwrap();}
        unsafe {radix_b_a.enq().unwrap();}
        unsafe {radix_b_b.enq().unwrap();}
        unsafe {radix_b_c.enq().unwrap();}
        unsafe {radix_c.enq().unwrap();}
      }
    });
   
    // show the result
    let mut res_in = initial_state.clone();
    let mut res_out = initial_state.clone();
    let mut res_hist = vec![0;n_256];
    let mut res_counts = vec![0;256];
    input.read(&mut res_in).enq().unwrap();
    output.read(&mut res_out).enq().unwrap();
    histograms.read(&mut res_hist).enq().unwrap();
    counts.read(&mut res_counts).enq().unwrap();
  }



  // #[bench]
  // fn old_gpu_radix_bench(b: &mut Bencher){
  //   // Define the OpenCL source code
  //   let src = include_str!("radix.cl");
  
  //   const WORKGROUP_SIZE: usize = 256; 
  //   let n: usize = 32_768;//rand::thread_rng().gen_range(2..5_000_000); 
  //   assert!(n <= 262_144); // <= 1024*256 since 1024 items/workgroup are the maximum
  //   let n_256 = ceil_div(n, 256);
  //   println!("hist array size: {}", n_256);
  
  //   let pro_que = ProQue::builder()
  //     .src(src)
  //     .dims(n_256)
  //     .build().unwrap();
  
  //   let input = pro_que.create_buffer::<Uint2>().unwrap();
  //   let output = pro_que.create_buffer::<Uint2>().unwrap();
  //   let histo_zeros  =  vec![0u32; n_256];
  //   let histograms = Buffer::builder()
  //     .queue(pro_que.queue().clone())
  //     .flags(MemFlags::new().read_write())
  //     .len(n_256)
  //     .fill_val(0u32)
  //     .build().unwrap();
  //   let count_zeros = vec![0u32; 256];
  //   let counts = Buffer::builder()
  //     .queue(pro_que.queue().clone())
  //     .flags(MemFlags::new().read_write())
  //     .len(256)
  //     .fill_val(0u32)
  //     .build().unwrap();
  
  //   let radix_a = pro_que.kernel_builder("radix_sort_histograms")
  //     .arg(&input)
  //     .arg(&output)
  //     .arg(&histograms)
  //     .arg(&counts)
  //     .arg_local::<u32>(WORKGROUP_SIZE)
  //     .arg(0u32)
  //     .arg(n as u32)
  //     .local_work_size(WORKGROUP_SIZE)
  //     .global_work_size(n_256) 
  //     .build().unwrap();

  //   let radix_b = pro_que.kernel_builder("radix_sort_b")
  //     .arg(&histograms)
  //     .arg(&counts)
  //     .arg(n_256 as u32)
  //     .local_work_size(256)
  //     .global_work_size(n_256) 
  //     .build().unwrap();
  //   let radix_c = pro_que.kernel_builder("radix_sort_reorder")
  //     .arg(&input)
  //     .arg(&output)
  //     .arg(&histograms)
  //     .arg(&counts)
  //     .arg_local::<u32>(WORKGROUP_SIZE)
  //     .arg(0u32)
  //     .arg(n as u32)
  //     .local_work_size(WORKGROUP_SIZE)
  //     .global_work_size(n_256) 
  //     .build().unwrap();
  
  
  //   // initialize data to sort
  //   let initial_state = (0..n_256)
  //     // .map(|i| Uint2::new((n-i-1) as u32, i as u32))
  //     .map(|i| Uint2::new(rand_uint(), i as u32))
  //     .collect::<Vec<Uint2>>();


  //   b.iter(|| {
  //     input.write(&initial_state).enq().unwrap();
  //     output.write(&initial_state).enq().unwrap();
    
  //     // execute the kernel passes
  //     for (i, shift) in (0u32..32).step_by(8).enumerate(){
  //       // set bitshift argument
  //       radix_a.set_arg(5, shift).unwrap();
  //       radix_c.set_arg(5, shift).unwrap();
  //       // swap input and output buffers
  //       radix_a.set_arg(0, if i%2==0 {&input} else {&output}).unwrap();
  //       radix_a.set_arg(1, if i%2==0 {&output} else {&input}).unwrap();
  //       radix_c.set_arg(0, if i%2==0 {&input} else {&output}).unwrap();
  //       radix_c.set_arg(1, if i%2==0 {&output} else {&input}).unwrap();
  //       // reset histograms
  //       counts.write(&count_zeros).enq().unwrap();
  //       histograms.write(&histo_zeros).enq().unwrap();
  //       //enqueue kernels
  //       unsafe {radix_a.enq().unwrap();}
  //       // unsafe {radix_b_counts.enq().unwrap();}
  //       unsafe {radix_b.enq().unwrap();}
  //       unsafe {radix_c.enq().unwrap();}
  //     }
  //   });
   
  //   // show the result
  //   let mut res_in = initial_state.clone();
  //   let mut res_out = initial_state.clone();
  //   let mut res_hist = vec![0;n_256];
  //   let mut res_counts = vec![0;256];
  //   input.read(&mut res_in).enq().unwrap();
  //   output.read(&mut res_out).enq().unwrap();
  //   histograms.read(&mut res_hist).enq().unwrap();
  //   counts.read(&mut res_counts).enq().unwrap();
  // }
}

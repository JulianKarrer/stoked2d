
use std::sync::atomic::Ordering::Relaxed;
use ocl::{prm::{Float, Float2, Uchar3, Uint2}, Buffer, MemFlags, ProQue};
use crate::{utils::next_multiple, BOUNDARY, BOUNDARY_LAYER_COUNT, FLUID, H, INITIAL_DT, KERNEL_SUPPORT, VIDEO_SIZE, WARP, WORKGROUP_SIZE};

use super::kernels::Kernels;



pub struct GpuBuffers{
  // general info
  pub n: usize,
  // single-value buffers
  pub dt:Buffer<Float>,
  pub current_t:Buffer<Float>,
  // particle attributes
  pub pos: Buffer<Float2>,
  pub vel: Buffer<Float2>,
  pub acc: Buffer<Float2>,
  pub den: Buffer<Float> ,
  pub prs: Buffer<Float> ,
  pub handles: Buffer<Uint2>,
  pub neighbours: Buffer<i32>,
  // boundary handling
  pub bdy: Buffer<Float2>,
  pub bdy_handles: Buffer<Uint2>,
  pub bdy_neighbours: Buffer<i32>,
  // buffers for sorting
  pub handles_temp: Buffer<Uint2>,
  pub histograms: Buffer<u32>,
  pub counts: Buffer<u32>,
  pub counts_b: Buffer<u32>,
  // host side buffers for zeroing via 'write'
  pub hist_zeros: Vec<u32>, 
  pub counts_zeros: Vec<u32>,
  pub counts_b_zeros: Vec<u32>,
  // buffers for resorting
  pub pos_resort: Buffer<Float2>,
  pub vel_resort: Buffer<Float2>,
  // buffers for reductions
  pub pos_min: Buffer<Float2>,
  pub vel_max: Buffer<Float>,
  // buffers for rendering,
  pub image: Buffer<Uchar3>
}

impl GpuBuffers{
  /// Creates a new instance of device-side buffers and a few host buffers
  /// for resorting to preserve memory coherence.
  pub fn new(pro_que: &ProQue, n:usize, n_bdy:usize)->Self{
    let n_256 = next_multiple(n as usize, WORKGROUP_SIZE);
    let n_groups = n_256/256;
    let splinters = 1 + ((n_groups - 1) / WARP);
    Self{ 
      n,
      dt: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(1)
        .fill_val(Float::new(INITIAL_DT.load(Relaxed) as f32))
        .build().unwrap(),
      current_t: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(1)
        .fill_val(Float::new(0.0))
        .build().unwrap(),

      // buffers for particle attributes
      pos: pro_que.create_buffer::<Float2>().unwrap(), 
      vel: pro_que.create_buffer::<Float2>().unwrap(), 
      acc: pro_que.create_buffer::<Float2>().unwrap(), 
      den: pro_que.create_buffer::<Float>().unwrap() , 
      prs: pro_que.create_buffer::<Float>().unwrap() , 
      // buffers for boundary handling
      bdy: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(n_bdy)
        .fill_val(Float2::new(0.,0.))
        .build().unwrap(),
      bdy_handles: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(n_bdy)
        .fill_val(Uint2::new(0,0))
        .build().unwrap(),
      bdy_neighbours: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(n*3)
        .copy_host_slice(&vec![-1;n*3])
        .build().unwrap(),
      // buffers for neighbour search
      handles: pro_que.create_buffer::<Uint2>().unwrap(), 
      handles_temp: pro_que.create_buffer::<Uint2>().unwrap(), 
      neighbours: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(n*3)
        .copy_host_slice(&vec![-1;n*3])
        .build().unwrap(),
      // buffers for resorting
      pos_resort: pro_que.create_buffer::<Float2>().unwrap(), 
      vel_resort: pro_que.create_buffer::<Float2>().unwrap(),
      // buffers for sorting
      histograms: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(n_256)
        .fill_val(0u32)
        .build().unwrap(),
      counts: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(256)
        .fill_val(0u32)
        .build().unwrap(),
      counts_b: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(splinters*256)
        .fill_val(0u32)
        .build().unwrap(),
      hist_zeros: vec![0u32; n_256],
      counts_zeros: vec![0u32; 256],
      counts_b_zeros: vec![0u32; splinters*256],
      // buffers for reducing / finding min pos and max vel
      pos_min: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(next_multiple(n, WORKGROUP_SIZE)/WORKGROUP_SIZE)
        .fill_val(Float2::new(0.0, 0.0))
        .build().unwrap(),
      vel_max: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(next_multiple(n, WORKGROUP_SIZE)/WORKGROUP_SIZE)
        .fill_val(Float::new(0.0))
        .build().unwrap(),
      image: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(VIDEO_SIZE.0*VIDEO_SIZE.1)
        .fill_val(Uchar3::new(0, 0, 0))
        .build().unwrap(),
    }
  }


  pub fn init_cpu_side(
    pos: &mut Vec<Float2>, 
    vel: &mut Vec<Float2>, 
    bdy: &mut Vec<Float2>
  )->(usize,usize){
    // initialization of positions and velocities
    let mut x = FLUID[0].x as f32;
    let mut y = FLUID[0].y as f32;
    while x <= FLUID[1].x as f32{
      while y <= FLUID[1].y as f32{
        pos.push(Float2::new(x, y));
        vel.push(Float2::new(0.0, 0.0));
        y += H as f32;
      }
      y = FLUID[0].y as f32;
      x += H as f32;
    }
    // initialization of boundary particles
      // horizontal
    for i in 0..BOUNDARY_LAYER_COUNT{
      let mut x = BOUNDARY[0].x+H;
      while x <= BOUNDARY[1].x{
        bdy.push(Float2::new(x as f32, (BOUNDARY[0].y-i as f64*H) as f32));
        bdy.push(Float2::new(x as f32, (BOUNDARY[1].y+i as f64*H) as f32));
        x += H
      }
    }
    // vertical
    for i in 0..BOUNDARY_LAYER_COUNT{
      let mut y = BOUNDARY[0].y-(BOUNDARY_LAYER_COUNT-1) as f64*H;
      while y < BOUNDARY[1].y+(BOUNDARY_LAYER_COUNT) as f64*H{
        bdy.push(Float2::new((BOUNDARY[0].x-i as f64*H) as f32, y as f32));
        bdy.push(Float2::new((BOUNDARY[1].x+i as f64*H) as f32, y as f32));
        y += H
      }
    }
    (pos.len(), bdy.len())
  }

  pub fn init_gpu_side(&self, n:usize, pos: &Vec<Float2>, vel: &Vec<Float2>, bdy: &Vec<Float2>, k:&Kernels, pro_que: &ProQue){
    self.pos.write(&*pos).enq().unwrap();
    self.vel.write(&*vel).enq().unwrap();
    self.acc.write(&vec![Float2::new(0.0, 0.0);n]).enq().unwrap();
    self.handles.write(&(
      (0..pos.len() as u32).map(|i|Uint2::new(i,i)).collect::<Vec<Uint2>>()
    )).enq().unwrap();

    // calculate the cell keys on the gpu
      // write unsorted bdy and bdy_handles
    self.bdy.write(&*bdy).enq().unwrap();
    let mut bdy_handles = (0..bdy.len())
      .map(|i|Uint2::new(0,i as u32))
      .collect::<Vec<Uint2>>();
    self.bdy_handles.write(&bdy_handles).enq().unwrap();
      // prepare and enqueue calculation of cell keys
    k.reduce_pos_min(n);
    let compute_bdy_cell_keys_kernel = pro_que.kernel_builder("compute_cell_keys")
      .arg(&self.pos_min)
      .arg(KERNEL_SUPPORT as f32)
      .arg(&self.bdy)
      .arg(&self.bdy_handles)
      .arg(bdy.len() as u32)
      .global_work_size(bdy.len())
      .build().unwrap();
    unsafe { compute_bdy_cell_keys_kernel.enq().unwrap(); }
    self.bdy_handles.read(&mut bdy_handles).enq().unwrap();
    // sort bdy by cell keys
    bdy_handles.sort_unstable_by_key(|h| h[0]);
    let mut bdy_sorted = bdy.clone();
    // use cell keys to resort bdy and bdy_handles
    for (i, h) in bdy_handles.iter().enumerate(){
      bdy_sorted[i] = bdy[h[1] as usize]
    }
    bdy_handles.iter_mut().enumerate().for_each(|(i,h)| h[1] = i as u32);
    // write back sorted bdy and bdy_handles
    self.bdy.write(&*bdy_sorted).enq().unwrap();
    self.bdy_handles.write(&bdy_handles).enq().unwrap();
    // assert that cell keys in handles are sorted and indices into bdy are 0,1,...,n_bdy
    for (i, h) in bdy_handles.iter().enumerate() {
      if i>0{
        assert!(h[0]>= bdy_handles[i-1][0]);
        assert!(h[1] == i as u32);
      }
    }
  }
}

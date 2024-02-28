use ocl::{prm::{Float, Float2}, Buffer, MemFlags, ProQue};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{FLUID, H, RESORT_ATTRIBUTES_EVERY_N};


pub struct GpuBuffers{
  pub pos: Buffer<Float2>,
  pub vel: Buffer<Float2>,
  pub acc: Buffer<Float2>,
  pub den: Buffer<Float> ,
  pub prs: Buffer<Float> ,
  pub handle_cells: Buffer<u32>,
  pub handle_indices: Buffer<u32>,
  pub neighbours: Buffer<i32>,
  pub pos_resort: Vec<Float2> ,
  pub vel_resort: Vec<Float2> ,
  pub indices_resort: Vec<u32>,
}

impl GpuBuffers{
  /// Creates a new instance of device-side buffers and a few host buffers
  /// for resorting to preserve memory coherence.
  pub fn new(pro_que: &ProQue, n:usize)->Self{
    Self{ 
      // buffers for particle attributes
      pos: pro_que.create_buffer::<Float2>().unwrap(), 
      vel: pro_que.create_buffer::<Float2>().unwrap(), 
      acc: pro_que.create_buffer::<Float2>().unwrap(), 
      den: pro_que.create_buffer::<Float>().unwrap() , 
      prs: pro_que.create_buffer::<Float>().unwrap() , 
      // buffers for neighbour search
      handle_cells: pro_que.create_buffer::<u32>().unwrap(), 
      handle_indices: pro_que.create_buffer::<u32>().unwrap(), 
      neighbours: Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(n*3)
        .copy_host_slice(&vec![-1;n*3])
        .build().unwrap(),
      // buffers for resorting
      pos_resort: vec![Float2::new(0.0, 0.0); n],
      vel_resort: vec![Float2::new(0.0, 0.0); n],
      indices_resort: vec![0u32; n],
    }
  }

  /// Load positions from the buffer to host memory and resort positions and
  /// velocities every n iterations to ensure memory coherency, updating
  /// the device side buffers
  pub fn load_and_resort_pos_vel(&mut self, since_resort: &mut u32, pos: &mut [Float2], vel: &mut [Float2]){
    if *since_resort > {*RESORT_ATTRIBUTES_EVERY_N.read()} {
      self.pos.read(&mut self.pos_resort).enq().unwrap();
      self.vel.read(&mut self.vel_resort).enq().unwrap();
      self.handle_indices.read(&mut self.indices_resort).enq().unwrap();
      pos.par_iter_mut()
        .zip(&mut *vel)
        .zip(&self.indices_resort)
        .for_each(|((p,v), i)|{
          debug_assert!(*i>0);
          *p = self.pos_resort[*i as usize];
          *v = self.vel_resort[*i as usize];
        });
      self.pos.write(&*pos).enq().unwrap();
      self.vel.write(&*vel).enq().unwrap();
      *since_resort = 0;
    } else {
      self.pos.read(&mut *pos).enq().unwrap();
      *since_resort += 1;
    }
  }

  pub fn init_fluid_pos_vel(&self, pos: &mut Vec<Float2>, vel: &mut Vec<Float2>, n:usize){
    let mut acc:Vec<Float2> = Vec::with_capacity(n); 
    // initialization
    let mut x = FLUID[0].x as f32;
    let mut y = FLUID[0].y as f32;
    while x <= FLUID[1].x as f32{
      while y <= FLUID[1].y as f32{
        pos.push(Float2::new(x, y));
        vel.push(Float2::new(0.0, 0.0));
        acc.push(Float2::new(0.0, -9.81));
        y += H as f32;
      }
      y = FLUID[0].y as f32;
      x += H as f32;
    }
    assert!(
      pos.len() <= n, 
      "n={} must be >= to pos.len()={} for gpu buffer dimensions to match", 
      n, pos.len()
    );
    // fill buffers with values
    self.pos.write(&*pos).enq().unwrap();
    self.vel.write(&*vel).enq().unwrap();
    self.acc.write(&acc).enq().unwrap();
  }
}

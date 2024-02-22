extern crate ocl;

use std::{ops::RangeBounds, sync::OnceLock};

use ocl::{prm::{Float, Float2}, Buffer, MemFlags, ProQue};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use voracious_radix_sort::RadixSort;
use crate::{simulation::update_fps, sph::KERNEL_CUBIC_NORMALIZE, *};

use self::datastructure::Handle;

pub fn run(){
  // init
  let n: usize = ((FLUID[1].x-FLUID[0].x)/(H) + 1.0).ceil() as usize * ((FLUID[1].y-FLUID[0].y)/(H) + 1.0).ceil() as usize;

  let pro_que = ProQue::builder()
    .src(include_str!("kernels.cl"))
    .dims(n)
    .build().unwrap();
  println!("Number of particles: {}",n);

  // allocation
  let pos_b: Buffer<Float2> = pro_que.create_buffer::<Float2>().unwrap();
  let vel_b: Buffer<Float2> = pro_que.create_buffer::<Float2>().unwrap();
  let acc_b: Buffer<Float2> = pro_que.create_buffer::<Float2>().unwrap();
  let den_b: Buffer<Float> = pro_que.create_buffer::<Float>().unwrap();
  let prs_b: Buffer<Float> = pro_que.create_buffer::<Float>().unwrap();
  let mut pos:Vec<Float2> = Vec::with_capacity(n); 
  let mut vel:Vec<Float2> = Vec::with_capacity(n); 
  {
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
    // fill buffers with values
    pos_b.write(&pos).enq().unwrap();
    vel_b.write(&vel).enq().unwrap();
    acc_b.write(&acc).enq().unwrap();
  }
  let mut den: Vec<Float> = vec![Float::new((M/(H*H)) as f32);pos.len()];
  let mut handles:Vec<Handle> = (0..pos.len()).map(|i| Handle::new(i, 0)).collect();
  let mut neighbours:Vec<i32> = vec![-1;pos.len()*3];
  let mut handle_cells:Vec<u32> = Vec::with_capacity(pos.len());
  let mut handle_indices:Vec<u32> = Vec::with_capacity(pos.len());
  let handle_cells_b: Buffer<u32> = pro_que.create_buffer::<u32>().unwrap();
  let handle_indices_b: Buffer<u32> = pro_que.create_buffer::<u32>().unwrap();
  let neighbours_b = Buffer::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().read_write())
    .len(pos.len()*3)
    .copy_host_slice(&neighbours)
    .build().unwrap();
  
  // define time step size
  let mut current_t = 0.0f32;
  let mut dt = INITIAL_DT.load(Relaxed) as f32;

  // build relevant kernels
  let euler_cromer_kernel = pro_que.kernel_builder("eulercromerstep")
    .arg(&pos_b)
    .arg(&vel_b)
    .arg(&acc_b)
    .arg(dt)
    .build().unwrap();
  let boundary_kernel = pro_que.kernel_builder("enforce_boundary")
    .arg(&pos_b)
    .arg(Float2::new(BOUNDARY[0][0] as f32, BOUNDARY[0][1] as f32))
    .arg(Float2::new(BOUNDARY[1][0] as f32, BOUNDARY[1][1] as f32))
    .build().unwrap();
  let densitiy_pressure_kernel = pro_que.kernel_builder("update_densities_pressures")
    .arg(K.load(Relaxed) as f32)
    .arg(RHO_ZERO.load(Relaxed) as f32)
    .arg(&den_b)
    .arg(&prs_b)
    .arg(&pos_b)
    .arg(&handle_cells_b)
    .arg(&handle_indices_b)
    .arg(&neighbours_b)
    .arg(M as f32)
    .arg(KERNEL_CUBIC_NORMALIZE as f32)
    .arg(H as f32)
    .arg(pos.len() as u32)
    .build().unwrap();
  let gravity_viscosity_kernel = pro_que.kernel_builder("apply_gravity_viscosity")
    .arg(NU.load(Relaxed) as f32)
    .arg(GRAVITY.load(Relaxed) as f32)
    .arg(&acc_b)
    .arg(&vel_b)
    .arg(&pos_b)
    .arg(&den_b)
    .arg(&handle_cells_b)
    .arg(&handle_indices_b)
    .arg(&neighbours_b)
    .arg(M as f32)
    .arg(KERNEL_CUBIC_NORMALIZE as f32)
    .arg(H as f32)
    .arg(pos.len() as u32)
    .build().unwrap();
  let pressure_acceleration_kernel = pro_que.kernel_builder("add_pressure_acceleration")
    .arg(RHO_ZERO.load(Relaxed) as f32)
    .arg(&pos_b)
    .arg(&acc_b)
    .arg(&den_b)
    .arg(&prs_b)
    .arg(&handle_cells_b)
    .arg(&handle_indices_b)
    .arg(&neighbours_b)
    .arg(M as f32)
    .arg(KERNEL_CUBIC_NORMALIZE as f32)
    .arg(H as f32)
    .arg(pos.len() as u32)
    .build().unwrap();

  // MAIN LOOP
  let mut last_update_time = timestamp();
  {  HISTORY.write().gpu_reset_and_add(&pos, &den, current_t.into()); }
  while !REQUEST_RESTART.fetch_and(false, Relaxed) {
    // update atomics to kernel programs
    densitiy_pressure_kernel.set_arg(0, K.load(Relaxed) as f32).unwrap();
    densitiy_pressure_kernel.set_arg(1, RHO_ZERO.load(Relaxed) as f32).unwrap();
    gravity_viscosity_kernel.set_arg(0, NU.load(Relaxed) as f32).unwrap();
    gravity_viscosity_kernel.set_arg(1, GRAVITY.load(Relaxed) as f32).unwrap();
    pressure_acceleration_kernel.set_arg(0, RHO_ZERO.load(Relaxed) as f32).unwrap();

    // update grid
      // copy to host and cpu side computations
    pos_b.read(&mut pos).enq().unwrap();
    update_grid(&pos, &mut handles, &mut neighbours);
    handles.par_iter().map(|h|h.cell as u32).collect_into_vec(&mut handle_cells);
    handles.par_iter().map(|h|h.index as u32).collect_into_vec(&mut handle_indices);
      // copy results back to GPU buffers
    handle_cells_b.write(&handle_cells).enq().unwrap();
    handle_indices_b.write(&handle_indices).enq().unwrap();
    neighbours_b.write(&neighbours).enq().unwrap();

    // update densities and pressures
    unsafe { densitiy_pressure_kernel.enq().unwrap(); }
    // apply gravity and viscosity
    unsafe { gravity_viscosity_kernel.enq().unwrap(); }
    // add pressure accelerations
    unsafe { pressure_acceleration_kernel.enq().unwrap(); }

    // integrate accelerations to position updates
    vel_b.read(&mut vel).enq().unwrap();
    dt = update_dt_gpu(&vel, &mut current_t);
    euler_cromer_kernel.set_arg(3, dt).unwrap();
    unsafe { euler_cromer_kernel.enq().unwrap(); }
    unsafe { boundary_kernel.enq().unwrap(); }
    

    // write back the positions to the global buffer for visualization and update the FPS count
    update_fps(&mut last_update_time);
    den_b.read(&mut den).enq().unwrap();
    {  HISTORY.write().gpu_add_step(&pos, &den, current_t.into()); }
  }
}

/// Update all cell indices in the array of handles. Returns the minimum point from which the 
/// space filling XY-curve starts.
fn update_cell_keys(pos:&[Float2], handles:&mut [Handle])->Float2{
  let min: Float2 = pos.par_iter().cloned().reduce(
    || Float2::new(f32::MAX, f32::MAX), 
    |a,b|Float2::new(a[0].min(b[0]), a[1].min(b[1]))) 
    - Float2::new(2.0* KERNEL_SUPPORT as f32, 2.0*KERNEL_SUPPORT as f32);
  handles.par_iter_mut().for_each(|c| c.cell = cell_key(&pos[c.index], &min));
  min
}

/// Calculate cell index from position and minimum point of the domain extent
fn cell_key(pos:&Float2, min:&Float2) -> u64{ 
  let x = ((pos[0]-min[0])/KERNEL_SUPPORT as f32) as u32;
  let y = ((pos[1]-min[1])/KERNEL_SUPPORT as f32) as u32;
  ((y as u64) << 16) | x as u64
}

/// Uses binary search to find the first occurence of an element in `look_for` 
/// amongst the keys of an ***ordered*** `arr`, returning the index of the result on success. 
/// The `extract_key` function is used to convert items in `arr` into such keys.
fn binary_first_occurence_by
<T:Copy+std::fmt::Debug,Q:PartialEq+PartialOrd,P:RangeBounds<Q>>
(arr: &[T], extract_key: fn(T)->Q, look_for: P)->Option<usize>{
  let mut low:i32 = 0; 
  let mut high:i32 = arr.len() as i32 -1;
  while low <= high{
    // search from the midpoint
    let mid = (high-low)/2+low;
    let hit = extract_key(arr[mid as usize]);
    if look_for.contains(&hit) && (mid==0 || !look_for.contains(&extract_key(arr[(mid-1) as usize]))){
      // if hit is the searched for key AND the first occurrence 
      // (i.e. the previous item is not equal to look_for), return
      return Some(mid as usize)
    } else {
      match look_for.start_bound() {
        std::ops::Bound::Included(lower) => if hit < *lower {low = mid+1} else {high = mid-1},
        std::ops::Bound::Excluded(lower) => if hit <= *lower {low = mid+1} else {high = mid-1},
        std::ops::Bound::Unbounded => panic!("Unbounded range was used in binary search"),
      }
    }
  }
  None
}

static ROWS_NEARBY:OnceLock<[Float2;3]> = OnceLock::new();
fn update_grid(pos: &[Float2], handles: &mut[Handle], neighbours: &mut[i32]){
  // update info on which particle is in which cell
  let min = update_cell_keys(pos, handles);
  // sort the handles by the cell key
  handles.voracious_mt_sort(*THREADS);
  // update the first neighbour particle j_k in each surrounding row k [0,1,2] for all particles i
  neighbours.chunks_mut(3).enumerate().for_each(|(i, n)|{
    ROWS_NEARBY.get_or_init(||[
      Float2::new(-KERNEL_SUPPORT as f32, -KERNEL_SUPPORT as f32),
      Float2::new(-KERNEL_SUPPORT as f32, 0.0),
      Float2::new(-KERNEL_SUPPORT as f32, KERNEL_SUPPORT as f32),
    ]).iter().enumerate().for_each(|(k, offset)|{
      // get the cell index of the first neighbouring particle in the given row
      let key: u64 = cell_key(&(pos[i]+(*offset)), &min);
      // look for the first neighbouring particle using its cell key
      n[k] = if let Some(ind) = binary_first_occurence_by(handles, |h|h.cell, key..key+3){
        ind as i32
      } else {-1}
    });
  })
}

fn len_float2(x:&Float2)->f32{(x[0]*x[0]+x[1]*x[1]).sqrt()}
fn update_dt_gpu(vel: &[Float2], current_t: &mut f32)->f32{
  let v_max = vel.par_iter().map(len_float2).reduce_with(|a,b| a.max(b)).unwrap();
  let mut dt = (LAMBDA.load(Relaxed) * H / v_max as f64)
    .min(MAX_DT.load(Relaxed)) as f32;
  if v_max < VELOCITY_EPSILON as f32 || !dt.is_normal() {dt = INITIAL_DT.load(Relaxed) as f32}
  *current_t += dt;
  dt
}


#[cfg(test)]
mod tests {
  use super::*;
  extern crate test;
  use rand::{thread_rng, Rng};
  fn random_float2(range:f32)->Float2{
    Float2::new(
      thread_rng().gen_range(-range..range), 
      thread_rng().gen_range(-range..range)
    )
  }

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

  #[test]
  fn gpu_binary_search_first_occurance(){
    assert_eq!(binary_first_occurence_by(&[1,2,5,5,5,6], |x|x, 1..2).unwrap(),0);
    assert_eq!(binary_first_occurence_by(&[1,2,5,5,5,6], |x|x, 4..=5).unwrap(),2);
    assert_eq!(binary_first_occurence_by(&[1,2,5,5,5,6], |x|x, 6..9).unwrap(),5);
    assert_eq!(binary_first_occurence_by(&[1,2,3,4,5,6,7,8], |x|x, 3..=6).unwrap(),2);
    assert_eq!(binary_first_occurence_by(&[Some(1),Some(2),Some(4)], |x|x.unwrap() as f32, 1.5..3.5).unwrap(),1);
  }

  
  #[test]
  fn gpu_grid_update(){
    // create grid of test positions
    let pos:Vec<Float2> = (0..10_000)
      .map(|_|random_float2((100.*H)as f32))
      .collect();
    // create ground truth of actual neighbour sets using brute force
    let actual_neighbour_sets:Vec<Vec<u32>> = pos.iter().map(|x_i|
      pos.iter().enumerate().filter(|(_, x_j)| 
        ((x_i[0]-x_j[0]).powi(2)+(x_i[1]-x_j[1]).powi(2)).sqrt() <= KERNEL_SUPPORT as f32
      ).map(|(i,_)|i as u32).collect::<Vec<u32>>()
    ).collect();
    
    // perform neighbour search as implemented above
    let mut handles:Vec<Handle> = (0..pos.len())
      .map(|i| Handle::new(i, 0))
      .collect();
    let mut neighbours = vec![-1; pos.len()*3];
    update_grid(&pos, &mut handles, &mut neighbours);

    // get the neighbour sets from the results
    let cells:Vec<u32> = handles
      .iter()
      .map(|h|h.cell as u32)
      .collect();
    let indices:Vec<u32> = handles
      .iter()
      .map(|h|h.index as u32)
      .collect();

    let result_neighbour_sets:Vec<Vec<u32>> = pos
      .iter()
      .enumerate()
      .map(|(i,_)|{
        let mut res = vec![];
        for k in 0..3{
          let mut j = neighbours[3*i+k];
          if j>=0{
            let init_cell = cells[j as usize];
            while (j as usize) < pos.len() && cells[j as usize]<init_cell+3{
              res.push(indices[j as usize]);
              j += 1;
            }
          }
        }
        res
      })
      .collect();
    // check if the result is a superset of the actual neighbours
    for i in 0..pos.len(){
      for j in &actual_neighbour_sets[i]{
        assert!(
          result_neighbour_sets[i].contains(j),
          "i:{}, j:{}\nactual neighbours: {:?}\n\nneighbours: {:?}\n\n\nn:{:?}\n\nc:{:?}\n\nin:{:?}",
          i,j,actual_neighbour_sets,result_neighbour_sets, neighbours, cells, indices
        )
      }
    }

  }
}
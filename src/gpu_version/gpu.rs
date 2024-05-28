extern crate ocl;

use std::time::Duration;

use crate::{
    gpu_version::{buffers::GpuBuffers, kernels::Kernels},
    gui::gui::{REQUEST_RESTART, SIMULATION_TOGGLE},
    simulation::update_fps,
    *,
};
use ocl::{
    prm::{Float, Uint2},
    ProQue,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use utils::create_progressbar;
// use crate::video::VideoHandler

pub fn run(run_for_t: Option<f32>) -> bool {
    let n_est: usize = ((FLUID[1].x - FLUID[0].x) / (H) + 1.0).ceil() as usize
        * ((FLUID[1].y - FLUID[0].y) / (H) + 1.0).ceil() as usize;

    // buffer allocation and initialization
    let mut bdy: Vec<Float2> = Vec::new();
    let mut pos: Vec<Float2> = Vec::with_capacity(n_est);
    let mut vel: Vec<Float2> = Vec::with_capacity(n_est);
    let (n, n_bdy) = GpuBuffers::init_cpu_side(&mut pos, &mut vel, &mut bdy);
    let pro_que: ProQue = ProQue::builder()
        .src(include_str!("kernels.cl"))
        .dims(n)
        .build()
        .unwrap();
    println!("Number of particles: {}", n);
    let b: GpuBuffers = GpuBuffers::new(&pro_que, n, n_bdy);
    // build relevant kernels
    let k = Kernels::new(&b, &pro_que, pos.len() as u32);
    // finish initializing buffers on gpu side
    b.init_gpu_side(n, &pos, &vel, &bdy, &k, &pro_que);
    let mut den: Vec<Float> = vec![Float::new((1.0) as f32); pos.len()];
    let mut handles: Vec<Uint2> = (0..pos.len())
        .map(|i| Uint2::new(i as u32, i as u32))
        .collect();

    // define time step size
    // let mut current_t = 0.0f32;
    let mut last_gui_update_t = 0.0f32;
    // let mut dt = INITIAL_DT.load(Relaxed) as f32;

    // intitial history timestep for visualization
    let handle_indices = handles.par_iter().map(|h| h[0]).collect::<Vec<u32>>();
    {
        HISTORY
            .write()
            .gpu_reset_and_add(&pos, &vel, &bdy, &handle_indices, &den, 0.0);
    }

    // set up video handler for rendering and progress bar for feedback
    // let mut vid = VideoHandler::default();
    let progressbar = create_progressbar(run_for_t);

    // MAIN LOOP
    let mut last_update_time = timestamp();
    let mut since_resort = 0;
    let mut current_t = vec![Float::new(0.0)];
    k.reduce_pos_min(b.n);
    while !*REQUEST_RESTART.read() {
        // wait if requested
        while *SIMULATION_TOGGLE.read() {
            thread::sleep(Duration::from_millis(10));
        }
        // update atomics to kernel programs
        k.update_atomics();

        // integrate accelerations to position updates
        k.update_dt_reduce_vel(&b);
        unsafe {
            k.euler_cromer.enq().unwrap();
            k.boundary.enq().unwrap();
        }

        // update grid
        if since_resort > { *RESORT_ATTRIBUTES_EVERY_N.read() } {
            unsafe {
                k.resort_pos_vel.enq().unwrap();
                k.resort_pos_vel_b.enq().unwrap();
            }
            since_resort = 0;
        } else {
            since_resort += 1
        }

        // k.reduce_pos_min(&b);
        unsafe {
            k.compute_cell_keys.enq().unwrap();
        }
        k.sort_handles(&b);
        unsafe {
            k.compute_neighbours.enq().unwrap();
        }
        if USE_GPU_BOUNDARY {
            unsafe {
                k.compute_boundary_neighbours.enq().unwrap();
            }
        }

        unsafe {
            // update densities and pressures
            k.densitiy_pressure.enq().unwrap();
            // apply gravity and viscosity
            k.gravity_viscosity.enq().unwrap();
            // add pressure accelerations
            k.pressure_acceleration.enq().unwrap();
        }

        // update the FPS count
        update_fps(&mut last_update_time);
        // every FRAME_TIME second, update GUI, render frame for video and update progress bar
        b.current_t.read(&mut current_t).enq().unwrap();
        if current_t[0][0] - last_gui_update_t > HISTORY_FRAME_TIME {
            last_gui_update_t = current_t[0][0];
            // for video
            unsafe { k.render.enq().unwrap() }
            // vid.add_frame(&b);
            // for progressbar
            if let Some(ref bar) = progressbar {
                bar.inc(1);
                bar.set_message(format!("{:.3} ITERS/S", SIM_FPS.load(Relaxed)));
            }
            // for GUI
            b.pos.read(&mut pos).enq().unwrap();
            b.vel.read(&mut vel).enq().unwrap();
            b.den.read(&mut den).enq().unwrap();
            b.handles.read(&mut handles).enq().unwrap();
            let handle_indices = handles.par_iter().map(|h| h[1]).collect::<Vec<u32>>();
            // analyze_handles(&handles);
            {
                HISTORY.write().gpu_add_step(
                    &pos,
                    &vel,
                    &handle_indices,
                    &den,
                    current_t[0][0] as f64,
                );
            }
        }
        // if only a specific time frame was requested, stop the simulation
        if let Some(max_t) = run_for_t {
            if max_t <= current_t[0][0] {
                progressbar.unwrap().finish();
                // vid.finish();
                return false;
            }
        }
    }
    *REQUEST_RESTART.write() = false;
    true
}

/// Calculate the length of a Float2
pub fn len_float2(x: &Float2) -> f64 {
    let xc: f64 = x[0] as f64;
    let yc: f64 = x[1] as f64;
    ((xc * xc) + (yc * yc)).sqrt()
}

// #[cfg(test)]
// mod tests {
//     extern crate test;
//     use crate::utils::next_multiple;

//     use super::*;
//     use approx::relative_eq;
//     use ocl::{Buffer, MemFlags};
//     use rand::Rng;
//     use test::Bencher;

//     #[test]
//     fn gpu_sorting_radix_small() {
//         for _ in 0..20 {
//             gpu_sorting_radix(rand::thread_rng().gen_range(2..262_144))
//         }
//     }

//     #[test]
//     fn gpu_sorting_radix_big() {
//         for _ in 0..10 {
//             gpu_sorting_radix(rand::thread_rng().gen_range(262_144..1_000_000))
//         }
//     }

//     fn rand_uint() -> u32 {
//         rand::thread_rng().gen_range(u32::MIN..u32::MAX)
//     }
//     fn gpu_sorting_radix(n: usize) {
//         let pro_que: ProQue = ProQue::builder()
//             .src(include_str!("kernels.cl"))
//             .dims(n)
//             .build()
//             .unwrap();
//         let b: GpuBuffers = GpuBuffers::new(&pro_que, n, n);
//         let mut handles: Vec<Uint2> = (0..n).map(|i| Uint2::new(rand_uint(), i as u32)).collect();
//         b.handles.write(&handles).enq().unwrap();

//         let k = Kernels::new(&b, &pro_que, n as u32);
//         k.sort_handles(&b);
//         b.handles.read(&mut handles).enq().unwrap();
//         handles
//             .iter()
//             .enumerate()
//             .skip(1)
//             .for_each(|(i, h)| assert!(h[0] >= handles[i - 1][0]))
//     }

//     #[bench]
//     fn gpu_radix_bench(b: &mut Bencher) {
//         let n = 30_000;
//         let pro_que: ProQue = ProQue::builder()
//             .src(include_str!("kernels.cl"))
//             .dims(n)
//             .build()
//             .unwrap();
//         let buffers: GpuBuffers = GpuBuffers::new(&pro_que, n, n);
//         let kernels = Kernels::new(&buffers, &pro_que, n as u32);
//         let mut handles: Vec<Uint2> = (0..n).map(|i| Uint2::new(rand_uint(), i as u32)).collect();

//         b.iter(|| {
//             buffers.handles.write(&handles).enq().unwrap();
//             kernels.sort_handles(&buffers);
//         });

//         buffers.handles.read(&mut handles).enq().unwrap();
//     }

//     #[test]
//     fn gpu_reduce() {
//         reduce_test(rand::thread_rng().gen_range(1..5_000_000))
//         // reduce_test(1_000_000)
//     }

//     #[test]
//     fn gpu_reduce_repeated() {
//         for _ in 0..20 {
//             gpu_reduce()
//         }
//     }

//     fn random_float() -> f32 {
//         rand::thread_rng().gen_range(-1_000_000.0..1_000_000.0)
//     }
//     fn small_random_float() -> f32 {
//         rand::thread_rng().gen_range(-100.0..100.0)
//     }
//     fn reduce_test(n: usize) {
//         let workgroup_size = 256;
//         let mut cur_size = n;
//         let pro_que: ProQue = ProQue::builder()
//             .src(include_str!("kernels.cl"))
//             .dims(cur_size)
//             .build()
//             .unwrap();
//         let pos: Vec<Float2> = (0..n)
//             .map(|_| Float2::new(random_float(), random_float()))
//             .collect();
//         let pos_b = pro_que.create_buffer::<Float2>().unwrap();
//         pos_b.write(&pos).enq().unwrap();
//         let pos_res_b = Buffer::builder()
//             .queue(pro_que.queue().clone())
//             .flags(MemFlags::new().read_write())
//             .len(next_multiple(cur_size, workgroup_size) / workgroup_size)
//             .fill_val(Float2::new(0.0, 0.0))
//             .build()
//             .unwrap();

//         let reduce = pro_que
//             .kernel_builder("reduce_min")
//             .arg(&pos_b)
//             .arg(&pos_res_b)
//             .arg_local::<u32>(workgroup_size)
//             .arg(n as u32)
//             .arg(1u32)
//             .arg(Float2::zero())
//             .local_work_size(workgroup_size)
//             .build()
//             .unwrap();

//         while cur_size > 1 {
//             reduce.set_arg(3, cur_size as u32).unwrap();
//             unsafe {
//                 reduce
//                     .cmd()
//                     .global_work_size(next_multiple(cur_size, workgroup_size))
//                     .enq()
//                     .unwrap()
//             }
//             reduce.set_arg(4, 0u32).unwrap();
//             cur_size = next_multiple(cur_size, workgroup_size) / workgroup_size;
//         }

//         // test result
//         let mut min_res: Vec<Float2> = vec![Float2::new(42.0, 42.0); pos_res_b.len()];
//         pos_res_b.read(&mut min_res).enq().unwrap();
//         let expected = pos
//             .iter()
//             .cloned()
//             .reduce(|a, b| Float2::new(a[0].min(b[0]), a[1].min(b[1])))
//             .unwrap();
//         assert!(
//             min_res[0][0] == expected[0],
//             "expected {:?}, found {:?}, size {}, \n\n {:?}",
//             expected,
//             min_res[0],
//             n,
//             min_res
//         );
//         assert!(
//             min_res[0][1] == expected[1],
//             "expected {:?}, found {:?}, size {}, \n\n {:?}",
//             expected,
//             min_res[0],
//             n,
//             min_res
//         );
//     }

//     #[test]
//     fn gpu_reduce_max() {
//         reduce_max_test(rand::thread_rng().gen_range(1..5_000_000))
//         // reduce_max_test(1000)
//     }

//     #[test]
//     fn gpu_reduce_max_repeated() {
//         for _ in 0..20 {
//             gpu_reduce()
//         }
//     }

//     fn len_float2_f32(x: &Float2) -> f32 {
//         let xc: f32 = x[0];
//         let yc: f32 = x[1];
//         ((xc * xc) + (yc * yc)).sqrt()
//     }

//     fn reduce_max_test(n: usize) {
//         let workgroup_size = 256;
//         let mut cur_size = n;
//         let pro_que: ProQue = ProQue::builder()
//             .src(include_str!("kernels.cl"))
//             .dims(cur_size)
//             .build()
//             .unwrap();
//         let vel: Vec<Float2> = (0..n)
//             .map(|_| Float2::new(small_random_float(), small_random_float()))
//             .collect();
//         let vel_b = pro_que.create_buffer::<Float2>().unwrap();
//         vel_b.write(&vel).enq().unwrap();
//         let vel_max_b = Buffer::builder()
//             .queue(pro_que.queue().clone())
//             .flags(MemFlags::new().read_write())
//             .len(next_multiple(cur_size, workgroup_size) / workgroup_size)
//             .fill_val(Float::new(0.0))
//             .build()
//             .unwrap();

//         let reduce = pro_que
//             .kernel_builder("reduce_max_magnitude")
//             .arg(&vel_b)
//             .arg(&vel_max_b)
//             .arg_local::<u32>(workgroup_size)
//             .arg(n as u32)
//             .arg(1u32)
//             .local_work_size(workgroup_size)
//             .build()
//             .unwrap();

//         while cur_size > 1 {
//             reduce.set_arg(3, cur_size as u32).unwrap();
//             unsafe {
//                 reduce
//                     .cmd()
//                     .global_work_size(next_multiple(cur_size, workgroup_size))
//                     .enq()
//                     .unwrap()
//             }
//             reduce.set_arg(4, 0u32).unwrap();
//             cur_size = next_multiple(cur_size, workgroup_size) / workgroup_size;
//         }

//         // test result
//         let mut res: Vec<Float> = vec![Float::new(0.0); vel_max_b.len()];
//         vel_max_b.read(&mut res).enq().unwrap();
//         let expected = vel
//             .iter()
//             .cloned()
//             .fold(0.0f32, |acc, cur| len_float2_f32(&cur).max(acc));
//         assert!(
//             relative_eq!(res[0][0], expected),
//             "expected {:?}, found {:?}, size {}, \n\n {:?}",
//             expected,
//             res[0][0],
//             n,
//             res
//         );
//     }

//     #[bench]
//     fn gpu_reduce_bench(b: &mut Bencher) {
//         let n = 1_000_000;
//         let workgroup_size = 32;
//         let mut cur_size = n;
//         let pro_que: ProQue = ProQue::builder()
//             .src(include_str!("kernels.cl"))
//             .dims(cur_size)
//             .build()
//             .unwrap();
//         let pos: Vec<Float2> = (0..n)
//             .map(|_| Float2::new(random_float(), random_float()))
//             .collect();
//         let pos_b = pro_que.create_buffer::<Float2>().unwrap();
//         let pos_res_b = Buffer::builder()
//             .queue(pro_que.queue().clone())
//             .flags(MemFlags::new().read_write())
//             .len(next_multiple(cur_size, workgroup_size) / workgroup_size)
//             .fill_val(Float2::new(f32::MAX, f32::MAX))
//             .build()
//             .unwrap();

//         let reduce = pro_que
//             .kernel_builder("reduce_min")
//             .arg(&pos_b)
//             .arg(&pos_res_b)
//             .arg_local::<u32>(workgroup_size)
//             .arg(n as u32)
//             .arg(1u32)
//             .arg(Float2::zero())
//             .local_work_size(workgroup_size)
//             .build()
//             .unwrap();

//         b.iter(|| {
//             pos_b.write(&pos).enq().unwrap();
//             while cur_size > 1 {
//                 reduce.set_arg(3, cur_size as u32).unwrap();
//                 unsafe {
//                     reduce
//                         .cmd()
//                         .global_work_size(next_multiple(cur_size, workgroup_size))
//                         .enq()
//                         .unwrap()
//                 }
//                 reduce.set_arg(4, 0u32).unwrap();
//                 cur_size = next_multiple(cur_size, workgroup_size) / workgroup_size;
//             }
//         });
//     }
// }

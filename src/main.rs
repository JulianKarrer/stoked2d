#![feature(test)]
#![feature(iter_collect_into)]
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, atomic::AtomicU32};
use std::sync::atomic::Ordering::Relaxed;
use std::thread::{self, available_parallelism};
use std::time::{SystemTime, UNIX_EPOCH};
use atomic_float::{AtomicF32, AtomicF64};
use datastructure::{GridCurve, AtomicGridCurve};
use glam::DVec2;
use gui::gui::StokedWindowHandler;
use lazy_static::lazy_static;
use egui_speedy2d::egui::mutex::RwLock;
use ocl::prm::Float2;
use simulation::{AtomicPressureEquation, AtomicSolver};
use speedy2d::Window;
use speedy2d::window::{WindowCreationOptions, WindowPosition};

mod gui{
  pub mod gui;
  pub mod history;
  pub mod video;
}
mod simulation;
mod sph;
mod datastructure;
mod utils;
mod gpu_version{
  pub mod gpu;
  pub mod buffers;
  pub mod kernels;
}
// switch default allocator
use mimalloc::MiMalloc;

use crate::gui::history::History;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static WINDOW_SIZE:[AtomicU32;2] = [AtomicU32::new(1280), AtomicU32::new(800)];
static SIM_FPS:AtomicF64 = AtomicF64::new(60.0);
const FPS_SMOOTING:f64 = 0.99;
const VELOCITY_EPSILON:f64 = 0.00001;

// SIMULATION RELATED CONSTANTS AND ATOMICS
lazy_static! {
  pub static ref THREADS:usize = available_parallelism().unwrap().get();
  static ref BOUNDARY:[DVec2;2] = [DVec2::new(-5.0,-2.0), DVec2::new(5.0,2.0)];
  static ref FLUID:[DVec2;2] = [DVec2::new(-5.0+H*2.,-2.0+H*2.), DVec2::new(3.0-H,-1.2)];
  static ref HARD_BOUNDARY:[Float2;2] = [
    Float2::new(BOUNDARY[0].x as f32-(BOUNDARY_LAYER_COUNT+5) as f32*H as f32,BOUNDARY[0].y as f32-(BOUNDARY_LAYER_COUNT+5) as f32*H as f32), 
    Float2::new(BOUNDARY[1].x as f32+(BOUNDARY_LAYER_COUNT+5) as f32*H as f32,BOUNDARY[1].y as f32+(BOUNDARY_LAYER_COUNT+5) as f32*H as f32)
  ];
  pub static ref HISTORY:Arc<RwLock<History>> = Arc::new(RwLock::new(History::default()));
  pub static ref SOLVER:AtomicSolver = AtomicSolver::new(simulation::Solver::SESPH);
  pub static ref RESORT_ATTRIBUTES_EVERY_N:Arc<RwLock<u32>> = Arc::new(RwLock::new(4));
  pub static ref BDY_MIN:Float2 = HARD_BOUNDARY[0] - Float2::new(
    (BOUNDARY_LAYER_COUNT+20) as f32*H as f32,
    (BOUNDARY_LAYER_COUNT+20) as f32*H as f32, 
  );
}

// datastructure settings
static GRID_CURVE:AtomicGridCurve = AtomicGridCurve::new(GridCurve::Morton);

// gpu settings
const WARP:usize = 256;
const WORKGROUP_SIZE:usize = 256;

/// The gravitational constant
static GRAVITY:AtomicF64 = AtomicF64::new(-9.807);
/// Particle spacing
// const H:f64 = 0.002234;
const H:f64 = 0.03;
const BOUNDARY_LAYER_COUNT:usize = 4;
// -> Consequence of kernel support radius 2H:
const KERNEL_SUPPORT:f64 = 2.0*H;
/// The factor of the maximum size of a time step taken each iteration
static LAMBDA:AtomicF64 = AtomicF64::new(0.1);
static MAX_DT:AtomicF64 = AtomicF64::new(0.001);
static INITIAL_DT:AtomicF64 = AtomicF64::new(0.00001);
/// Mass of a particle
const M:f64 = H*H;
/// Rest density of the fluid
static RHO_ZERO:AtomicF64 = AtomicF64::new(M/(H*H));
/// Stiffness constant determining the incompressibility in the state equation
static K:AtomicF64 = AtomicF64::new(2_200.0);
/// The maximum acceptable absolute density deviation in iterative SESPH with splitting
static MAX_RHO_DEVIATION:AtomicF64 = AtomicF64::new(0.05);
/// The type of equation relating density to pressure (stress to strain)
static PRESSURE_EQ:AtomicPressureEquation = AtomicPressureEquation::new(simulation::PressureEquation::ClampedRelative);
/// Viscosity constant Nu
static NU:AtomicF64 = AtomicF64::new(0.3);

const VIDEO_SIZE:(usize, usize) = (2048, 1152);
const VIDEO_HEIGHT_WORLD:f32 = 10.1f32;
const FRAME_TIME:f32 = 1./60.;
// const FRAME_TIME:f32 = 0.001;

/// Get the current timestamp in microseconds
fn timestamp()->u128{SystemTime::now().duration_since(UNIX_EPOCH).expect("Error getting current time").as_micros()}
fn micros_to_seconds(timespan: u128)->f64{0.000_001*timespan as f64}
fn seconds_to_micros(timespan: f64)->u128{(1_000_000.0*timespan).round() as u128}

// ENTRY POINT
fn main() {
  let window = Window::new_with_options(
    "Stoked 2D", 
    WindowCreationOptions::new_windowed(
      speedy2d::window::WindowSize::PhysicalPixels((WINDOW_SIZE[0].load(Relaxed), WINDOW_SIZE[1].load(Relaxed)).into()), 
      Some(WindowPosition::Center)
    ).with_maximized(true)
  ).unwrap();

  thread::spawn(||{
    if cfg!(feature = "gpu"){
      while gpu_version::gpu::run(None) {}
    } else {
      while simulation::run() {}
    }
  });
  window.run_loop(egui_speedy2d::WindowWrapper::new(StokedWindowHandler{}));

  // while gpu_version::gpu::run(Some(10.0)) {}
  
}

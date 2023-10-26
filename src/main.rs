use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, atomic::AtomicU32};
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::thread::{self, available_parallelism};
use std::time::{SystemTime, UNIX_EPOCH};
use atomic_float::{AtomicF32, AtomicF64};
use glam::DVec2;
use lazy_static::lazy_static;
use egui_speedy2d::egui::mutex::RwLock;
use speedy2d::Window;

mod gui;
use gui::StokedWindowHandler;
mod simulation;
mod sph;
mod datastructure;
type History = Arc<RwLock<Vec<(Vec<DVec2>, f64)>>>;

static WINDOW_SIZE:[AtomicU32;2] = [AtomicU32::new(1280), AtomicU32::new(720)];
static SIM_FPS:AtomicF64 = AtomicF64::new(60.0);
const FPS_SMOOTING:f64 = 0.8;

// SIMULATION RELATED CONSTANTS AND ATOMICS
lazy_static! {
  pub static ref THREADS:usize = available_parallelism().unwrap().get();
  pub static ref REQUEST_RESTART:Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
  static ref BOUNDARY:[DVec2;2] = [DVec2::new(-10.0,-10.0), DVec2::new(10.0,10.0)];
  static ref FLUID:[DVec2;2] = [DVec2::new(-5.0,-5.0), DVec2::new(5.0,5.0)];
  pub static ref HISTORY:History = Arc::new(RwLock::new(vec![(vec![], 0.0)]));
  pub static ref COLOUR:Arc<RwLock<Vec<f64>>> = Arc::new(RwLock::new(vec![]));
}
static SIMULATION_THROTTLE_MICROS:AtomicU64 = AtomicU64::new(50);
/// The gravitational constant
static GRAVITY:AtomicF64 = AtomicF64::new(-9.807);
/// Particle spacing
const H:f64 = 0.2;
// -> Consequence of kernel support radius 2H:
const GRIDSIZE:f64 = 2.0*H;
/// The factor of the maximum size of a time step taken each iteration
static LAMBDA:AtomicF64 = AtomicF64::new(0.05);
const DEFAULT_DT:f64 = 0.01;
/// Mass of a particle
const M:f64 = H*H;
/// Rest density of the fluid
static RHO_ZERO:AtomicF64 = AtomicF64::new(M/(H*H));
/// Stiffness constant determining the incompressibility in the state equation
static K:AtomicF64 = AtomicF64::new(1_000.0);

/// Get the current timestamp in microseconds
fn timestamp()->u128{SystemTime::now().duration_since(UNIX_EPOCH).expect("Error getting current time").as_micros()}
fn micros_to_seconds(timespan: u128)->f64{0.000_001*timespan as f64}

// ENTRY POINT
fn main() {
  let window = Window::new_centered(
    "Stoked2D", 
    (WINDOW_SIZE[0].load(Relaxed), WINDOW_SIZE[1].load(Relaxed))
  ).unwrap();

  let _worker = thread::spawn(simulation::run);
  window.run_loop(egui_speedy2d::WindowWrapper::new(StokedWindowHandler{}))
}

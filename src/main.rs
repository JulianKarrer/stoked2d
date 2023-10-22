use std::sync::atomic::AtomicBool;
use std::sync::{Arc, atomic::AtomicU32};
use std::sync::atomic::Ordering::Relaxed;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use atomic_float::{AtomicF32, AtomicF64};
use glam::DVec2;
use lazy_static::lazy_static;
use egui_speedy2d::egui::mutex::RwLock;
use speedy2d::Window;

mod gui;
use gui::StokedWindowHandler;
mod simulation;

static WINDOW_SIZE:[AtomicU32;2] = [AtomicU32::new(1280), AtomicU32::new(720)];
static SIM_FPS:AtomicF64 = AtomicF64::new(60.0);
const FPS_SMOOTING:f64 = 0.95;
static REQUEST_RESTART:AtomicBool = AtomicBool::new(true);

// SIMULATION RELATED CONSTANTS AND ATOMICS
lazy_static! {
  static ref BOUNDARY:[DVec2;2] = [DVec2::new(-10.0,-10.0), DVec2::new(10.0,10.0)];
  pub static ref POSITIONS:Arc<RwLock<Vec<DVec2>>> = Arc::new(RwLock::new(vec![]));
  pub static ref COLOUR:Arc<RwLock<Vec<f64>>> = Arc::new(RwLock::new(vec![]));
  // static ref VELOCITIES:Arc<RwLock<Vec<DVec2>>> = Arc::new(RwLock::new(vec![]));
  // static ref ACCELERATIONS:Arc<RwLock<Vec<DVec2>>> = Arc::new(RwLock::new(vec![]));
}
static GRAVITY:AtomicF64 = AtomicF64::new(-9.807);

fn timestamp()->f64{SystemTime::now().duration_since(UNIX_EPOCH).expect("Error getting current time").as_secs_f64()}

// ENTRY POINT
fn main() {
  let window = Window::new_centered(
    "Stoked2D", 
    (WINDOW_SIZE[0].load(Relaxed), WINDOW_SIZE[1].load(Relaxed))
  ).unwrap();

  let _worker = thread::spawn(simulation::run);
  window.run_loop(egui_speedy2d::WindowWrapper::new(StokedWindowHandler{}))
}

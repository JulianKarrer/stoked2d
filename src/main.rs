#![feature(test)]
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, atomic::AtomicU32};
use std::sync::atomic::Ordering::Relaxed;
use std::thread::{self, available_parallelism};
use std::time::{SystemTime, UNIX_EPOCH};
use atomic_float::{AtomicF32, AtomicF64};
use datastructure::{GridCurve, AtomicGridCurve};
use glam::DVec2;
use lazy_static::lazy_static;
use egui_speedy2d::egui::mutex::RwLock;
use simulation::{AtomicPressureEquation, AtomicSolver};
use speedy2d::Window;
use speedy2d::window::{WindowCreationOptions, WindowPosition};

mod gui;
use gui::StokedWindowHandler;
mod simulation;
mod sph;
mod datastructure;

// switch default allocator
use mimalloc::MiMalloc;
use crate::gui::History;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static WINDOW_SIZE:[AtomicU32;2] = [AtomicU32::new(1280), AtomicU32::new(720)];
static SIM_FPS:AtomicF64 = AtomicF64::new(60.0);
const FPS_SMOOTING:f64 = 0.95;

// SIMULATION RELATED CONSTANTS AND ATOMICS
lazy_static! {
  pub static ref THREADS:usize = available_parallelism().unwrap().get();
  pub static ref REQUEST_RESTART:Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
  static ref BOUNDARY:[DVec2;2] = [DVec2::new(-10.0,-10.0), DVec2::new(10.0,10.0)];
  static ref FLUID:[DVec2;2] = [DVec2::new(-5.0,-5.0), DVec2::new(5.0,5.0)];
  pub static ref HISTORY:Arc<RwLock<History>> = Arc::new(RwLock::new(History::default()));
  pub static ref SOLVER:AtomicSolver = AtomicSolver::new(simulation::Solver::SESPH);
}


static SIMULATION_THROTTLE_MICROS:AtomicU64 = AtomicU64::new(1);

// datastructure settings
static RESORT_ATTRIBUTES_EVERY_N:AtomicU32 = AtomicU32::new(16);
static GRID_CURVE:AtomicGridCurve = AtomicGridCurve::new(GridCurve::Morton);

/// The gravitational constant
static GRAVITY:AtomicF64 = AtomicF64::new(-9.807);
/// Particle spacing
const H:f64 = 0.2;
// -> Consequence of kernel support radius 2H:
const KERNEL_SUPPORT:f64 = 2.0*H;
/// The factor of the maximum size of a time step taken each iteration
static LAMBDA:AtomicF64 = AtomicF64::new(0.2);
static MAX_DT:AtomicF64 = AtomicF64::new(0.1);
/// Mass of a particle
const M:f64 = H*H;
/// Rest density of the fluid
static RHO_ZERO:AtomicF64 = AtomicF64::new(M/(H*H));
/// Stiffness constant determining the incompressibility in the state equation
static K:AtomicF64 = AtomicF64::new(1_500.0);
/// The type of equation relating density to pressure (stress to strain)
static PRESSURE_EQ:AtomicPressureEquation = AtomicPressureEquation::new(simulation::PressureEquation::ClampedRelative);
/// Viscosity constant Nu
static NU:AtomicF64 = AtomicF64::new(0.3);

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

  thread::spawn(||{loop {simulation::run()}});
  window.run_loop(egui_speedy2d::WindowWrapper::new(StokedWindowHandler{}));
}

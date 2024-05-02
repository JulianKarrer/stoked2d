#![feature(test)]
#![feature(iter_collect_into)]
#![feature(const_fn_floating_point_arithmetic)]
use atomic_float::{AtomicF32, AtomicF64};
use datastructure::{AtomicGridCurve, GridCurve};
use egui_speedy2d::egui::mutex::RwLock;
use glam::DVec2;
use lazy_static::lazy_static;
use ocl::prm::Float2;
use simulation::{AtomicPressureEquation, AtomicSolver};
use sph::SphKernel;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{atomic::AtomicU32, Arc};
use std::thread::{self, available_parallelism};
use std::time::{SystemTime, UNIX_EPOCH};
pub mod gui {
    pub mod gui;
    pub mod history;
    pub mod plot;
    pub mod video;
}
pub mod datastructure;
pub mod simulation;
pub mod sph;
pub mod utils;
pub mod gpu_version {
    pub mod buffers;
    pub mod gpu;
    pub mod kernels;
}
// switch default allocator
use mimalloc::MiMalloc;

use crate::gui::history::History;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub static WINDOW_SIZE: [AtomicU32; 2] = [AtomicU32::new(1280), AtomicU32::new(800)];
static SIM_FPS: AtomicF64 = AtomicF64::new(60.0);
const FPS_SMOOTING: f64 = 0.99;
const VELOCITY_EPSILON: f64 = 0.00001;

// SIMULATION RELATED CONSTANTS AND ATOMICS
lazy_static! {
    static ref BOUNDARY: [DVec2; 2] = [DVec2::new(-3.0, -3.0), DVec2::new(3.0, 3.0)];
    static ref FLUID: [DVec2; 2] = [
        DVec2::new(-3.0 + H * 1., -3.0 + H * 1.),
        DVec2::new(0. - H, -0.)
    ];
    static ref HARD_BOUNDARY: [Float2; 2] = [
        Float2::new(
            BOUNDARY[0].x as f32 - (BOUNDARY_LAYER_COUNT + 10) as f32 * H as f32,
            BOUNDARY[0].y as f32 - (BOUNDARY_LAYER_COUNT + 10) as f32 * H as f32
        ),
        Float2::new(
            BOUNDARY[1].x as f32 + (BOUNDARY_LAYER_COUNT + 10) as f32 * H as f32,
            BOUNDARY[1].y as f32 + (BOUNDARY_LAYER_COUNT + 10) as f32 * H as f32
        )
    ];
    static ref HISTORY: Arc<RwLock<History>> = Arc::new(RwLock::new(History::default()));
    static ref SOLVER: AtomicSolver = AtomicSolver::new(simulation::Solver::SESPH);
    static ref SPH_KERNELS: Arc<RwLock<SphKernel>> = Arc::new(RwLock::new(SphKernel::default()));
    static ref RESORT_ATTRIBUTES_EVERY_N: Arc<RwLock<u32>> = Arc::new(RwLock::new(4));
    static ref BDY_MIN: Float2 = HARD_BOUNDARY[0]
        - Float2::new(
            (BOUNDARY_LAYER_COUNT + 20) as f32 * H as f32,
            (BOUNDARY_LAYER_COUNT + 20) as f32 * H as f32,
        );
    static ref THREADS: usize = available_parallelism().unwrap().get();
}

/// The gravitational constant
static GRAVITY: AtomicF64 = AtomicF64::new(-9.807);
/// Particle spacing
pub const H: f64 = 0.05;
const BOUNDARY_LAYER_COUNT: usize = 3;
const USE_GPU_BOUNDARY: bool = true;

/// Viscosity constant Nu
static NU: AtomicF64 = AtomicF64::new(0.03);
/// Stiffness constant determining the incompressibility in the state equation
static K: AtomicF64 = AtomicF64::new(2300.0);

/// The factor of the maximum size of a time step taken each iteration
static LAMBDA: AtomicF64 = AtomicF64::new(0.1);
static MAX_DT: AtomicF64 = AtomicF64::new(0.001);
static INITIAL_DT: AtomicF64 = AtomicF64::new(0.0001);
/// Rest density of the fluid
static RHO_ZERO: AtomicF64 = AtomicF64::new(M / (H * H));
/// The maximum acceptable absolute density deviation in iterative SESPH with splitting
static MAX_RHO_DEVIATION: AtomicF64 = AtomicF64::new(0.05);
/// The type of equation relating density to pressure (stress to strain)
static PRESSURE_EQ: AtomicPressureEquation =
    AtomicPressureEquation::new(simulation::PressureEquation::ClampedRelative);
/// Mass of a particle
const M: f64 = H * H;
// -> Consequence of kernel support radius 2H:
pub const KERNEL_SUPPORT: f64 = 2.0 * H;

// datastructure settings
static GRID_CURVE: AtomicGridCurve = AtomicGridCurve::new(GridCurve::XYZ);

// gpu settings
const WARP: usize = 256;
const WORKGROUP_SIZE: usize = 256;

// video settings
const VIDEO_SIZE: (usize, usize) = (1280, 800);
const VIDEO_HEIGHT_WORLD: f32 = 10.1f32;
const FRAME_TIME: f32 = 1. / 60.;
// const FRAME_TIME:f32 = 0.001;

/// Get the current timestamp in microseconds
fn timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Error getting current time")
        .as_micros()
}
fn micros_to_seconds(timespan: u128) -> f64 {
    0.000_001 * timespan as f64
}
fn seconds_to_micros(timespan: f64) -> u128 {
    (1_000_000.0 * timespan).round() as u128
}
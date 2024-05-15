use glam::DVec2;
use ocl::prm::{Float, Float2};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    attributes::Attributes,
    datastructure::Grid,
    gpu_version::gpu::len_float2,
    utils::{average_val, hamiltonian},
    BOUNDARY, H, M,
};

/// Represents the features of the simulation that are stored for visualization at any given time step
#[derive(Default)]
pub struct HistoryTimestep {
    pub pos: Vec<[f64; 2]>,
    pub velocities: Vec<f64>,
    pub current_t: f64,
    pub densities: Vec<f64>,
    pub grid_handle_index: Vec<u32>,
}

/// Contains the features of the simulation that are stored for visualization.
pub struct History {
    pub steps: Vec<HistoryTimestep>,
    pub plot_density: Vec<[f64; 2]>,
    pub plot_hamiltonian: Vec<[f64; 2]>,
    pub bdy: Vec<[f64; 2]>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            steps: vec![HistoryTimestep::default()],
            plot_density: vec![[0.0, M / (H * H)]],
            plot_hamiltonian: vec![],
            bdy: vec![],
        }
    }
}

// general implementations
impl History {
    fn reset(&mut self) {
        self.bdy.clear();
        self.steps.clear();
        self.steps.shrink_to_fit();
        self.plot_density.clear();
        self.plot_density.shrink_to_fit();
        self.plot_hamiltonian.clear();
        self.plot_hamiltonian.shrink_to_fit();
    }
}

// CPU implementations
impl History {
    pub fn reset_and_add(
        &mut self,
        state: &Attributes,
        grid: &Grid,
        bdy: &[DVec2],
        current_t: f64,
    ) {
        self.reset();
        self.bdy = bdy.par_iter().map(|p| p.to_array()).collect();
        self.add(state, grid, current_t);
    }

    pub fn add_step(&mut self, state: &Attributes, grid: &Grid, current_t: f64) {
        self.add(state, grid, current_t)
    }

    fn add(&mut self, state: &Attributes, grid: &Grid, current_t: f64) {
        let densities = state.den.clone();
        let average_density = average_val(&densities);
        self.plot_density.push([current_t, average_density]);
        self.plot_hamiltonian.push([
            current_t,
            hamiltonian(&state.pos, &state.vel, M, BOUNDARY[0].y),
        ]);
        self.steps.push(HistoryTimestep {
            pos: state.pos.par_iter().map(|p| p.to_array()).collect(),
            current_t,
            densities,
            velocities: state.vel.par_iter().map(|v| v.length()).collect(),
            grid_handle_index: grid.handles.par_iter().map(|x| x.index as u32).collect(),
        })
    }

    pub fn add_plot_data_only(&mut self, state: &Attributes, current_t: f64) {
        let densities = state.den.clone();
        let average_density = average_val(&densities);
        self.plot_density.push([current_t, average_density]);
        self.plot_hamiltonian.push([
            current_t,
            hamiltonian(&state.pos, &state.vel, M, BOUNDARY[0].y),
        ]);
    }
}

// GPU implementations
impl History {
    pub fn gpu_reset_and_add(
        &mut self,
        pos: &[Float2],
        vel: &[Float2],
        bdy: &[Float2],
        handle_indices: &[u32],
        den: &[Float],
        current_t: f64,
    ) {
        self.reset();
        self.bdy = bdy
            .par_iter()
            .map(|pos| [pos[0] as f64, pos[1] as f64])
            .collect();
        self.gpu_add_step(pos, vel, handle_indices, den, current_t);
    }

    pub fn gpu_add_step(
        &mut self,
        pos: &[Float2],
        vel: &[Float2],
        handle_indices: &[u32],
        den: &[Float],
        current_t: f64,
    ) {
        self.plot_density.push([
            current_t,
            den.par_iter().map(|f| f[0]).sum::<f32>() as f64 / den.len() as f64,
        ]);
        self.steps.push(HistoryTimestep {
            pos: pos.par_iter().map(|p| [p[0] as f64, p[1] as f64]).collect(),
            current_t,
            densities: den.par_iter().map(|d| d[0] as f64).collect(),
            velocities: vel.par_iter().map(|v| len_float2(v)).collect(),
            grid_handle_index: handle_indices.to_vec(),
        })
    }
}

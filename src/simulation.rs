use crate::{datastructure::Grid, sph::KernelType, *};
use atomic_enum::atomic_enum;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::{fmt::Debug, time::Duration};

use self::{
    attributes::Attributes,
    boundary::Boundary,
    gui::gui::{REQUEST_RESTART, SIMULATION_TOGGLE},
};

// MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn run(run_for_t: Option<f64>) -> bool {
    let boundary = Boundary::from_image("setting3.png", 0.01, &{ *SPH_KERNELS.read() }.density);
    let mut state = Attributes::from_image("setting3.png", 0.01, &boundary);
    // state.resort(&grid);
    let mut current_t = 0.0;
    let mut since_resort = 0;
    // reset history and add first timestep
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        &boundary,
        &KernelType::GaussSpline3,
    );
    {
        HISTORY
            .write()
            .reset_and_add(&state, &state.grid, &boundary.pos, current_t);
    }
    let mut last_update_time = timestamp();
    let mut last_gui_update_t = 0.0f64;
    while !*REQUEST_RESTART.read() {
        // wait if requested
        while *SIMULATION_TOGGLE.read() {
            thread::sleep(Duration::from_millis(10));
        }
        // update the datastructure and potentially resort particle attributes
        if since_resort > { *RESORT_ATTRIBUTES_EVERY_N.read() } {
            state.resort();
            since_resort = 0;
        } else {
            since_resort += 1;
        }
        state.grid.update_grid(&state.pos, KERNEL_SUPPORT);

        // perform an update step using the selected fluid solver
        let kernels = { *SPH_KERNELS.read() };
        SOLVER
            .load(Relaxed)
            .solve(&mut state, &mut current_t, &boundary, &kernels);
        enforce_boundary_conditions(&mut state.pos, &mut state.vel, &mut state.acc);

        // write back the positions to the global buffer for visualization and update the FPS count
        update_fps(&mut last_update_time);
        if current_t - last_gui_update_t > FRAME_TIME.into() {
            last_gui_update_t = current_t;
            {
                HISTORY.write().add_step(&state, &state.grid, current_t);
            }
        } else {
            {
                HISTORY.write().add_plot_data_only(&state, current_t);
            }
        }
        // if only a specific time frame was requested, stop the simulation
        if let Some(max_t) = run_for_t {
            if max_t <= current_t {
                return false;
            }
        }
    }
    *REQUEST_RESTART.write() = false;
    true
}

// SOLVERS AVAILABLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Perform a simulation update step using the basic SESPH solver
fn sesph(state: &mut Attributes, current_t: &mut f64, boundary: &Boundary, knls: &SphKernel) {
    // update densities and pressures
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
    update_pressures(&state.den, &mut state.prs);
    // apply external forces
    apply_gravity_and_viscosity(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        &knls.viscosity,
    );
    // apply pressure forces
    add_pressure_accelerations(
        &state.pos,
        &state.den,
        &state.prs,
        &state.mas,
        &mut state.acc,
        &state.grid,
        boundary,
        &knls.pressure,
    );
    // perform a time step
    let dt = update_dt(&state.vel, current_t);
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}

// FUNCTIONS USED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Determines the size of the next time step in seconds using the maximum velocity of a particle
/// in the previous step, the particle spacing H and a factor LAMBDA, updating the current time and
/// returning the dt for numerical time integration
///
/// This correpsponds to the Courant-Friedrichs-Lewy condition
fn update_dt(vel: &[DVec2], current_t: &mut f64) -> f64 {
    let v_max = vel
        .par_iter()
        .map(|v| v.length())
        .reduce_with(|a, b| a.max(b))
        .unwrap();
    let mut dt = (LAMBDA.load(Relaxed) * H / v_max).min(MAX_DT.load(Relaxed));
    if v_max < VELOCITY_EPSILON || !dt.is_normal() {
        dt = INITIAL_DT.load(Relaxed)
    }
    *current_t += dt;
    dt
}

/// Update the FPS counter based on the previous iterations timestamp
pub fn update_fps(previous_timestamp: &mut u128) {
    let now = timestamp();
    SIM_FPS
        .fetch_update(Relaxed, Relaxed, |fps| {
            Some(
                fps * FPS_SMOOTING
                    + 1.0 / micros_to_seconds(now - *previous_timestamp) * (1.0 - FPS_SMOOTING),
            )
        })
        .unwrap();
    *previous_timestamp = now;
}

/// Apply external forces such as gravity and viscosity, overwriting the accelerations vec
fn apply_gravity_and_viscosity(
    pos: &[DVec2],
    vel: &[DVec2],
    acc: &mut [DVec2],
    den: &[f64],
    mas: &[f64],
    grid: &Grid,
    knl: &KernelType,
) {
    // account for gravity
    let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
    // calculate viscosity
    let nu = NU.load(Relaxed);
    acc.par_iter_mut()
        .enumerate()
        .zip(pos)
        .zip(vel)
        .for_each(|(((i, a), p), v)| {
            let vis: DVec2 = nu
                * 2.0
                * grid
                    .query_index(i)
                    .iter()
                    .map(|j| {
                        let x_i_j = *p - pos[*j];
                        let v_i_j = *v - vel[*j];
                        mas[*j] / den[*j] * (v_i_j).dot(x_i_j)
                            / (x_i_j.length_squared() + 0.01 * H * H)
                            * knl.dw(p, &pos[*j])
                    })
                    .reduce(|a, b| a + b)
                    .unwrap_or(DVec2::ZERO);
            assert!(vis.is_finite());
            *a = acc_g + vis
        });
}

/// Update the densities at each particle
fn update_densities(
    pos: &[DVec2],
    den: &mut [f64],
    mas: &[f64],
    grid: &Grid,
    boundary: &Boundary,
    knl: &KernelType,
) {
    pos.par_iter()
        .enumerate()
        .zip(den)
        .for_each(|((i, x_i), rho_i)| {
            *rho_i = grid
                .query_index(i)
                .iter()
                .map(|j| knl.w(x_i, &pos[*j]) * mas[i])
                .sum::<f64>()
                + boundary
                    .grid
                    .query_radius(x_i, &boundary.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|j| knl.w(x_i, &boundary.pos[*j]) * &boundary.m[*j])
                    .sum::<f64>();
        });
}

/// Update the pressures at each particle using densities
fn update_pressures(den: &[f64], prs: &mut [f64]) {
    let k = K.load(Relaxed);
    let rho_0 = RHO_ZERO.load(Relaxed);
    let eq = PRESSURE_EQ.load(Relaxed);
    prs.par_iter_mut().zip(den).for_each(|(p_i, rho_i)| {
        *p_i = match eq {
            PressureEquation::Absolute => k * (rho_i - rho_0),
            PressureEquation::Relative => k * (rho_i / rho_0 - 1.0),
            PressureEquation::ClampedRelative => (k * (rho_i / rho_0 - 1.0)).max(0.0),
            PressureEquation::Compressible => k * ((rho_i / rho_0).powi(7) - 1.0),
            PressureEquation::ClampedCompressible => (k * ((rho_i / rho_0).powi(7) - 1.0)).max(0.0),
        }
    });
}

/// Compute pressure accelerations from the momentum-preserving SPH approximation of the density gradient,
/// adding the result to the current accelerations
fn add_pressure_accelerations(
    pos: &[DVec2],
    den: &[f64],
    prs: &[f64],
    mas: &[f64],
    acc: &mut [DVec2],
    grid: &Grid,
    boundary: &Boundary,
    knl: &KernelType,
) {
    let rho_0 = RHO_ZERO.load(Relaxed);
    let one_over_rho_0_squared = 1.0 / (rho_0 * rho_0);
    let gamma_2 = GAMMA_2.load(Relaxed);
    pos.par_iter()
        .enumerate()
        .zip(prs)
        .zip(den)
        .zip(acc)
        .for_each(|((((i, x_i), p_i), rho_i), acc)| {
            let p_i_over_rho_i_squared = p_i / (rho_i * rho_i);
            *acc += -mas[i]
                * grid
                    .query_index(i)
                    .iter()
                    .map(|j| {
                        (p_i_over_rho_i_squared + prs[*j] / (den[*j] * den[*j]))
                            * knl.dw(x_i, &pos[*j])
                    })
                    .sum::<DVec2>()
                - gamma_2
                    * (p_i_over_rho_i_squared + p_i * one_over_rho_0_squared)
                    * boundary
                        .grid
                        .query_radius(x_i, &boundary.pos, KERNEL_SUPPORT)
                        .iter()
                        .map(|j| knl.dw(x_i, &boundary.pos[*j]) * boundary.m[*j])
                        .sum::<DVec2>();
        })
}

/// Perform a numerical time integration step using the symplectic Euler or Eurler-Cromer integration scheme.
fn time_step_euler_cromer(pos: &mut [DVec2], vel: &mut [DVec2], acc: &[DVec2], dt: f64) {
    pos.par_iter_mut()
        .zip(vel)
        .zip(acc)
        .for_each(|((p, v), a)| {
            *v += *a * dt;
            *p += *v * dt;
        })
}

/// Harshly enforce rudimentary boundary conditions by simply setting velocity and acceleration of
/// particles penetrating the boundary to zero while adjusting ther positions in a direction
/// orthogonal to the boundary to be within bounds.
fn enforce_boundary_conditions(pos: &mut [DVec2], vel: &mut [DVec2], acc: &mut [DVec2]) {
    let hbdy = { *HARD_BOUNDARY.read() };
    pos.par_iter_mut()
        .zip(vel)
        .zip(acc)
        .for_each(|((p, v), a)| {
            if p.x < hbdy[0][0] as f64 {
                p.x = hbdy[0][0] as f64;
                a.x = 0.0;
                v.x = 0.0;
            }
            if p.y < hbdy[0][1] as f64 {
                p.y = hbdy[0][1] as f64;
                a.y = 0.0;
                v.y = 0.0;
            }
            if p.x > hbdy[1][0] as f64 {
                p.x = hbdy[1][0] as f64;
                a.x = 0.0;
                v.x = 0.0;
            }
            if p.y > hbdy[1][1] as f64 {
                p.y = hbdy[1][1] as f64;
                a.y = 0.0;
                v.y = 0.0;
            }
        })
}

// STRUCTURE DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[atomic_enum]
#[derive(PartialEq)]
/// A fluid solver, implementing a single simulation step of some SPH method.
pub enum Solver {
    SESPH,
}

impl Solver {
    /// Perform a single simulation step using some SPH method.
    ///
    /// The solver can rely on current particle positions and velocities to be accurate.
    /// Accelerations, densities and pressures etc. are not resorted and must be assumed to be
    /// uninitialized garbage.
    ///
    /// The grid can also be assumed to be accurate.
    fn solve(
        &self,
        state: &mut Attributes,
        current_t: &mut f64,
        boundary: &Boundary,
        knls: &SphKernel,
    ) {
        match self {
            Solver::SESPH => sesph(state, current_t, boundary, knls),
        }
    }
}

#[atomic_enum]
#[derive(PartialEq)]
/// An equation relating measured density deviation to pressure, or stress to strain.
/// While Clamped equations produce only pressures > 0 counteracting compression, the respective
/// unclamped version can lead to negative pressures, creating attraction between particles in
/// low density regions. The global static stiffness K may factor into these equations.
pub enum PressureEquation {
    Absolute,
    Relative,
    ClampedRelative,
    Compressible,
    ClampedCompressible,
}

impl std::fmt::Display for PressureEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PressureEquation::Absolute => "k·(ρᵢ-ρ₀)",
                PressureEquation::Relative => "k·(ρᵢ/ρ₀-1)",
                PressureEquation::ClampedRelative => "k·(ρᵢ/ρ₀-1).max(0)",
                PressureEquation::Compressible => "k·((ρᵢ/ρ₀)⁷-1)",
                PressureEquation::ClampedCompressible => "k·((ρᵢ/ρ₀)⁷-1)",
            }
        )
    }
}

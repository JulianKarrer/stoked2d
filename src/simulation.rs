use crate::{datastructure::Grid, sph::KernelType, *};
use atomic_enum::atomic_enum;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::{fmt::Debug, time::Duration};
use strum_macros::EnumIter;
use utils::create_progressbar;

use self::{
    attributes::Attributes,
    boundary::Boundary,
    gui::gui::{REQUEST_RESTART, SIMULATION_TOGGLE},
};

// MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn run(run_for_t: Option<f32>, path: &str) -> bool {
    let boundary = Boundary::from_image(path, 0.01, &{ *SPH_KERNELS.read() }.density);
    let mut state = Attributes::from_image(path, 0.01, &boundary);
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
            .reset_and_add(&state, &state.grid, &boundary, current_t);
    }
    // set up progress bar
    let progressbar = create_progressbar(run_for_t);
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
        if current_t - last_gui_update_t > HISTORY_FRAME_TIME.into() {
            // update the progress bar
            if let Some(ref bar) = progressbar {
                bar.inc(1);
                bar.set_message(format!("{:.3} ITERS/S", SIM_FPS.load(Relaxed)));
            }
            // update the gui
            last_gui_update_t = current_t;
            {
                HISTORY
                    .write()
                    .add_step(&state, &state.grid, current_t, &boundary);
            }
        } else {
            {
                HISTORY
                    .write()
                    .add_plot_data_only(&state, current_t, &boundary);
            }
        }
        // if only a specific time frame was requested, stop the simulation
        if let Some(max_t) = run_for_t {
            if max_t as f64 <= current_t {
                progressbar.unwrap().finish();
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
    // apply external forces
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.viscosity,
    );
    update_pressures(&state.den, &mut state.prs);
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

/// Perform a simulation update step using the SESPH solver with splitting.
/// Non-pressure forces are integrated first, then densities are predicted
/// based on the predicted velocities
fn sesph_splitting(
    state: &mut Attributes,
    current_t: &mut f64,
    boundary: &Boundary,
    knls: &SphKernel,
) {
    // apply external forces
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.viscosity,
    );
    let dt = update_dt(&state.vel, current_t);
    // predict velocities
    state
        .vel
        .par_iter_mut()
        .zip(&state.acc)
        .for_each(|(vel_i, acc_i)| *vel_i += dt * (*acc_i));
    // predict densities from predicted velocities
    predict_densities(
        dt,
        &state.vel,
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
    update_pressures(&state.den, &mut state.prs);
    // apply pressure forces
    state
        .acc
        .par_iter_mut()
        .for_each(|acc_i| *acc_i = DVec2::ZERO);
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
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
    // update densities at the end to make sure the gui output matches reality
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
}

/// Perform a simulation update step using an iterative SESPH solver with splitting.
/// Non-pressure forces are integrated first, then densities are predicted
/// based on the predicted velocities. This is iterated, refining the predicted velocities.
fn sesph_iter(state: &mut Attributes, current_t: &mut f64, boundary: &Boundary, knls: &SphKernel) {
    // apply external forces
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.viscosity,
    );
    let dt = update_dt(&state.vel, current_t);
    for _ in 0..5 {
        // predict velocities
        state
            .vel
            .par_iter_mut()
            .zip(&state.acc)
            .for_each(|(vel_i, acc_i)| *vel_i += dt * (*acc_i));
        // predict densities from predicted velocities
        predict_densities(
            dt,
            &state.vel,
            &state.pos,
            &mut state.den,
            &state.mas,
            &state.grid,
            boundary,
            &knls.density,
        );
        update_pressures(&state.den, &mut state.prs);
        // apply pressure forces
        state
            .acc
            .par_iter_mut()
            .for_each(|acc_i| *acc_i = DVec2::ZERO);
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
    }
    // perform a time step
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
    // update densities at the end to make sure the gui output matches reality
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
}

fn iisph(state: &mut Attributes, current_t: &mut f64, boundary: &Boundary, knls: &SphKernel) {
    // initialization
    // compute densities
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
    // apply non-pressure forces
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.viscosity,
    );
    let dt = update_dt(&state.vel, current_t);
    // predict velocities v* due to non-pressure forces
    state
        .vel
        .par_iter_mut()
        .zip(&state.acc)
        .for_each(|(vel_i, acc_i)| *vel_i += dt * (*acc_i));
    // compute source term and save it to `den`
    compute_source_term(
        &mut state.den,
        &state.mas,
        &state.pos,
        &state.vel,
        &knls.density,
        &state.grid,
        boundary,
        dt,
    );
    // compute the diagonal matrix element of Ap for the jacobi solver
    compute_diagonal_element(
        &mut state.diag,
        &state.pos,
        &state.mas,
        &state.grid,
        boundary,
        dt,
        &knls.density,
    );
    // set initial pressures to half the previous value
    state.prs.par_iter_mut().for_each(|prs| *prs = 0.5 * (*prs));
    // ITERATE
    let one_over_rho_0_2 = 1. / RHO_ZERO.load(Relaxed).powi(2);
    let mut rho_err: f64 = 0.0;
    let mut l = 0;
    while l < JACOBI_MIN_ITER.load(Relaxed)
        || rho_err.max(0.0) > MAX_RHO_DEVIATION.load(Relaxed) && !{ *REQUEST_RESTART.read() }
    {
        // compute pressure acc
        state.acc.par_iter_mut().enumerate().for_each(|(i, acc)| {
            *acc = -state
                .grid
                .query_index(i)
                .iter()
                .map(|j| {
                    state.mas[*j]
                        * (state.prs[i] * one_over_rho_0_2 + state.prs[*j] * one_over_rho_0_2)
                        * knls.pressure.dw(&state.pos[i], &state.pos[*j])
                })
                .sum::<DVec2>()
                - boundary
                    .grid
                    .query_radius(&state.pos[i], &boundary.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|k| {
                        boundary.mas[*k]
                            * 2.
                            * state.prs[i]
                            * one_over_rho_0_2
                            * knls.pressure.dw(&state.pos[i], &boundary.pos[*k])
                    })
                    .sum::<DVec2>()
        });
        // update pressures
        let omega = OMEGA_JACOBI.load(Relaxed);
        rho_err = state
            .prs
            .par_iter_mut()
            .enumerate()
            .map(|(i, prs)| {
                let a_p = dt
                    * dt
                    * state
                        .grid
                        .query_index(i)
                        .iter()
                        .map(|j| {
                            state.mas[*j]
                                * (state.acc[i] - state.acc[*j])
                                    .dot(knls.pressure.dw(&state.pos[i], &state.pos[*j]))
                        })
                        .sum::<f64>()
                    + dt * dt
                        * boundary
                            .grid
                            .query_radius(&state.pos[i], &boundary.pos, KERNEL_SUPPORT)
                            .iter()
                            .map(|k| {
                                boundary.mas[*k]
                                    * state.acc[i]
                                        .dot(knls.pressure.dw(&state.pos[i], &boundary.pos[*k]))
                            })
                            .sum::<f64>();
                if state.diag[i].is_normal() {
                    *prs = ((1. - omega) * (*prs) + omega * (state.den[i] - a_p) / state.diag[i])
                        .max(0.0);
                }

                a_p - state.den[i]
            })
            .sum::<f64>()
            / state.pos.len() as f64;
        l += 1;
    }
    // integrate time
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
    // update density for GUI
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
}

// FUNCTIONS USED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Determines the size of the next time step in seconds using the maximum velocity of a particle
/// in the previous step, the particle spacing H and a factor LAMBDA, updating the current time and
/// returning the dt for numerical time integration
///
/// This correpsponds to the Courant-Friedrichs-Lewy condition
fn update_dt(vel: &[DVec2], current_t: &mut f64) -> f64 {
    let dt = if USE_FIXED_DT.load(Relaxed) {
        FIXED_DT.load(Relaxed)
    } else {
        let v_max = vel
            .par_iter()
            .map(|v| v.length())
            .reduce_with(|a, b| a.max(b))
            .unwrap();
        let mut dt = (LAMBDA.load(Relaxed) * H / v_max).min(MAX_DT.load(Relaxed));
        if v_max < VELOCITY_EPSILON || !dt.is_normal() {
            dt = INITIAL_DT.load(Relaxed)
        }
        dt
    };
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

/// Apply non-pressure forces such as gravity and viscosity, overwriting the accelerations
fn apply_non_pressure_forces(
    pos: &[DVec2],
    vel: &[DVec2],
    acc: &mut [DVec2],
    den: &[f64],
    mas: &[f64],
    grid: &Grid,
    boundary: &Boundary,
    knl: &KernelType,
) {
    // account for gravity
    let acc_g = DVec2::Y * GRAVITY.load(Relaxed);
    // calculate viscosity
    let nu = NU.load(Relaxed);
    let nu2 = NU_2.load(Relaxed);
    let rho_0 = RHO_ZERO.load(Relaxed);
    acc.par_iter_mut()
        .enumerate()
        .zip(pos)
        .zip(vel)
        .for_each(|(((i, a_i), x_i), v_i)| {
            // fluid viscosity
            let vis: DVec2 = nu
                * 2.*(DIMENSIONS + 2.) // 2*(dimensions + 2)
                * grid
                    .query_index(i)
                    .iter()
                    .map(|j| {
                        let x_i_j = *x_i - pos[*j];
                        let v_i_j = *v_i - vel[*j];
                        mas[*j] / den[*j] * (v_i_j).dot(x_i_j)
                            / (x_i_j.length_squared() + 0.01 * H * H)
                            * knl.dw(x_i, &pos[*j])
                    }).sum::<DVec2>();
            assert!(vis.is_finite());
            // boundary viscosity
            let bdy_neighbours = boundary
                .grid
                .query_radius(x_i, &boundary.pos, KERNEL_SUPPORT);
            let bdy_normal = bdy_neighbours
                .iter()
                .map(|j| knl.dw(x_i, &boundary.pos[*j]))
                .sum::<DVec2>()
                .normalize_or_zero();
            let vis_bdy: DVec2 = nu2
                * 2.*(DIMENSIONS + 2.) // 2*(dimensions + 2)
                * bdy_neighbours.iter()
                    .map(|j| {
                        let x_i_j = *x_i - boundary.pos[*j];
                        let v_i_j = *v_i; // v at boundary is zero
                        boundary.mas[*j] / rho_0 * (v_i_j).dot(x_i_j)
                            / (x_i_j.length_squared() + 0.01 * H * H)
                            * knl.dw(x_i, &boundary.pos[*j])
                    }).sum::<DVec2>();
            assert!(vis_bdy.is_finite());
            *a_i = acc_g + vis + vis_bdy.dot(bdy_normal) * bdy_normal
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
                .map(|j| knl.w(x_i, &pos[*j]) * mas[*j])
                .sum::<f64>()
                + boundary
                    .grid
                    .query_radius(x_i, &boundary.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|j| knl.w(x_i, &boundary.pos[*j]) * &boundary.mas[*j])
                    .sum::<f64>();
        });
}

/// Predict densities at each particle, given a set of predicted velocities
fn predict_densities(
    dt: f64,
    vel: &[DVec2],
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
        .zip(vel)
        .for_each(|(((i, x_i), rho_i), vel_i)| {
            *rho_i = grid
                .query_index(i)
                .iter()
                .map(|j| knl.w(x_i, &pos[*j]) * mas[*j])
                .sum::<f64>()
                + dt * grid
                    .query_index(i)
                    .iter()
                    .map(|j| mas[*j] * knl.dw(x_i, &pos[*j]).dot(*vel_i - vel[*j]))
                    .sum::<f64>()
                + boundary
                    .grid
                    .query_radius(x_i, &boundary.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|j| knl.w(x_i, &boundary.pos[*j]) * &boundary.mas[*j])
                    .sum::<f64>()
                + dt * boundary
                    .grid
                    .query_radius(x_i, &boundary.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|j| &boundary.mas[*j] * knl.dw(x_i, &boundary.pos[*j]).dot(*vel_i))
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
                        .map(|j| knl.dw(x_i, &boundary.pos[*j]) * boundary.mas[*j])
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

fn compute_source_term(
    den: &mut [f64],
    mas: &[f64],
    pos: &[DVec2],
    vel: &[DVec2],
    knl: &KernelType,
    grid: &Grid,
    boundary: &Boundary,
    dt: f64,
) {
    let rho_0 = RHO_ZERO.load(Relaxed);
    den.par_iter_mut().enumerate().for_each(|(i, rho_f)| {
        *rho_f = rho_0
            - (*rho_f)
            - dt * grid
                .query_index(i)
                .iter()
                .map(|j| mas[*j] * (vel[i] - vel[*j]).dot(knl.dw(&pos[i], &pos[*j])))
                .sum::<f64>()
            - dt * boundary
                .grid
                .query_radius(&pos[i], &boundary.pos, KERNEL_SUPPORT)
                .iter()
                .map(|k| {
                    boundary.mas[*k]
                        * (vel[i] - boundary.vel[*k]).dot(knl.dw(&pos[i], &boundary.pos[*k]))
                })
                .sum::<f64>()
    })
}

fn compute_diagonal_element(
    diag: &mut [f64],
    pos: &[DVec2],
    mas: &[f64],
    grid: &Grid,
    boundary: &Boundary,
    dt: f64,
    knl: &KernelType,
) {
    let rho_0 = RHO_ZERO.load(Relaxed);
    diag.par_iter_mut().enumerate().for_each(|(i, a_ff)| {
        let dt_2 = dt * dt;
        let x_i = pos[i];
        let f_f = grid.query_index(i);
        let f_b = boundary
            .grid
            .query_radius(&x_i, &boundary.pos, KERNEL_SUPPORT);

        let t1f: DVec2 = -f_f
            .iter()
            .map(|j| mas[*j] / (rho_0 * rho_0) * knl.dw(&x_i, &pos[*j]))
            .sum::<DVec2>();
        let t2f: DVec2 = -2.
            * f_b
                .iter()
                .map(|k| boundary.mas[*k] / (rho_0 * rho_0) * knl.dw(&x_i, &boundary.pos[*k]))
                .sum::<DVec2>();

        *a_ff = dt_2
            * f_f
                .iter()
                .map(|j| mas[*j] * (t1f + t2f).dot(knl.dw(&x_i, &pos[*j])))
                .sum::<f64>()
            + dt_2
                * f_f
                    .iter()
                    .map(|j| {
                        mas[*j]
                            * (mas[i] / (rho_0 * rho_0) * knl.dw(&pos[*j], &x_i))
                                .dot(knl.dw(&x_i, &pos[*j]))
                    })
                    .sum::<f64>()
            + dt_2
                * f_b
                    .iter()
                    .map(|k| boundary.mas[*k] * (t1f + t2f).dot(knl.dw(&x_i, &boundary.pos[*k])))
                    .sum::<f64>()
    })
}

// STRUCTURE DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[atomic_enum]
#[derive(PartialEq, EnumIter)]
/// A fluid solver, implementing a single simulation step of some SPH method.
pub enum Solver {
    SESPH,
    SplittingSESPH,
    IterSESPH,
    IISPH,
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
            Solver::SplittingSESPH => sesph_splitting(state, current_t, boundary, knls),
            Solver::IterSESPH => sesph_iter(state, current_t, boundary, knls),
            Solver::IISPH => iisph(state, current_t, boundary, knls),
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

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
fn sesph(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    // update densities and pressures
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        bdy,
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
        bdy,
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
        bdy,
        &knls.pressure,
    );
    // perform a time step
    let dt = update_dt(&state.vel, current_t);
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
}

/// Perform a simulation update step using the SESPH solver with splitting.
/// Non-pressure forces are integrated first, then densities are predicted
/// based on the predicted velocities
fn sesph_splitting(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    // apply external forces
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        bdy,
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
        bdy,
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
        bdy,
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
        bdy,
        &knls.density,
    );
}

/// Perform a simulation update step using an iterative SESPH solver with splitting.
/// Non-pressure forces are integrated first, then densities are predicted
/// based on the predicted velocities. This is iterated, refining the predicted velocities.
fn sesph_iter(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    // apply external forces
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        bdy,
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
            bdy,
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
            bdy,
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
        bdy,
        &knls.density,
    );
}

// ATTEMPT AFTER 2013 PAPER
fn alt_iisph(state: &mut Attributes, current_t: &mut f64, boundary: &Boundary, knls: &SphKernel) {
    let knl = &knls.density;
    // PREDICT ADVECTION
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        &knls.density,
    );
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
    state
        .vel
        .par_iter_mut()
        .zip(&state.acc)
        .for_each(|(vel_i, acc_i)| *vel_i += dt * (*acc_i));
    state.d_ii.par_iter_mut().enumerate().for_each(|(i, d_ii)| {
        *d_ii = dt
            * dt
            * state
                .grid
                .query_index(i)
                .iter()
                .map(|j| {
                    -state.mas[*j] / state.den[i].powi(2) * knl.dw(&state.pos[i], &state.pos[*j])
                })
                .sum::<DVec2>();
    });
    predict_densities(
        dt,
        &state.vel,
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        boundary,
        knl,
    );
    state.prs.par_iter_mut().for_each(|prs| *prs = *prs * 0.5);
    state.a_ii.par_iter_mut().enumerate().for_each(|(i, a_ii)| {
        *a_ii = state
            .grid
            .query_index(i)
            .iter()
            .map(|j| {
                let d_ji = -dt.powi(2) * state.mas[i] / state.den[i].powi(2)
                    * knl.dw(&state.pos[*j], &state.pos[i]);
                state.mas[*j] * (state.d_ii[i] - d_ji).dot(knl.dw(&state.pos[i], &state.pos[*j]))
            })
            .sum::<f64>()
    });
    for l in 0..3 {
        let omega = OMEGA_JACOBI.load(Relaxed);
        let rho_0 = RHO_ZERO.load(Relaxed);
        state
            .d_ij_p_j
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, sum_d_ij_p_j)| {
                *sum_d_ij_p_j = dt.powi(2)
                    * state
                        .grid
                        .query_index(i)
                        .iter()
                        .map(|j| {
                            -state.mas[*j] / state.den[*j].powi(2)
                                * state.prs[*j]
                                * knl.dw(&state.pos[i], &state.pos[*j])
                        })
                        .sum::<DVec2>()
            });
        let prs_2 = state.prs.clone();
        state.prs.par_iter_mut().enumerate().for_each(|(i, prs_i)| {
            *prs_i = (1. - omega) * (*prs_i)
                + omega
                    * (1. / state.a_ii[i])
                    * (rho_0
                        - state.den[i]
                        - state
                            .grid
                            .query_index(i)
                            .iter()
                            .map(|j| {
                                state.mas[*j]
                                    * knl.dw(&state.pos[i], &state.pos[*j]).dot(
                                        state.d_ij_p_j[i]
                                            - state.d_ii[*j] * prs_2[*j]
                                            - (state.d_ij_p_j[*j]
                                                - (-dt.powi(2) * state.mas[i]
                                                    / state.den[i].powi(2)
                                                    * knl.dw(&state.pos[*j], &state.pos[i])
                                                    * (*prs_i))),
                                    )
                            })
                            .sum::<f64>())
        })
    }
    state.acc.fill(DVec2::ZERO);
    add_pressure_accelerations(
        &state.pos,
        &state.den,
        &state.prs,
        &state.mas,
        &mut state.acc,
        &state.grid,
        boundary,
        knl,
    );
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt)
}

/// One timestep of the Implicit Incompressible SPH scheme (IISPH)
/// - Paper: https://cg.informatik.uni-freiburg.de/publications/2013_TVCG_IISPH.pdf
/// - SplishSplash implementation for reference: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/IISPH/TimeStepIISPH.cpp
/// - coure notes p.168 onwards: https://cg.informatik.uni-freiburg.de/course_notes/sim_03_particleFluids.pdf
/// - Described in the Tutorial Survey: https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf
fn iisph(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    // compute non-pressure accelerations
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.grid,
        bdy,
        &knls.viscosity,
    );
    // compute v_adv
    let dt = update_dt(&state.vel, current_t);
    half_time_step_euler(&mut state.vel, &state.acc, dt);

    // d_ii term
    state.d_ii.par_iter_mut().enumerate().for_each(|(i, d_ii)| {
        *d_ii = -state.grid.sum_fluid(i, |j| {
            state.mas[*j] / state.den[i].powi(2) * knls.density.dw(&state.pos[i], &state.pos[*j])
        }) - bdy.grid.sum_bdy(&state.pos[i], bdy, |k| {
            bdy.mas[*k] / state.den[i].powi(2) * knls.density.dw(&state.pos[i], &bdy.pos[*k])
        })
    });

    // predicted density due to non-pressure forces
    state
        .den_adv
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, rho_adv_i)| {
            *rho_adv_i = state.den[i]
                + state.grid.sum_fluid(i, |j| {
                    dt * state.mas[*j]
                        * (state.vel[i] - state.vel[*j])
                            .dot(knls.density.dw(&state.pos[i], &state.pos[*j]))
                })
                + bdy.grid.sum_bdy(&state.pos[i], bdy, |k| {
                    dt * bdy.mas[*k]
                        * (state.vel[i] - bdy.vel[*k])
                            .dot(knls.density.dw(&state.pos[i], &bdy.pos[*k]))
                });
        });

    // reset pressures to half value
    state.prs.par_iter_mut().for_each(|prs_i| *prs_i *= 0.5);

    // pre-compute diagonal element for jacobi solver
    state.a_ii.par_iter_mut().enumerate().for_each(|(i, a_ii)| {
        *a_ii = state.grid.sum_fluid(i, |j| {
            let grad = knls.density.dw(&state.pos[i], &state.pos[*j]);
            let d_ji = grad * state.mas[i] / state.den[i].powi(2); // drop the minus by using W_ij instead of W_ji -> antisymmetry
            state.mas[*j] * (state.d_ii[i] - d_ji).dot(grad)
        }) + bdy.grid.sum_bdy(&state.pos[i], bdy, |k| {
            let grad = knls.density.dw(&state.pos[i], &bdy.pos[*k]);
            let d_ji = grad * state.mas[i] / state.den[i].powi(2);
            bdy.mas[*k] * (state.d_ii[i] - d_ji).dot(grad)
        })
    });

    // PRESSURE SOLVE
    let mut l = 0;
    let mut avg_den_err: f64 = 0.0;
    let omega = OMEGA_JACOBI.load(Relaxed);
    let rho_0 = RHO_ZERO.load(Relaxed);
    while avg_den_err >= MAX_RHO_DEVIATION.load(Relaxed) * rho_0
        || l < JACOBI_MIN_ITER.load(Relaxed)
    {
        // pressure solve iteration
        // compute sum over d_ij_p_j
        state
            .d_ij_p_j
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, d_ij_p_j)| {
                *d_ij_p_j = state.grid.sum_fluid(i, |j| {
                    -state.mas[*j] / state.den[*j].powi(2)
                        * state.prs[*j]
                        * knls.pressure.dw(&state.pos[i], &state.pos[*j])
                });
            });

        // compute new pressure, collect into the prs_swap buffer and return predicted error
        avg_den_err = state
            .prs_swap
            .par_iter_mut()
            .enumerate()
            .map(|(i, prs_swap)| {
                let sum = state.grid.sum_fluid(i, |j| {
                    let d_jk_p_k = state.d_ij_p_j[*j];
                    let grad = knls.pressure.dw(&state.pos[i], &state.pos[*j]);
                    let d_ji = grad * state.mas[i] / state.den[i].powi(2);
                    let d_ji_p_i = d_ji * state.prs[i];
                    state.mas[*j]
                        * (state.d_ij_p_j[i]
                            - state.d_ii[*j] * state.prs[*j]
                            - (d_jk_p_k - d_ji_p_i))
                            .dot(grad)
                }) + bdy.grid.sum_bdy(&state.pos[i], bdy, |k| {
                    bdy.mas[*k]
                        * state.d_ij_p_j[i].dot(knls.pressure.dw(&state.pos[i], &bdy.pos[*k]))
                });
                let b = 1. - state.den_adv[i];
                let denominator = state.a_ii[i] * dt.powi(2);
                let new_prs = if denominator.abs() > JACOBI_DENOMINATOR_EPSILON {
                    ((1.0 - omega) * state.prs[i] + omega / denominator * (b - dt.powi(2) * sum))
                        .max(0.0)
                } else {
                    0.0
                };
                *prs_swap = new_prs;
                // return density error
                if new_prs != 0.0 {
                    rho_0 * ((state.a_ii[i] * new_prs + sum) * dt.powi(2) - b)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / (state.pos.len() as f64);

        // swap the buffers back without moving data for efficiency
        std::mem::swap(&mut state.prs, &mut state.prs_swap);

        l += 1;
    }
    // integrate time
    state.acc.fill(DVec2::ZERO);
    add_pressure_accelerations(
        &state.pos,
        &state.den,
        &state.prs,
        &state.mas,
        &mut state.acc,
        &state.grid,
        &bdy,
        &knls.pressure,
    );
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);

    // update densities
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.grid,
        bdy,
        &knls.density,
    );
}

// ATTEMPT AFTER INTERNAL DOC
fn iisph_semiworking(
    state: &mut Attributes,
    current_t: &mut f64,
    boundary: &Boundary,
    knls: &SphKernel,
) {
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
        &mut state.den_adv,
        &state.den,
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
        &mut state.a_ii,
        &state.den,
        &state.pos,
        &state.mas,
        &state.grid,
        boundary,
        dt,
        &knls.density,
    );
    state.prs.fill(0.0);
    state.acc.fill(DVec2::ZERO);
    // state
    //     .prs
    //     .par_iter_mut()
    //     .enumerate()
    //     .for_each(|(i, prs)| *prs = (omega * state.source[i] / state.a_ii[i]).max(0.0));

    // ITERATE
    let mut rho_err: f64 = 0.0;
    let rho_0 = RHO_ZERO.load(Relaxed);
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
                        * (state.prs[i] / state.den[i].powi(2)
                            + state.prs[*j] / state.den[*j].powi(2))
                        * knls.pressure.dw(&state.pos[i], &state.pos[*j])
                })
                .sum::<DVec2>()
                - boundary
                    .grid
                    .query_radius(&state.pos[i], &boundary.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|k| {
                        boundary.mas[*k] * 2. * state.prs[i] / rho_0.powi(2)
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
                let a_p = dt.powi(2)
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
                    + dt.powi(2)
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
                if state.a_ii[i].is_normal() {
                    *prs = ((*prs) + omega / state.a_ii[i] * (state.den_adv[i] - a_p)).max(0.0);
                }

                a_p - state.den_adv[i]
            })
            .max_by(|x, y| x.total_cmp(y))
            .unwrap();
        // .sum::<f64>()
        // / state.pos.len() as f64;

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
                * grid.map_fluid(i, |j| {
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
    bdy: &Boundary,
    knl: &KernelType,
) {
    pos.par_iter()
        .enumerate()
        .zip(den)
        .for_each(|((i, x_i), rho_i)| {
            *rho_i = grid.sum_fluid(i, |j| knl.w(x_i, &pos[*j]) * mas[*j])
                + bdy
                    .grid
                    .sum_bdy(x_i, bdy, |j| knl.w(x_i, &bdy.pos[*j]) * &bdy.mas[*j]);
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

/// Integrates accelerations to velocities using a forwards Euler scheme. This adds to the current velocity.
/// - `v += a * h`
fn half_time_step_euler(vel: &mut [DVec2], acc: &[DVec2], dt: f64) {
    vel.par_iter_mut().zip(acc).for_each(|(v, a)| {
        *v += *a * dt;
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
    source: &mut [f64],
    den: &[f64],
    mas: &[f64],
    pos: &[DVec2],
    vel: &[DVec2],
    knl: &KernelType,
    grid: &Grid,
    boundary: &Boundary,
    dt: f64,
) {
    let rho_0 = RHO_ZERO.load(Relaxed);
    source.par_iter_mut().enumerate().for_each(|(i, source)| {
        *source = rho_0
            - den[i]
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
    den: &[f64],
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
            .map(|j| mas[*j] / den[*j].powi(2) * knl.dw(&x_i, &pos[*j]))
            .sum::<DVec2>();
        let t2f: DVec2 = -2.
            * f_b
                .iter()
                .map(|k| boundary.mas[*k] / rho_0.powi(2) * knl.dw(&x_i, &boundary.pos[*k]))
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
                            * (mas[i] / den[i].powi(2) * knl.dw(&pos[*j], &x_i))
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
    fn solve(&self, state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
        match self {
            Solver::SESPH => sesph(state, current_t, bdy, knls),
            Solver::SplittingSESPH => sesph_splitting(state, current_t, bdy, knls),
            Solver::IterSESPH => sesph_iter(state, current_t, bdy, knls),
            Solver::IISPH => iisph(state, current_t, bdy, knls),
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

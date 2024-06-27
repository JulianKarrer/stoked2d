use crate::{sph::KernelType, *};
use atomic_enum::atomic_enum;
use grid::{Accelerator, Datastructure};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::{fmt::Debug, time::Duration};
use strum_macros::EnumIter;
use utils::{average_val, create_progressbar, max_length, micros_to_seconds, timestamp};

use self::{
    attributes::Attributes,
    boundary::Boundary,
    gui::gui::{REQUEST_RESTART, SIMULATION_TOGGLE},
};

// MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn run(run_for_t: Option<f32>, path: &str) -> bool {
    let bdy = Boundary::from_image(path, BDY_SAMPLING_DENSITY, &{ *SPH_KERNELS.read() }.density);
    let mut state = Attributes::from_image(path, &bdy);
    let mut current_t = 0.0;
    let mut since_resort = 0;
    // reset history and add first timestep
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.ds,
        &bdy,
        &KernelType::GaussSpline3,
    );
    {
        HISTORY
            .write()
            .reset_and_add(&state, &state.ds, &bdy, current_t);
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
        if since_resort > { *RESORT_ATTRIBUTES_EVERY_N.read() } && { *RESORT_ATTRIBUTES.read() } {
            state.resort();
            since_resort = 0;
        } else {
            since_resort += 1;
        }
        state.ds.update_grid(&state.pos, KERNEL_SUPPORT);

        // perform an update step using the selected fluid solver
        let kernels = { *SPH_KERNELS.read() };
        SOLVER
            .load(Relaxed)
            .solve(&mut state, &mut current_t, &bdy, &kernels);
        enforce_boundary_conditions(&mut state.pos, &mut state.vel, &mut state.acc);

        // write back the positions to the global buffer for visualization and update the FPS count
        update_fps(&mut last_update_time);
        if current_t - last_gui_update_t > HISTORY_FRAME_TIME as f64 {
            // update the progress bar
            if let Some(ref bar) = progressbar {
                bar.inc(1);
                bar.set_message(format!("{:.3} ITERS/S", SIM_FPS.load(Relaxed)));
            }
            // update the gui
            last_gui_update_t = current_t;
            {
                HISTORY.write().add_step(&state, &state.ds, current_t);
            }
        } else {
            {
                HISTORY.write().add_plot_data_only(&state, current_t);
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
        &state.ds,
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
        &state.ds,
        bdy,
        &knls.viscosity,
    );
    update_pressures(&state.den, &mut state.prs);
    // apply pressure forces
    add_pressure_accelerations_bdy_rho_0(
        &state.pos,
        &state.den,
        &state.prs,
        &state.mas,
        &mut state.acc,
        &state.ds,
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
        &state.ds,
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
        &state.ds,
        bdy,
        &knls.density,
    );
    update_pressures(&state.den, &mut state.prs);
    // apply pressure forces
    state
        .acc
        .par_iter_mut()
        .for_each(|acc_i| *acc_i = DVec2::ZERO);
    add_pressure_accelerations_bdy_rho_0(
        &state.pos,
        &state.den,
        &state.prs,
        &state.mas,
        &mut state.acc,
        &state.ds,
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
        &state.ds,
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
        &state.ds,
        bdy,
        &knls.viscosity,
    );
    let dt = update_dt(&state.vel, current_t);
    let mut rho_avg_err = 0.0;
    let mut l = 0;
    let rho_0 = RHO_ZERO.load(Relaxed);
    while (rho_avg_err >= MAX_RHO_DEVIATION.load(Relaxed) * rho_0
        || l < JACOBI_MIN_ITER.load(Relaxed))
        && l < JACOBI_MAX_ITER.load(Relaxed)
        && !{ *REQUEST_RESTART.read() }
    {
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
            &state.ds,
            bdy,
            &knls.density,
        );
        rho_avg_err = (average_val(&state.den) - 1.0).max(0.0);
        update_pressures(&state.den, &mut state.prs);
        // apply pressure forces
        state
            .acc
            .par_iter_mut()
            .for_each(|acc_i| *acc_i = DVec2::ZERO);
        add_pressure_accelerations_bdy_rho_0(
            &state.pos,
            &state.den,
            &state.prs,
            &state.mas,
            &mut state.acc,
            &state.ds,
            bdy,
            &knls.pressure,
        );
        l += 1
    }
    JACOBI_LAST_ITER.store(l, Relaxed);
    // perform a time step
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);
    // update densities at the end to make sure the gui output matches reality
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.density,
    );
}

fn iisph_notes(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    // compute & overwrite density
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.density,
    );
    // compute non-pressure accelerations (overwrite)
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.viscosity,
    );
    // compute predicted velocity
    let dt = update_dt_acc(&state.vel, &state.acc, &state.mas, current_t);
    state.vel.par_iter_mut().zip(&state.acc).for_each(|(v, a)| {
        *v += *a * dt;
    });
    // compute source term
    let rho_0 = RHO_ZERO.load(Relaxed);
    let knl = knls.density;
    state
        .source
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, source)| {
            let rho_i = state.den[i];
            let x_i = &state.pos[i];
            let v_i = state.vel[i];
            *source = rho_0
                - rho_i
                - dt * state.ds.sum_fluid(i, |j| {
                    state.mas[*j] * (v_i - state.vel[*j]).dot(knl.dw(x_i, &state.pos[*j]))
                })
                - dt * bdy.ds.sum_bdy(x_i, bdy, |k| {
                    bdy.mas[*k] * (v_i - bdy.vel[*k]).dot(knl.dw(x_i, &bdy.pos[*k]))
                })
        });
    // compute diagonal element
    let gamma_2 = GAMMA_2.load(Relaxed);
    state.a_ii.par_iter_mut().enumerate().for_each(|(i, a_ii)| {
        let x_i = &state.pos[i];
        let m_i = state.mas[i];
        let big_term = -state.ds.sum_fluid(i, |j| {
            state.mas[*j] / rho_0.powi(2) * knl.dw(x_i, &state.pos[*j])
        }) - 2.0
            * gamma_2
            * bdy.ds.sum_bdy(x_i, bdy, |k| {
                bdy.mas[*k] / rho_0.powi(2) * knl.dw(x_i, &bdy.pos[*k])
            });
        *a_ii = state.ds.sum_fluid(i, |j| {
            state.mas[*j] * big_term.dot(knl.dw(x_i, &state.pos[*j]))
        }) + state.ds.sum_fluid(i, |j| {
            state.mas[*j]
                * (m_i / rho_0.powi(2) * knl.dw(&state.pos[*j], x_i))
                    .dot(knl.dw(x_i, &state.pos[*j]))
        }) + bdy.ds.sum_bdy(x_i, bdy, |k| {
            bdy.mas[*k] * big_term.dot(knl.dw(x_i, &bdy.pos[*k]))
        })
    });

    // reset pressure values!
    state.prs.fill(0.0);
    // iterate
    let mut l = 0;
    let mut rho_avg_err = 0.0;
    let omega = OMEGA_JACOBI.load(Relaxed);
    while (rho_avg_err >= MAX_RHO_DEVIATION.load(Relaxed) * rho_0
        || l < JACOBI_MIN_ITER.load(Relaxed))
        && l < JACOBI_MAX_ITER.load(Relaxed)
        && !{ *REQUEST_RESTART.read() }
    {
        // first loop
        state.acc.par_iter_mut().enumerate().for_each(|(i, acc_i)| {
            // compute pressure acceleration
            let x_i = &state.pos[i];
            *acc_i = -state.ds.sum_fluid(i, |j| {
                state.mas[*j]
                    * (state.prs[i] / rho_0.powi(2) + state.prs[*j] / rho_0.powi(2))
                    * knl.dw(x_i, &state.pos[*j])
            }) - gamma_2
                * bdy.ds.sum_bdy(x_i, bdy, |k| {
                    bdy.mas[*k] * 2.0 * state.prs[i] / rho_0.powi(2) * knl.dw(x_i, &bdy.pos[*k])
                });
        });
        // second loop
        rho_avg_err = state
            .prs
            .par_iter_mut()
            .enumerate()
            .map(|(i, prs_i)| {
                // compute divergence of velocity change dt*a due to pressure acc
                let x_i = &state.pos[i];
                let ap_i = dt.powi(2)
                    * state.ds.sum_fluid(i, |j| {
                        state.mas[*j]
                            * (state.acc[i] - state.acc[*j]).dot(knl.dw(x_i, &state.pos[*j]))
                    })
                    + dt.powi(2)
                        * bdy.ds.sum_bdy(x_i, bdy, |k| {
                            bdy.mas[*k] * state.acc[i].dot(knl.dw(x_i, &bdy.pos[*k]))
                        });
                // update pressure if a_ii != 0
                let a_ii = state.a_ii[i] * dt.powi(2);
                if a_ii.abs() > JACOBI_DENOMINATOR_EPSILON.load(Relaxed) {
                    *prs_i = ((*prs_i) + omega * (state.source[i] - ap_i) / a_ii).max(0.0)
                };
                // return predicted density error
                ap_i - state.source[i]
            })
            // .reduce_with(|a, b| a.max(b))
            // .unwrap();
            .sum::<f64>()
            / (state.pos.len() as f64);

        l += 1;
    }
    JACOBI_LAST_ITER.store(l, Relaxed);
    // compute pressure accelerations with final pressure values
    // exact same function as above for consistency!!
    state.acc.par_iter_mut().enumerate().for_each(|(i, acc_i)| {
        // compute pressure acceleration
        let x_i = &state.pos[i];
        *acc_i = -state.ds.sum_fluid(i, |j| {
            state.mas[*j]
                * (state.prs[i] / rho_0.powi(2) + state.prs[*j] / rho_0.powi(2))
                * knl.dw(x_i, &state.pos[*j])
        }) - gamma_2
            * bdy.ds.sum_bdy(x_i, bdy, |k| {
                bdy.mas[*k] * 2.0 * state.prs[i] / rho_0.powi(2) * knl.dw(x_i, &bdy.pos[*k])
            });
    });
    // integrate accelerations to velocity and position updates
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt)
}

// iisph_notes adjusted to be more like iisph_2013
fn iisph_mine(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    // compute & overwrite density
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.density,
    );
    // compute non-pressure accelerations (overwrite)
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.viscosity,
    );
    // compute predicted velocity
    let dt = update_dt_acc(&state.vel, &state.acc, &state.mas, current_t);
    state.vel.par_iter_mut().zip(&state.acc).for_each(|(v, a)| {
        *v += *a * dt;
    });
    // compute source term
    let rho_0 = RHO_ZERO.load(Relaxed);
    let knl = knls.density;
    state
        .source
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, source)| {
            let rho_i = state.den[i];
            let x_i = &state.pos[i];
            let v_i = state.vel[i];
            *source = rho_0
                - rho_i
                - dt * state.ds.sum_fluid(i, |j| {
                    state.mas[*j] * (v_i - state.vel[*j]).dot(knl.dw(x_i, &state.pos[*j]))
                })
                - dt * bdy.ds.sum_bdy(x_i, bdy, |k| {
                    bdy.mas[*k] * (v_i - bdy.vel[*k]).dot(knl.dw(x_i, &bdy.pos[*k]))
                })
        });
    // compute diagonal element
    let gamma_2 = GAMMA_2.load(Relaxed);
    state.a_ii.par_iter_mut().enumerate().for_each(|(i, a_ii)| {
        let x_i = &state.pos[i];
        let m_i = state.mas[i];
        let c_f = -state.ds.sum_fluid(i, |j| {
            state.mas[*j] / state.den[i].powi(2) * knl.dw(x_i, &state.pos[*j])
        }) - 2.0
            * gamma_2
            * bdy.ds.sum_bdy(x_i, bdy, |k| {
                bdy.mas[*k] / state.den[i].powi(2) * knl.dw(x_i, &bdy.pos[*k])
            });
        *a_ii = state
            .ds
            .sum_fluid(i, |j| state.mas[*j] * c_f.dot(knl.dw(x_i, &state.pos[*j])))
            + state.ds.sum_fluid(i, |j| {
                state.mas[*j]
                    * (m_i / state.den[i].powi(2) * knl.dw(&state.pos[*j], x_i))
                        .dot(knl.dw(x_i, &state.pos[*j]))
            })
            + bdy.ds.sum_bdy(x_i, bdy, |k| {
                bdy.mas[*k] * c_f.dot(knl.dw(x_i, &bdy.pos[*k]))
            })
    });

    // reset pressure values!
    state.prs.par_iter_mut().for_each(|prs| *prs *= 0.5);
    // iterate
    let mut l = 0;
    let mut rho_avg_err = 0.0;
    let omega = OMEGA_JACOBI.load(Relaxed);
    while (rho_avg_err >= MAX_RHO_DEVIATION.load(Relaxed) * rho_0
        || l < JACOBI_MIN_ITER.load(Relaxed))
        && l < JACOBI_MAX_ITER.load(Relaxed)
        && !{ *REQUEST_RESTART.read() }
    {
        // first loop
        state.acc.par_iter_mut().enumerate().for_each(|(i, acc_i)| {
            // compute pressure acceleration
            let x_i = &state.pos[i];
            *acc_i = -state.ds.sum_fluid(i, |j| {
                state.mas[*j]
                    * (state.prs[i] / state.den[i].powi(2) + state.prs[*j] / state.den[*j].powi(2))
                    * knl.dw(x_i, &state.pos[*j])
            }) - gamma_2
                * bdy.ds.sum_bdy(x_i, bdy, |k| {
                    bdy.mas[*k] * 2.0 * state.prs[i] / state.den[i].powi(2)
                        * knl.dw(x_i, &bdy.pos[*k])
                });
        });
        // second loop
        rho_avg_err = state
            .prs
            .par_iter_mut()
            .enumerate()
            .map(|(i, prs_i)| {
                // compute divergence of velocity change dt*a due to pressure acc
                let x_i = &state.pos[i];
                let ap_i = dt.powi(2)
                    * state.ds.sum_fluid(i, |j| {
                        state.mas[*j]
                            * (state.acc[i] - state.acc[*j]).dot(knl.dw(x_i, &state.pos[*j]))
                    })
                    + dt.powi(2)
                        * bdy.ds.sum_bdy(x_i, bdy, |k| {
                            bdy.mas[*k] * state.acc[i].dot(knl.dw(x_i, &bdy.pos[*k]))
                        });
                // update pressure if a_ii != 0
                let a_ii = state.a_ii[i] * dt.powi(2);
                if a_ii.abs() > JACOBI_DENOMINATOR_EPSILON.load(Relaxed) {
                    *prs_i = ((*prs_i) + omega * (state.source[i] - ap_i) / a_ii).max(0.0);
                };
                // return predicted density error
                ap_i - state.source[i]
            })
            // .reduce_with(|a, b| a.max(b))
            // .unwrap();
            .sum::<f64>()
            / (state.pos.len() as f64);

        l += 1;
    }
    JACOBI_LAST_ITER.store(l, Relaxed);
    // compute pressure accelerations with final pressure values
    // exact same function as above for consistency!!
    state.acc.par_iter_mut().enumerate().for_each(|(i, acc_i)| {
        // compute pressure acceleration
        let x_i = &state.pos[i];
        *acc_i = -state.ds.sum_fluid(i, |j| {
            state.mas[*j]
                * (state.prs[i] / state.den[i].powi(2) + state.prs[*j] / state.den[*j].powi(2))
                * knl.dw(x_i, &state.pos[*j])
        }) - gamma_2
            * bdy.ds.sum_bdy(x_i, bdy, |k| {
                bdy.mas[*k] * 2.0 * state.prs[i] / state.den[i].powi(2) * knl.dw(x_i, &bdy.pos[*k])
            });
    });
    // integrate accelerations to velocity and position updates
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt)
}

/// One timestep of the Implicit Incompressible SPH scheme (IISPH)
/// - Paper: https://cg.informatik.uni-freiburg.de/publications/2013_TVCG_IISPH.pdf
/// - SplishSplash implementation for reference: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/IISPH/TimeStepIISPH.cpp
/// - coure notes p.168 onwards: https://cg.informatik.uni-freiburg.de/course_notes/sim_03_particleFluids.pdf
/// - Described in the Tutorial Survey: https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf
fn iisph_2013(state: &mut Attributes, current_t: &mut f64, bdy: &Boundary, knls: &SphKernel) {
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.density,
    );
    // compute non-pressure accelerations
    apply_non_pressure_forces(
        &state.pos,
        &state.vel,
        &mut state.acc,
        &state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.viscosity,
    );
    // compute v_adv
    let dt = update_dt(&state.vel, current_t);
    half_time_step_euler(&mut state.vel, &state.acc, dt);

    // d_ii term
    state.d_ii.par_iter_mut().enumerate().for_each(|(i, d_ii)| {
        let one_over_rho_i_2 = 1.0 / state.den[i].powi(2);
        *d_ii = -state.ds.sum_fluid(i, |j| {
            state.mas[*j] * one_over_rho_i_2 * knls.density.dw(&state.pos[i], &state.pos[*j])
        }) - bdy.ds.sum_bdy(&state.pos[i], bdy, |k| {
            bdy.mas[*k] * one_over_rho_i_2 * knls.density.dw(&state.pos[i], &bdy.pos[*k])
        })
    });

    // predicted density due to non-pressure forces
    state
        .den_adv
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, rho_adv_i)| {
            *rho_adv_i = state.den[i]
                + dt * state.ds.sum_fluid(i, |j| {
                    state.mas[*j]
                        * (state.vel[i] - state.vel[*j])
                            .dot(knls.density.dw(&state.pos[i], &state.pos[*j]))
                })
                + dt * bdy.ds.sum_bdy(&state.pos[i], bdy, |k| {
                    bdy.mas[*k]
                        * (state.vel[i] - bdy.vel[*k])
                            .dot(knls.density.dw(&state.pos[i], &bdy.pos[*k]))
                });
        });

    // pre-compute diagonal element for jacobi solver
    state.a_ii.par_iter_mut().enumerate().for_each(|(i, a_ii)| {
        let m_i_over_rho_i_2 = state.mas[i] / state.den[i].powi(2);
        let d_ii = state.d_ii[i];
        *a_ii = state.ds.sum_fluid(i, |j| {
            let grad = knls.density.dw(&state.pos[i], &state.pos[*j]);
            let d_ji = grad * m_i_over_rho_i_2; // drop the minus by using W_ij instead of W_ji -> antisymmetry
            state.mas[*j] * (d_ii - d_ji).dot(grad)
        }) + bdy.ds.sum_bdy(&state.pos[i], bdy, |k| {
            let grad = knls.density.dw(&state.pos[i], &bdy.pos[*k]);
            let d_ji = grad * m_i_over_rho_i_2;
            bdy.mas[*k] * (d_ii - d_ji).dot(grad)
        })
    });

    // reset pressures to half value
    state.prs.par_iter_mut().for_each(|prs_i| *prs_i *= 0.5);

    // PRESSURE SOLVE
    let mut l = 0;
    let mut avg_den_err: f64 = 0.0;
    let omega = OMEGA_JACOBI.load(Relaxed);
    let rho_0 = RHO_ZERO.load(Relaxed);
    while (avg_den_err >= MAX_RHO_DEVIATION.load(Relaxed) * rho_0
        || l < JACOBI_MIN_ITER.load(Relaxed))
        && !{ *REQUEST_RESTART.read() }
    {
        // pressure solve iteration
        // compute sum over d_ij_p_j
        state
            .d_ij_p_j
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, d_ij_p_j)| {
                let x_i = state.pos[i];
                *d_ij_p_j = -state.ds.sum_fluid(i, |j| {
                    state.mas[*j] / state.den[*j].powi(2)
                        * state.prs[*j]
                        * knls.pressure.dw(&x_i, &state.pos[*j])
                });
            });

        // compute new pressure, collect into the prs_swap buffer and return predicted error
        avg_den_err = state
            .prs_swap
            .par_iter_mut()
            .enumerate()
            .map(|(i, prs_swap)| {
                let m_i_over_rho_i_2 = state.mas[i] / state.den[i].powi(2);
                let prs_i = state.prs[i];
                let x_i = &state.pos[i];
                let d_ij_pj = state.d_ij_p_j[i];
                let a_ii = state.a_ii[i];
                let sum = state.ds.sum_fluid(i, |j| {
                    let d_jk_p_k = state.d_ij_p_j[*j];
                    let grad = knls.pressure.dw(x_i, &state.pos[*j]);
                    let d_ji = grad * m_i_over_rho_i_2;
                    let d_ji_p_i = d_ji * prs_i;
                    state.mas[*j]
                        * (d_ij_pj - state.d_ii[*j] * state.prs[*j] - (d_jk_p_k - d_ji_p_i))
                            .dot(grad)
                }) + bdy.ds.sum_bdy(x_i, bdy, |k| {
                    bdy.mas[*k] * d_ij_pj.dot(knls.pressure.dw(x_i, &bdy.pos[*k]))
                });
                let b = rho_0 - state.den_adv[i];
                let denominator = a_ii * dt.powi(2);
                let new_prs = if denominator.abs() > JACOBI_DENOMINATOR_EPSILON.load(Relaxed) {
                    ((1.0 - omega) * state.prs[i] + omega / denominator * (b - dt.powi(2) * sum))
                        .max(0.0)
                } else {
                    0.0
                };
                *prs_swap = new_prs;
                // return density error
                if new_prs != 0.0 {
                    rho_0 * ((a_ii * new_prs + sum) * dt.powi(2) - b)
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
    JACOBI_LAST_ITER.store(l, Relaxed);
    // integrate time
    set_pressure_accelerations_bdy_mirror(
        &state.pos,
        &state.den,
        &state.prs,
        &state.mas,
        &mut state.acc,
        &state.ds,
        &bdy,
        &knls.pressure,
    );
    time_step_euler_cromer(&mut state.pos, &mut state.vel, &state.acc, dt);

    // update densities
    update_densities(
        &state.pos,
        &mut state.den,
        &state.mas,
        &state.ds,
        bdy,
        &knls.density,
    );

    // // ASSERT DENSITY FINITE  AND != ZERO
    // state.den.par_iter().enumerate().for_each(|(i, den)| {
    //     assert!(
    //         den.is_normal() || den.is_subnormal(),
    //         "is normal {}, i {}, neighbours {:?}, neighbours radius {:?} ",
    //         den.is_normal(),
    //         i,
    //         state.grid.query_index(i),
    //         state
    //             .grid
    //             .query_radius(&state.pos[i], &state.pos, KERNEL_SUPPORT),
    //     )
    // })
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
        let v_max = max_length(vel);
        let dt_est = LAMBDA.load(Relaxed) * H / v_max;
        if dt_est.is_finite() {
            dt_est.clamp(MIN_DT.load(Relaxed), MAX_DT.load(Relaxed))
        } else {
            MIN_DT.load(Relaxed)
        }
    };
    *current_t += dt;
    dt
}

/// More accurate CFL condition (?) that factors in forces acting on the particles
/// https://maverick.inria.fr/~Prashant.Goswami/Research/Papers/VRIPHYS11_TAASPH.pdf
fn update_dt_acc(vel: &[DVec2], acc: &[DVec2], m: &[f64], current_t: &mut f64) -> f64 {
    let dt = if USE_FIXED_DT.load(Relaxed) {
        FIXED_DT.load(Relaxed)
    } else {
        let v_max = max_length(vel);
        let f_max = acc
            .par_iter()
            .zip(m)
            .map(|(acc, m)| acc.length() * m)
            .reduce_with(|a, b| a.max(b))
            .unwrap();
        let v_max = v_max + (H * f_max).sqrt();
        let dt_est = LAMBDA.load(Relaxed) * H / v_max;
        if dt_est.is_finite() {
            dt_est.clamp(MIN_DT.load(Relaxed), MAX_DT.load(Relaxed))
        } else {
            MIN_DT.load(Relaxed)
        }
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
    ds: &Datastructure,
    bdy: &Boundary,
    vis_knl: &KernelType,
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
                * ds.sum_fluid(i, |j| {
                        let x_i_j = *x_i - pos[*j];
                        let v_i_j = *v_i - vel[*j];
                        mas[*j] / den[*j] * (v_i_j).dot(x_i_j)
                            / (x_i_j.length_squared() + 0.01 * H * H)
                            * vis_knl.dw(x_i, &pos[*j])
                    });
            assert!(vis.is_finite());
            // boundary viscosity
            let bdy_normal = bdy
                .ds
                .sum_bdy(x_i, bdy, |j| vis_knl.dw(x_i, &bdy.pos[*j]))
                .normalize_or_zero();
            let vis_bdy: DVec2 = nu2
                * 2.*(DIMENSIONS + 2.) // 2*(dimensions + 2)
                *  bdy
                .ds
                .sum_bdy(x_i, bdy, |j| {
                    let x_i_j = *x_i - bdy.pos[*j];
                    let v_i_j = *v_i; // v at boundary is zero
                    bdy.mas[*j] / rho_0 * (v_i_j).dot(x_i_j)
                        / (x_i_j.length_squared() + 0.01 * H * H)
                        * vis_knl.dw(x_i, &bdy.pos[*j])
                });
            assert!(vis_bdy.is_finite());
            *a_i = acc_g + vis + vis_bdy.dot(bdy_normal) * bdy_normal
        });
}

/// Update the densities at each particle
pub fn update_densities(
    pos: &[DVec2],
    den: &mut [f64],
    mas: &[f64],
    ds: &Datastructure,
    bdy: &Boundary,
    knl: &KernelType,
) {
    pos.par_iter()
        .enumerate()
        .zip(den)
        .for_each(|((i, x_i), rho_i)| {
            *rho_i = ds.sum_fluid(i, |j| knl.w(x_i, &pos[*j]) * mas[*j])
                + bdy
                    .ds
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
    ds: &Datastructure,
    bdy: &Boundary,
    knl: &KernelType,
) {
    pos.par_iter()
        .enumerate()
        .zip(den)
        .zip(vel)
        .for_each(|(((i, x_i), rho_i), vel_i)| {
            *rho_i = ds.sum_fluid(i, |j| knl.w(x_i, &pos[*j]) * mas[*j])
                + dt * ds.sum_fluid(i, |j| mas[*j] * knl.dw(x_i, &pos[*j]).dot(*vel_i - vel[*j]))
                + bdy
                    .ds
                    .sum_bdy(x_i, bdy, |j| knl.w(x_i, &bdy.pos[*j]) * &bdy.mas[*j])
                + dt * bdy.ds.sum_bdy(x_i, bdy, |j| {
                    &bdy.mas[*j] * knl.dw(x_i, &bdy.pos[*j]).dot(*vel_i)
                });
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
/// adding the result to the current accelerations.
/// Assumes that boundary particles have rest density, but mirrors pressure.
fn add_pressure_accelerations_bdy_rho_0(
    pos: &[DVec2],
    den: &[f64],
    prs: &[f64],
    mas: &[f64],
    acc: &mut [DVec2],
    ds: &Datastructure,
    bdy: &Boundary,
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
            *acc += -ds.sum_fluid(i, |j| {
                mas[*j]
                    * (p_i_over_rho_i_squared + prs[*j] / (den[*j] * den[*j]))
                    * knl.dw(x_i, &pos[*j])
            }) - gamma_2
                * (p_i_over_rho_i_squared + p_i * one_over_rho_0_squared)
                * bdy
                    .ds
                    .sum_bdy(x_i, bdy, |k| knl.dw(x_i, &bdy.pos[*k]) * bdy.mas[*k]);
        })
}

/// Compute pressure accelerations from the momentum-preserving SPH approximation of the density gradient,
/// overwriting the current accelerations with the result.
/// Mirrors both density and pressure of particles to neighbouring boundary particles.
fn set_pressure_accelerations_bdy_mirror(
    pos: &[DVec2],
    den: &[f64],
    prs: &[f64],
    mas: &[f64],
    acc: &mut [DVec2],
    ds: &Datastructure,
    bdy: &Boundary,
    knl: &KernelType,
) {
    let gamma_2 = GAMMA_2.load(Relaxed);
    pos.par_iter()
        .enumerate()
        .zip(prs)
        .zip(den)
        .zip(acc)
        .for_each(|((((i, x_i), p_i), rho_i), acc)| {
            let p_i_over_rho_i_squared = p_i / (rho_i * rho_i);
            let two_p_i_over_rho_i_squared = 2. * p_i_over_rho_i_squared;
            *acc = -ds.sum_fluid(i, |j| {
                mas[*j]
                    * (p_i_over_rho_i_squared + prs[*j] / (den[*j] * den[*j]))
                    * knl.dw(x_i, &pos[*j])
            }) - gamma_2
                * two_p_i_over_rho_i_squared
                * bdy
                    .ds
                    .sum_bdy(x_i, bdy, |k| knl.dw(x_i, &bdy.pos[*k]) * bdy.mas[*k]);
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

// STRUCTURE DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[atomic_enum]
#[derive(PartialEq, EnumIter, Default)]
/// A fluid solver, implementing a single simulation step of some SPH method.
pub enum Solver {
    SESPH,
    SplittingSESPH,
    IterSESPH,
    #[default]
    Iisph2013,
    IisphNotes,
    IisphMine,
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
            Solver::Iisph2013 => iisph_2013(state, current_t, bdy, knls),
            Solver::IisphNotes => iisph_notes(state, current_t, bdy, knls),
            Solver::IisphMine => iisph_mine(state, current_t, bdy, knls),
        }
    }
}

#[atomic_enum]
#[derive(PartialEq, EnumIter, Default)]
/// An equation relating measured density deviation to pressure, or stress to strain.
/// While Clamped equations produce only pressures > 0 counteracting compression, the respective
/// unclamped version can lead to negative pressures, creating attraction between particles in
/// low density regions. The global static stiffness K may factor into these equations.
pub enum PressureEquation {
    Absolute,
    Relative,
    #[default]
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

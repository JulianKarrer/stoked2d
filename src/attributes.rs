use crate::{
    boundary::Boundary,
    grid::{Accelerator, Datastructure},
    simulation::update_densities,
    utils::{average_val, is_black, is_blue, linspace},
    FLUID, GRAVITY, H, INITIAL_JITTER, KERNEL_SUPPORT, RHO_ZERO, SCALE, SPH_KERNELS, V_ZERO,
};
use glam::DVec2;
use image::open;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::sync::atomic::Ordering::Relaxed;

/// Holds all particle data as a struct of arrays
pub struct Attributes {
    // quantities that need to be resorted
    pub pos: Vec<DVec2>,
    pub vel: Vec<DVec2>,
    pub mas: Vec<f64>,
    pub prs: Vec<f64>,
    pub den: Vec<f64>,
    // quantities that are overwritten in each timestep
    pub acc: Vec<DVec2>,
    pub a_ii: Vec<f64>,
    pub den_adv: Vec<f64>,
    pub d_ii: Vec<DVec2>,
    pub d_ij_p_j: Vec<DVec2>,
    pub prs_swap: Vec<f64>,
    pub source: Vec<f64>,
    // structures held that might be updated every timestep
    pub ds: Datastructure,
}

impl Attributes {
    fn attributes_from_pos(pos: Vec<DVec2>, bdy: &Boundary) -> Self {
        // initial velocities, accelerations and pressures are zero
        // initial densities are defined to be the rest density
        let rho_0 = RHO_ZERO.load(Relaxed);
        let vel: Vec<DVec2> = vec![DVec2::ZERO; pos.len()];
        let acc: Vec<DVec2> = vec![DVec2::ZERO; pos.len()];
        let prs: Vec<f64> = vec![0.0; pos.len()];
        let den: Vec<f64> = vec![rho_0; pos.len()];
        // buffers for IISPH
        let a_ii = vec![0.0; pos.len()];
        let den_adv = vec![0.0; pos.len()];
        let d_ii = vec![DVec2::ZERO; pos.len()];
        let sum_d_ij_p_j = vec![DVec2::ZERO; pos.len()];
        let prs_swap: Vec<f64> = vec![0.0; pos.len()];
        let source = vec![0.0; pos.len()];
        // create an acceleration datastructure for the positions
        let mut ds = Datastructure::new(pos.len());
        ds.update_grid(&pos, KERNEL_SUPPORT);
        // define the masses of each particle such that rest density is
        // achieved in the initial configuration
        let knl = { SPH_KERNELS.read().density };
        let mut den_measured = den.clone();
        let m_0 = rho_0 * V_ZERO;
        let mut mas: Vec<f64> = vec![m_0; pos.len()];

        for _ in 0..500 {
            update_densities(&pos, &mut den_measured, &mas, &ds, bdy, &knl);
            mas.par_iter_mut()
                .zip(&den_measured)
                .for_each(|(m, rho_i)| *m = 0.5 * (*m) + 0.5 * ((*m) * rho_0 / rho_i));
        }

        println!("average mass/m_0: {}", average_val(&mas) / m_0);

        let mut res = Self {
            pos,
            vel,
            acc,
            prs,
            den,
            mas,
            ds,
            a_ii,
            d_ii,
            d_ij_p_j: sum_d_ij_p_j,
            den_adv,
            prs_swap,
            source,
        };
        // immediately resort the attributes that are stored between timesteps to be aligned
        // in memory with the space-filling curve used in the grid
        // -> better memory coherence, more cache hits, faster queries
        res.resort();
        // return the struct
        res
    }

    /// Initialize a new set of particles attributes, filling the area within the
    /// box given by FLUID with particles using spacing H
    pub fn new(bdy: &Boundary) -> Self {
        // estimate the number of particles beforehand for allocation
        let n: usize = ((FLUID[1].x - FLUID[0].x) / (H) + 1.0).ceil() as usize
            * ((FLUID[1].y - FLUID[0].y) / (H) + 1.0).ceil() as usize;
        let mut pos: Vec<DVec2> = Vec::with_capacity(n);
        // initialization
        let jitter = INITIAL_JITTER.load(Relaxed);
        let mut small_rng = SmallRng::seed_from_u64(42);
        let mut x = FLUID[0].x;
        let mut y = FLUID[0].y;
        while y <= FLUID[1].y {
            while x <= FLUID[1].x {
                pos.push(
                    DVec2::new(x, y)
                        + DVec2::new(
                            small_rng.gen_range(-jitter..=jitter),
                            small_rng.gen_range(-jitter..=jitter),
                        ),
                );
                x += H;
            }
            x = FLUID[0].x;
            y += H;
        }
        Self::attributes_from_pos(pos, bdy)
    }

    /// Generate a new fluid particle set from a path to an image.
    /// - each pixel in the input image has a length given in`spacing`
    /// - the centre of the image will be centred around the origin
    /// - all blue pixels are boundaries (b > r+10,g+10 and alpha > 100)
    pub fn from_image(path: &str, bdy: &Boundary) -> Self {
        let rgba = open(path).unwrap().into_rgba8();
        let (xsize, ysize) = rgba.dimensions();
        let x_half = (xsize as f64) * SCALE / 2.;
        let y_half = (ysize as f64) * SCALE / 2.;
        let jitter = INITIAL_JITTER.load(Relaxed);

        let y_num = ((2. * y_half) / H) as usize;
        let pos: Vec<DVec2> = linspace(-y_half, y_half, y_num)
            .par_iter()
            .enumerate()
            .flat_map(|(y_index, y)| {
                let hex_offset = if y_index % 2 == 0 {
                    0.0 // 0.25 * H
                } else {
                    0.0 // -0.25 * H
                };
                linspace(-x_half, x_half, ((2. * x_half) / H) as usize)
                    .iter()
                    .enumerate()
                    .filter_map(|(x_index, x)| {
                        let mut small_rng =
                            SmallRng::seed_from_u64((x_index * y_num + y_index) as u64);
                        // if the current pixel position is in bounds
                        if let Some(pixel) = rgba.get_pixel_checked(
                            ((x + x_half) / SCALE) as u32,
                            ((y + y_half) / SCALE) as u32,
                        ) {
                            // and if the current pixel is blue
                            let [r, g, b, a] = pixel.0;
                            if is_blue(r, g, b, a) && (!is_black(r, g, b, a)) {
                                let new_pos = DVec2::new(x + hex_offset, -y)
                                    + DVec2::new(
                                        small_rng.gen_range(-jitter..=jitter),
                                        small_rng.gen_range(-jitter..=jitter),
                                    );
                                // only place fluid where there is no boundary
                                if bdy.ds.all_in_radius(
                                    &new_pos,
                                    &bdy.pos,
                                    KERNEL_SUPPORT,
                                    |bdy_j| bdy.pos[*bdy_j].distance(new_pos) >= H,
                                ) {
                                    // place a particle at the position
                                    return Some(new_pos);
                                }
                            }
                        }
                        None
                    })
                    .collect::<Vec<DVec2>>()
            })
            .collect();
        println!("N={}", pos.len());
        Self::attributes_from_pos(pos, bdy)
    }

    /// Resort all particle attributes according to some given order, which must be
    /// a permutation of (0..NUMBER_OF_PARTICLES). This is meant to eg. improve cache-hit-rates
    /// by employing the same sorting as the acceleration datastructure provides for neighbourhood queries.
    pub fn resort(&mut self) {
        // extract the order of the attributes according to cell-wise z-ordering
        let order: Vec<usize> = self.ds.resort_order();
        debug_assert!(order.len() == self.pos.len());
        // re-order relevant particle attributes in accordance with the order
        self.pos = order.par_iter().map(|i| self.pos[*i]).collect();
        self.vel = order.par_iter().map(|i| self.vel[*i]).collect();
        self.mas = order.par_iter().map(|i| self.mas[*i]).collect();
        self.prs = order.par_iter().map(|i| self.prs[*i]).collect();
        self.den = order.par_iter().map(|i| self.den[*i]).collect();
    }

    // HAMIULTONIAN COMPUTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pub fn compute_hamiltonian(&self) -> f64 {
        self.compute_average_kinetic_energy()
            + self.compute_average_gravitational_potential_energy()
    }

    // Compute the total kinteic energy of the system.
    pub fn compute_average_kinetic_energy(&self) -> f64 {
        self.vel
            .par_iter()
            .zip(&self.mas)
            .map(|(v_i, m_i)| 0.5 * m_i * v_i.length_squared())
            .sum::<f64>()
            / (self.pos.len() as f64)
    }

    // Compute the total kinteic energy of the system.
    fn compute_average_gravitational_potential_energy(&self) -> f64 {
        let g = GRAVITY.load(Relaxed).abs();
        self.pos
            .par_iter()
            .zip(&self.mas)
            .map(|(x_i, m_i)| m_i * g * (x_i.y))
            .sum::<f64>()
            / (self.pos.len() as f64)
    }
}

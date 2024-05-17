use crate::{
    boundary::Boundary,
    datastructure::Grid,
    utils::{is_black, is_blue, linspace},
    FLUID, H, INITIAL_JITTER, KERNEL_SUPPORT, RHO_ZERO, SPH_KERNELS, V_ZERO,
};
use glam::DVec2;
use image::open;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::atomic::Ordering::Relaxed;

/// Holds all particle data as a struct of arrays
pub struct Attributes {
    // quantities that need to be resorted
    pub pos: Vec<DVec2>,
    pub vel: Vec<DVec2>,
    pub mas: Vec<f64>,
    // quantities that are overwritten in each timestep
    pub den: Vec<f64>,
    pub prs: Vec<f64>,
    pub acc: Vec<DVec2>,
    // structures held that might be updated every timestep
    pub grid: Grid,
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
        // create an acceleration datastructure for the positions
        let mut grid = Grid::new(pos.len());
        grid.update_grid(&pos, KERNEL_SUPPORT);
        // define the masses of each particle such that rest density is
        // achieved in the initial configuration
        let knl = { SPH_KERNELS.read().density };
        let m_0 = rho_0 * V_ZERO;
        let mas = pos
            .par_iter()
            .enumerate()
            .map(|(i, x_i)| {
                // https://cg.informatik.uni-freiburg.de/publications/2018_TOG_pressureBoundaries.pdf
                // see Equation (13)
                // actual volume of a particle is determined and mass set to enforce rho_0
                let v_f = (m_0 / rho_0)
                    / (grid
                        .query_index(i)
                        .iter()
                        .map(|j_f| knl.w(x_i, &pos[*j_f]))
                        .sum::<f64>()
                        * (m_0 / rho_0)
                        + bdy
                            .grid
                            .query_radius(x_i, &bdy.pos, KERNEL_SUPPORT)
                            .iter()
                            .map(|j_b| knl.w(x_i, &bdy.pos[*j_b]) * (bdy.m[*j_b] / rho_0))
                            .sum::<f64>());
                v_f * rho_0
            })
            .collect();
        let mut res = Self {
            pos,
            vel,
            acc,
            prs,
            den,
            mas,
            grid,
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
    pub fn from_image(path: &str, spacing: f64, bdy: &Boundary) -> Self {
        let rgba = open(path).unwrap().into_rgba8();
        let (xsize, ysize) = rgba.dimensions();
        let x_half = (xsize as f64) * spacing / 2.;
        let y_half = (ysize as f64) * spacing / 2.;
        let jitter = INITIAL_JITTER.load(Relaxed);

        let y_num = ((2. * y_half) / H) as usize;
        let pos = linspace(-y_half, y_half, y_num)
            .par_iter()
            .enumerate()
            .flat_map(|(y_index, y)| {
                let hex_offset = if y_index % 2 == 0 {
                    0.25 * H
                } else {
                    -0.25 * H
                };
                linspace(-x_half, x_half, ((2. * x_half) / H) as usize)
                    .iter()
                    .enumerate()
                    .filter_map(|(x_index, x)| {
                        let mut small_rng =
                            SmallRng::seed_from_u64((x_index * y_num + y_index) as u64);
                        // if the current pixel position is in bounds
                        if let Some(pixel) = rgba.get_pixel_checked(
                            ((x + x_half) / spacing) as u32,
                            ((y + y_half) / spacing) as u32,
                        ) {
                            // and if the current pixel is blue
                            let [r, g, b, a] = pixel.0;
                            if is_blue(r, g, b, a) && (!is_black(r, g, b, a)) {
                                let new_pos = DVec2::new(x + hex_offset, -y);
                                // only place fluid where there is no boundary
                                if bdy
                                    .grid
                                    .query_radius(&new_pos, &bdy.pos, KERNEL_SUPPORT)
                                    .iter()
                                    .all(|bdy_j| bdy.pos[*bdy_j].distance(new_pos) >= H)
                                {
                                    // place a particle at the position
                                    return Some(
                                        new_pos
                                            + DVec2::new(
                                                small_rng.gen_range(-jitter..=jitter),
                                                small_rng.gen_range(-jitter..=jitter),
                                            ),
                                    );
                                }
                            }
                        }
                        None
                    })
                    .collect::<Vec<DVec2>>()
            })
            .collect();
        Self::attributes_from_pos(pos, bdy)
    }

    /// Resort all particle attributes according to some given order, which must be
    /// a permutation of (0..NUMBER_OF_PARTICLES). This is meant to eg. improve cache-hit-rates
    /// by employing the same sorting as the acceleration datastructure provides for neighbourhood queries.
    pub fn resort(&mut self) {
        // extract the order of the attributes according to cell-wise z-ordering
        let order: Vec<usize> = self.grid.handles.par_iter().map(|h| h.index).collect();
        debug_assert!(order.len() == self.pos.len());
        // re-order relevant particle attributes in accordance with the order
        self.pos = order.par_iter().map(|i| self.pos[*i]).collect();
        self.vel = order.par_iter().map(|i| self.vel[*i]).collect();
        self.mas = order.par_iter().map(|i| self.mas[*i]).collect();
    }
}

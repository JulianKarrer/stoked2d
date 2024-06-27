use crate::{
    grid::{Accelerator, Datastructure},
    gui::gui::ZOOM,
    sph::KernelType,
    utils::{is_black, linspace},
    BDY_SAMPLING_DENSITY, BOUNDARY, GAMMA_1, GAMMA_2, H, HARD_BOUNDARY, INITIAL_JITTER,
    KERNEL_SUPPORT, RHO_ZERO, SCALE, WINDOW_SIZE,
};
use glam::DVec2;
use image::open;
use ocl::prm::Float2;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::atomic::Ordering::Relaxed;

/// A struct representing a set of boundary particles and a respective data structure
/// for querying their positions which can be used to mirror pressure forces, creating
/// static boundaries for the simulation that use the SPH pressure solver to enforce
/// impenetrability of the boundary.
pub struct Boundary {
    pub pos: Vec<DVec2>,
    pub vel: Vec<DVec2>,
    pub mas: Vec<f64>,
    pub ds: Datastructure,
}

impl Boundary {
    /// Create a `Boundary` from a vector of boundary particle positions as generated in
    /// `Boundary::new` or `Boundary::from_image`.
    fn boundary_from_positions(pos: Vec<DVec2>, knl: &KernelType) -> Self {
        let mut pos = pos;
        // create a grid with the boundary particles
        let mut ds = Datastructure::new(pos.len());
        ds.update_grid(&pos, KERNEL_SUPPORT);
        // immediately resort the positions vector for spatial locality
        let order: Vec<usize> = ds.resort_order().par_iter().map(|x| *x as usize).collect();
        pos = order.par_iter().map(|i| pos[*i]).collect();
        ds.update_grid(&pos, KERNEL_SUPPORT);

        // compute gamma values
        // Boundary::calculate_gammas(knl);

        // compute the masses of each boundary particle
        let mut res = Self {
            mas: vec![0.; pos.len()],
            vel: vec![DVec2::ZERO; pos.len()],
            pos,
            ds,
        };
        res.update_masses(knl);
        // set the hard boundary
        let xmin = res
            .pos
            .par_iter()
            .min_by(|a, b| a.x.total_cmp(&b.x))
            .unwrap()
            .x;
        let xmax = res
            .pos
            .par_iter()
            .min_by(|a, b| b.x.total_cmp(&a.x))
            .unwrap()
            .x;
        let ymin = res
            .pos
            .par_iter()
            .min_by(|a, b| a.y.total_cmp(&b.y))
            .unwrap()
            .y;
        let ymax = res
            .pos
            .par_iter()
            .min_by(|a, b| b.y.total_cmp(&a.y))
            .unwrap()
            .y;
        let hbdy = [
            Float2::new(xmin as f32 - 5. * H as f32, ymin as f32 - 5. * H as f32),
            Float2::new(xmax as f32 + 5. * H as f32, ymax as f32 + 5. * H as f32),
        ];
        {
            *HARD_BOUNDARY.write() = hbdy;
        }
        // adjust zoom to updated boundary
        ZOOM.store(
            (WINDOW_SIZE[0].load(Relaxed) as f32).min(WINDOW_SIZE[1].load(Relaxed) as f32)
                / (hbdy[1][0] - hbdy[0][0]).max(hbdy[1][1] - hbdy[0][1])
                * 0.9,
            Relaxed,
        );
        // return the result
        res
    }

    /// Creates a new set of boundary particles in the pos Vec, with an accompanying
    /// grid to query for boundary neighbours.
    /// The initialization creates layers of particles around the rectangle specified by
    /// the static BOUNDARY.
    ///
    /// The internal grid is not meant to be updated, since the boundary is static.
    /// Boundary structs can and should therefore always be immutable.
    pub fn new(layers: usize, knl: &KernelType) -> Self {
        // initialize boundary particle positions
        let mut pos = vec![];
        let mut rng = SmallRng::seed_from_u64(42);
        for i in 0..layers {
            let mut x = BOUNDARY[0].x + H;
            while x <= BOUNDARY[1].x {
                pos.push(DVec2::new(x, BOUNDARY[0].y - i as f64 * H));
                pos.push(DVec2::new(x, BOUNDARY[1].y + i as f64 * H));
                x += H * rng.gen_range(0.0..=1.);
            }
        }
        for i in 0..layers {
            let mut y = BOUNDARY[0].y - (layers - 1) as f64 * H;
            while y < BOUNDARY[1].y + (layers) as f64 * H {
                pos.push(DVec2::new(BOUNDARY[0].x - i as f64 * H, y));
                pos.push(DVec2::new(BOUNDARY[1].x + i as f64 * H, y));
                y += H * rng.gen_range(0.0..=1.);
            }
        }
        Self::boundary_from_positions(pos, knl)
    }

    /// Generate a new boundary particle set from a path to an image.
    /// - each pixel in the input image has a length given in`spacing`
    /// - the centre of the image will be centred around the origin
    /// - all black pixels are boundaries (r,g,b < 10 and alpha > 100)
    pub fn from_image(path: &str, spacing: f64, knl: &KernelType) -> Self {
        let rgba = open(path).unwrap().into_rgba8();
        let (xsize, ysize) = rgba.dimensions();
        let x_half = (xsize as f64) * SCALE / 2.;
        let y_half = (ysize as f64) * SCALE / 2.;

        let jitter = INITIAL_JITTER.load(Relaxed) / H * BDY_SAMPLING_DENSITY;
        let y_num = ((2. * y_half) / spacing) as usize;
        let pos: Vec<DVec2> = linspace(-y_half, y_half, y_num)
            .par_iter()
            .flat_map(|y| {
                linspace(-x_half, x_half, ((2. * x_half) / spacing) as usize)
                    .iter()
                    .filter_map(|x| {
                        let mut small_rng = SmallRng::seed_from_u64(
                            ((((x + x_half) / SCALE) as u64) << 32)
                                | (((y + y_half) / SCALE) as u64),
                        );
                        // if the current pixel position is in bounds
                        if let Some(pixel) = rgba.get_pixel_checked(
                            ((x + x_half) / SCALE) as u32,
                            ((y + y_half) / SCALE) as u32,
                        ) {
                            // and if the current pixel is blue
                            let [r, g, b, a] = pixel.0;
                            if is_black(r, g, b, a) {
                                return Some(
                                    DVec2::new(*x, -y)
                                        + DVec2::new(
                                            small_rng.gen_range(-jitter..=jitter),
                                            small_rng.gen_range(-jitter..=jitter),
                                        ),
                                );
                            }
                        }
                        None
                    })
                    .collect::<Vec<DVec2>>()
            })
            .collect();
        Self::boundary_from_positions(pos, knl)
    }

    /// Update the virtual masses of the boundary particles,
    /// which are calculated as a correcting factor for non-uniformly sampled
    /// boundaries.
    ///
    /// These masses are computed as: m_i_b = \frac{ \rho_0 \gamma_1 }{\sum_{i_{b_b}} W_{i_b, i_{b_b}}}
    pub fn update_masses(&mut self, knl: &KernelType) {
        let gamma_1 = GAMMA_1.load(Relaxed);
        let rho_0 = RHO_ZERO.load(Relaxed);
        self.mas = self
            .pos
            .par_iter()
            .map(|x_i| rho_0 * gamma_1 / self.ds.sum_bdy(x_i, &self, |j| knl.w(x_i, &self.pos[*j])))
            .collect();
    }

    /// Calculate and set γ<sub>1</sub> and γ<sub>2</sub> into `GAMMA_1` and `GAMMA_2`.
    /// -  γ<sub>1</sub>  is the correcting factor for the density computation in a
    /// single-layer boundary scenario and depends on the Kernel function, Kernel
    /// support and dimensionality
    /// - similarly, γ<sub>2</sub> is the correction factor for the caluclation
    /// of pressure accelerations.
    /// - both factors should be one for a 2D Cubic Spline Kernel with 2h support.
    pub fn calculate_gammas(knl: &KernelType) {
        // the ideal setting should sample points at least within the kernel support range
        let size = (KERNEL_SUPPORT / H).ceil() + 1.;
        let size_int = size as i32;
        let mut x = -size * H;
        let mut pos: Vec<DVec2> = vec![];
        // the ideal scenario: fluid resting on a single uniform boundary layer
        // -> start fluid from y=0
        while x <= size * H + 10e-12 {
            for y in 0..size_int {
                pos.push(DVec2::new(x, y as f64 * H));
            }
            x += H;
        }
        let bdy: Vec<DVec2> = (-size_int..=size_int)
            .map(|i| DVec2::new((i as f64) * H, -H))
            .collect();
        let x_i = DVec2::ZERO;
        // calculate gamma 1
        let fluid_sum: f64 = pos.iter().map(|x_j| knl.w(&x_i, x_j)).sum();
        let bdy_sum: f64 = bdy.iter().map(|x_j| knl.w(&x_i, x_j)).sum();
        GAMMA_1.store(((1. / (H * H)) - fluid_sum) / bdy_sum, Relaxed);
        // calculate gamma 2
        let grad_fluid_sum: DVec2 = pos.iter().map(|x_j| -knl.dw(&x_i, x_j)).sum::<DVec2>();
        let grad_bdy_sum: DVec2 = bdy.iter().map(|x_j| knl.dw(&x_i, x_j)).sum::<DVec2>();
        let gamma_2 = (grad_fluid_sum.dot(grad_bdy_sum)) / (grad_bdy_sum.dot(grad_bdy_sum));
        GAMMA_2.store(gamma_2, Relaxed)
    }
}

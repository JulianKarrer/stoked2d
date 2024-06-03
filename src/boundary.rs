use crate::{
    datastructure::Grid, gui::gui::ZOOM, sph::KernelType, utils::is_black, BOUNDARY, GAMMA_1,
    GAMMA_2, H, HARD_BOUNDARY, KERNEL_SUPPORT, RHO_ZERO, SPH_KERNELS, WINDOW_SIZE,
};
use glam::DVec2;
use image::open;
use ocl::prm::Float2;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge,
    ParallelIterator,
};
use std::sync::atomic::Ordering::Relaxed;

/// A struct representing a set of boundary particles and a respective data structure
/// for querying their positions which can be used to mirror pressure forces, creating
/// static boundaries for the simulation that use the SPH pressure solver to enforce
/// impenetrability of the boundary.
pub struct Boundary {
    pub pos: Vec<DVec2>,
    pub vel: Vec<DVec2>,
    pub mas: Vec<f64>,
    pub grid: Grid,
}

impl Boundary {
    /// Create a `Boundary` from a vector of boundary particle positions as generated in
    /// `Boundary::new` or `Boundary::from_image`.
    fn boundary_from_positions(pos: Vec<DVec2>, knl: &KernelType) -> Self {
        let mut pos = pos;
        // create a grid with the boundary particles
        let mut grid = Grid::new(pos.len());
        grid.update_grid(&pos, KERNEL_SUPPORT);
        // immediately resort the positions vector for spatial locality
        let order: Vec<usize> = grid.handles.par_iter().map(|h| h.index).collect();
        pos = order.par_iter().map(|i| pos[*i]).collect();
        grid.update_grid(&pos, KERNEL_SUPPORT);

        // compute gamma values
        // Boundary::calculate_gammas(knl);

        // compute the masses of each boundary particle
        let mut res = Self {
            mas: vec![0.; pos.len()],
            vel: vec![DVec2::ZERO; pos.len()],
            pos,
            grid,
        };
        res.update_masses();
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
            Float2::new(xmin as f32, ymin as f32),
            Float2::new(xmax as f32, ymax as f32),
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
        let x_half = (xsize as f64) * spacing / 2.;
        let y_half = (ysize as f64) * spacing / 2.;
        let pos: Vec<DVec2> = rgba
            .pixels()
            .enumerate()
            .par_bridge()
            .filter(|(_i, p)| {
                let [r, g, b, a] = p.0;
                is_black(r, g, b, a)
            })
            .map(|(i, _p)| {
                let x = ((i as u32) % xsize) as f64 * spacing;
                let y = ((i as u32) / xsize) as f64 * spacing;
                DVec2::new(x - x_half, -(y - y_half))
            })
            .collect();
        Self::boundary_from_positions(pos, knl)
    }

    /// Update the virtual masses of the boundary particles,
    /// which are calculated as a correcting factor for non-uniformly sampled
    /// boundaries.
    ///
    /// These masses are computed as: m_i = \frac{ \rho_0 \gamma_1 }{\sum_{i_b} W_{i, i_{b}}}
    pub fn update_masses(&mut self) {
        let gamma_1 = GAMMA_1.load(Relaxed);
        let rho_0 = RHO_ZERO.load(Relaxed);
        let knl = { SPH_KERNELS.read().clone() }.density;
        self.mas.par_iter_mut().zip(&self.pos).for_each(|(m, x_i)| {
            *m = rho_0 * gamma_1
                / self
                    .grid
                    .query_radius(x_i, &self.pos, KERNEL_SUPPORT)
                    .iter()
                    .map(|j| knl.w(x_i, &self.pos[*j]))
                    .sum::<f64>();
        })
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

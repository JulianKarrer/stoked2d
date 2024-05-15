use crate::{datastructure::Grid, FLUID, H, INITIAL_JITTER, M};
use glam::DVec2;
use image::open;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::atomic::Ordering::Relaxed;

/// Holds all particle data as a struct of arrays
pub struct Attributes {
    pub pos: Vec<DVec2>,
    pub vel: Vec<DVec2>,
    pub acc: Vec<DVec2>,
    pub prs: Vec<f64>,
    pub den: Vec<f64>,
}

impl Attributes {
    fn attributes_from_pos(pos: Vec<DVec2>) -> Self {
        let vel: Vec<DVec2> = vec![DVec2::ZERO; pos.len()];
        let acc: Vec<DVec2> = vec![DVec2::ZERO; pos.len()];
        let prs: Vec<f64> = vec![0.0; pos.len()];
        let den: Vec<f64> = vec![M / (H * H); pos.len()];
        Self {
            pos,
            vel,
            acc,
            prs,
            den,
        }
    }

    /// Initialize a new set of particles attributes, filling the area within the
    /// box given by FLUID with particles using spacing H
    pub fn new() -> Self {
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
        Self::attributes_from_pos(pos)
    }

    /// Generate a new fluid particle set from a path to an image.
    /// - each pixel in the input image has a length given in`spacing`
    /// - the centre of the image will be centred around the origin
    /// - all blue pixels are boundaries (b > r+10,g+10 and alpha > 100)
    pub fn from_image(path: &str, spacing: f64) -> Self {
        let rgba = open(path).unwrap().into_rgba8();
        let (xsize, ysize) = rgba.dimensions();
        let x_half = (xsize as f64) * spacing / 2.;
        let y_half = (ysize as f64) * spacing / 2.;

        let mut pos = vec![];
        let mut x = -x_half;
        let mut y = -y_half;
        let x_max = x_half;
        let y_max = y_half;

        let jitter = INITIAL_JITTER.load(Relaxed);
        let mut small_rng = SmallRng::seed_from_u64(42);
        while y <= y_max {
            while x <= x_max {
                // if the current pixel position is in bounds
                if let Some(pixel) = rgba.get_pixel_checked(
                    ((x + x_half) / spacing) as u32,
                    ((y + y_half) / spacing) as u32,
                ) {
                    // and if the current pixel is blue
                    let [r, g, b, a] = pixel.0;
                    if b > (r + 10) && b > (g + 10) && a > 100 {
                        // place a particle at the position
                        pos.push(
                            DVec2::new(x, -y)
                                + DVec2::new(
                                    small_rng.gen_range(-jitter..=jitter),
                                    small_rng.gen_range(-jitter..=jitter),
                                ),
                        );
                    }
                }
                x += H;
            }
            x = -x_half;
            y += H;
        }
        println!("pos length: {}", pos.len());
        Self::attributes_from_pos(pos)
    }

    /// Resort all particle attributes according to some given order, which must be
    /// a permutation of (0..NUMBER_OF_PARTICLES). This is meant to eg. improve cache-hit-rates
    /// by employing the same sorting as the acceleration datastructure provides for neighbourhood queries.
    pub fn resort(&mut self, grid: &Grid) {
        // extract the order of the attributes according to cell-wise z-ordering
        let order: Vec<usize> = grid.handles.par_iter().map(|h| h.index).collect();
        debug_assert!(order.len() == self.pos.len());
        // re-order relevant particle attributes in accordance with the order
        self.pos = order.par_iter().map(|i| self.pos[*i]).collect();
        self.vel = order.par_iter().map(|i| self.vel[*i]).collect();
    }
}

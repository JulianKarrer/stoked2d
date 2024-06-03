use crate::*;
use atomic_enum::atomic_enum;
use boundary::Boundary;
use lindel::{hilbert_encode, morton_encode};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use voracious_radix_sort::{RadixSort, Radixable};

#[derive(Copy, Clone, Debug, Default)]
pub struct Handle {
    pub index: usize,
    pub cell: u64,
}
impl Handle {
    pub fn new(index: usize, cell: u64) -> Self {
        Self { index, cell }
    }
}
impl PartialOrd for Handle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cell.partial_cmp(&other.cell)
    }
}
impl PartialEq for Handle {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}
impl Radixable<u64> for Handle {
    type Key = u64;

    fn key(&self) -> Self::Key {
        self.cell
    }
}

#[derive(Debug)]
/// A uniform grid used for neghbourhood queries.
///
/// A new grid must be created by specifying the number of
/// particles to be used so that allocations can be made.
/// Then, the grid is updated once per timestep and returns a
/// superset of the neighbours for a given point on queries.
///
/// Internally, this uses a z-ordering space filling curve to linearise
/// space which is discretized into cubes, the edge length of which is the length
/// of the kernel support. Handles to the particles with cell indices are sorted
/// with respect to cell index and a compact list of filled cells subsequently
/// created which indexes into the Vector of handles.
pub struct Grid {
    min: DVec2,
    pub handles: Vec<Handle>,
    pub neighbours: Vec<Vec<usize>>,
    curve: GridCurve,
}

#[derive(PartialEq)]
#[atomic_enum]
pub enum GridCurve {
    Morton,
    XYZ,
    Hilbert,
}
impl GridCurve {
    fn key(&self, p: &DVec2, min: &DVec2, gridsize: f64) -> u64 {
        let ind = ((*p - *min) / gridsize).as_uvec2().to_array();
        match self {
            GridCurve::Morton => morton_encode(ind),
            GridCurve::Hilbert => hilbert_encode(ind),
            GridCurve::XYZ => ((ind[1] as u64) << 32) | (ind[0] as u64),
        }
    }
}

static DIRECTIONS: [DVec2; 9] = [
    DVec2::new(-1.0, -1.0),
    DVec2::new(-1.0, 0.0),
    DVec2::new(-1.0, 1.0),
    DVec2::new(0.0, -1.0),
    DVec2::new(1.0, -1.0),
    DVec2::new(0.0, 0.0),
    DVec2::new(1.0, 0.0),
    DVec2::new(0.0, 1.0),
    DVec2::new(1.0, 1.0),
];

impl Grid {
    /// Create a new grid, allocating space for the given number of particles
    pub fn new(number_of_particles: usize) -> Self {
        Self {
            min: DVec2::NEG_INFINITY,
            handles: vec![Handle::default(); number_of_particles],
            neighbours: vec![vec![]; number_of_particles],
            curve: GRID_CURVE.load(Relaxed),
        }
    }

    /// Update a grid using the current positions of particles so that neighbours can be queried
    pub fn update_grid(&mut self, pos: &[DVec2], gridsize: f64) {
        // instead of defining the grid from a fixed point or in a fixed rectangle, define it from the minimum
        // point of the bounding volume containing all points to allow potentially inifnite grids
        self.min = pos
            .par_iter()
            .cloned()
            .reduce(|| DVec2::MAX, |a, b| a.min(b))
            - DVec2::ONE * gridsize * 2.;
        // compute the cell index for each particle, creating a "handle" as defined above
        self.curve = GRID_CURVE.load(Relaxed);
        self.handles
            .par_iter_mut()
            .zip(pos)
            .enumerate()
            .for_each(|(i, (h, p))| *h = Handle::new(i, self.curve.key(p, &self.min, gridsize)));
        // sort particles with respect to their cell index
        self.handles.voracious_mt_sort(*THREADS);
        // assert that the handles are in fact sorted by cell index
        // precompute all neighbours
        self.neighbours.par_iter_mut().zip(pos).for_each(|(n, p)| {
            *n = DIRECTIONS
                .iter()
                .map(|d| {
                    let x = *p + (*d) * gridsize;
                    self.curve.key(&x, &self.min, gridsize)
                })
                .flat_map(|code| {
                    let start = self.handles.partition_point(|h| h.cell < code);
                    if start < self.handles.len() && self.handles[start].cell == code {
                        Some(
                            self.handles
                                .iter()
                                .skip(start)
                                .take_while(|h| h.cell == code)
                                .map(|h| h.index)
                                .filter(|j| p.distance(pos[*j]) <= gridsize)
                                .collect::<Vec<usize>>(),
                        )
                    } else {
                        None
                    }
                })
                .flatten()
                .collect()
        });
    }

    /// Query the for all neighbours within a given radius.
    /// Returns only the actual neighbours, filtering anything more distant than 'search_radius'
    pub fn query_radius(&self, p: &DVec2, pos: &[DVec2], search_radius: f64) -> Vec<usize> {
        DIRECTIONS
            .iter()
            .map(|d| {
                let x = *p + (*d) * search_radius;
                self.curve.key(&x, &self.min, search_radius)
            })
            .flat_map(|code| {
                let start = self.handles.partition_point(|h| h.cell < code);
                if start < self.handles.len() && self.handles[start].cell == code {
                    Some(
                        self.handles
                            .iter()
                            .skip(start)
                            .take_while(|h| h.cell == code)
                            .map(|h| h.index)
                            .filter(|j| p.distance(pos[*j]) <= search_radius)
                            .collect::<Vec<usize>>(),
                    )
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    /// Query the for the precomputed neighbour list computed when grid.update is called.
    /// A superset of the actual neighbours is returned, potentially containing non-neighbours.
    /// Since the results of this function are typically used as input to compact kernel functions,
    /// this is unproblematic.
    pub fn query_index(&self, i: usize) -> &[usize] {
        &self.neighbours[i]
    }

    // CONVENIENCE FUNCTIONS FOR ITERATING WITH CLOSURES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /// Take a `FnMut` closure and apply it to each neighbour of the particle with index `i`.
    /// Shorthand for `self.query_index(i).iter().for_each(f)`
    pub fn for_each_fluid<F>(&self, i: usize, f: F)
    where
        F: FnMut(&usize),
    {
        self.query_index(i).iter().for_each(f)
    }

    /// Take a `FnMut` closure and map it to each neighbour of the particle with index `i`.
    /// Shorthand for `self.query_index(i).iter().map(f)`
    pub fn map_fluid<F, R>(&self, i: usize, f: F) -> std::iter::Map<std::slice::Iter<'_, usize>, F>
    where
        F: FnMut(&usize) -> R,
    {
        self.query_index(i).iter().map(f)
    }

    /// Take a `FnMut` closure, map it to each neighbour of the particle with index `i` and sum the result.
    /// Shorthand for `self.query_index(i).iter().map(f).sum()`
    pub fn sum_fluid<F, R>(&self, i: usize, f: F) -> R
    where
        F: FnMut(&usize) -> R,
        R: std::iter::Sum,
    {
        self.query_index(i).iter().map(f).sum()
    }

    /// Take an immutable closure and apply it to all boundary neighbours of the given particle index,
    /// returning the sum of the results.
    pub fn sum_bdy<F, R>(&self, x_i: &DVec2, bdy: &Boundary, f: F) -> R
    where
        F: FnMut(&usize) -> R,
        R: std::iter::Sum,
    {
        self.query_radius(x_i, &bdy.pos, KERNEL_SUPPORT)
            .iter()
            .map(f)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use rand::{thread_rng, Rng};
    use rayon::prelude::ParallelBridge;
    use test::Bencher;

    const REPETITIONS: usize = 10;
    const TEST_RUNS: usize = 100;
    const PARTICLES: usize = 5000;
    const RANGE: f64 = 10.0;
    const GRIDSIZE: f64 = 0.5;

    const BENCH_PARTICLES: usize = 2500;
    const BENCH_TIMESTEPS_SIMULATED: usize = 1;
    const BENCH_PARTICLE_DX_PER_TIMESTEP: f64 = 0.1;
    const BENCH_QUERIES_PER_STEP: usize = 2;

    fn random_vec2(range: f64) -> DVec2 {
        DVec2::new(
            thread_rng().gen_range(-range..range),
            thread_rng().gen_range(-range..range),
        )
    }

    #[test]
    /// Compare the neighbour set computed from 'query_radius' with a reference set computed using brute force.
    fn grid_query_radius_correctness() {
        for gridsize in [0.1, 0.5, 1., 1.5] {
            for _ in 0..REPETITIONS {
                let mut grid = Grid::new(PARTICLES);
                let pos: Vec<DVec2> = (0..PARTICLES)
                    .par_bridge()
                    .map(|_| random_vec2(RANGE))
                    .collect();
                grid.update_grid(&pos, gridsize);
                for _ in 0..TEST_RUNS {
                    let p = random_vec2(RANGE);
                    let answer: Vec<usize> = grid.query_radius(&p, &pos, gridsize);
                    let expected: Vec<(usize, DVec2)> = pos
                        .par_iter()
                        .enumerate()
                        .flat_map(|(i, x)| {
                            if x.distance(p) <= gridsize {
                                Some((i, *x))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // all expected neightbours must be in the answer
                    for (n, (i, _)) in expected.iter().enumerate() {
                        assert!(
                            answer.contains(i),
                            "answer at {}: {:?}\nexpected{:?}",
                            n,
                            answer,
                            expected
                        )
                    }
                    // only the neighbours must be in the answer
                    assert!(answer.len() == expected.len())
                }
            }
        }
    }

    #[test]
    /// Compare the neighbour set computed from 'query_index' with a reference set computed using brute force.
    fn grid_query_index_correctness() {
        for gridsize in [0.9, 1.0, 1.1] {
            for _ in 0..REPETITIONS {
                let mut grid = Grid::new(PARTICLES);
                let pos: Vec<DVec2> = (0..PARTICLES)
                    .par_bridge()
                    .map(|_| random_vec2(RANGE))
                    .collect();
                grid.update_grid(&pos, gridsize);
                for (i, p) in pos.iter().enumerate() {
                    let answer = grid.query_index(i);
                    let expected: Vec<(usize, DVec2)> = pos
                        .par_iter()
                        .enumerate()
                        .flat_map(|(j, x)| {
                            if x.distance(*p) <= gridsize {
                                Some((j, *x))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // all expected neightbours must be in the answer
                    for (j, _) in &expected {
                        assert!(answer.contains(j))
                    }
                    // only the neighbours must be in the answer
                    assert!(
                        answer.len() == expected.len(),
                        "{:?} \n {:?}",
                        answer,
                        expected
                    )
                }
            }
        }
    }

    #[bench]
    fn grid_bench_grid(b: &mut Bencher) {
        b.iter(|| {
            let queries: Vec<DVec2> = (0..BENCH_PARTICLES).map(|_| random_vec2(RANGE)).collect();
            let mut pos: Vec<DVec2> = (0..BENCH_PARTICLES).map(|_| random_vec2(RANGE)).collect();
            let mut grid = Grid::new(BENCH_PARTICLES);
            for _ in 0..BENCH_TIMESTEPS_SIMULATED {
                grid.update_grid(&pos, GRIDSIZE);
                for _ in 0..BENCH_QUERIES_PER_STEP {
                    queries.par_iter().for_each(|q| {
                        test::black_box(grid.query_radius(q, &pos, GRIDSIZE));
                    });
                }
                pos.par_iter_mut().for_each(|p| {
                    *p += random_vec2(BENCH_PARTICLE_DX_PER_TIMESTEP);
                })
            }
        });
    }
}

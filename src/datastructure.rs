use crate::*;
use atomic_enum::atomic_enum;
use lindel::{morton_encode, hilbert_encode};
use rayon::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefMutIterator};
use voracious_radix_sort::{Radixable, RadixSort};


#[derive(Copy, Clone, Debug, Default)]
pub struct Handle{
  pub index: usize,
  cell: u64
}
impl PartialOrd for Handle{
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.cell.partial_cmp(&other.cell)
  }
}
impl PartialEq for Handle{
  fn eq(&self, other: &Self) -> bool {
    self.cell == other.cell
  }
}
impl Radixable<u64> for Handle{
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
pub struct Grid{
  min: DVec2,
  pub handles:Vec<Handle>,
  cells:Vec<Handle>
}

#[derive(PartialEq)]
#[atomic_enum]
pub enum GridCurve{
  Morton,
  Hilbert,
}

fn get_key(p:&DVec2, min: &DVec2, gridsize: f64, curve: GridCurve)->u64{
  let ind = ((*p-*min)/gridsize).as_uvec2().to_array();
  match curve{
    GridCurve::Morton => morton_encode(ind),
    GridCurve::Hilbert => hilbert_encode(ind),
  }
  
}

static DIRECTIONS:[DVec2;9] = [
  DVec2::new(-1.0, -1.0), DVec2::new(0.0, -1.0), DVec2::new(1.0, -1.0),
  DVec2::new(-1.0, 0.0), DVec2::new(0.0, 0.0), DVec2::new(1.0, 0.0),
  DVec2::new(-1.0, 1.0), DVec2::new(0.0, 1.0), DVec2::new(1.0, 1.0),
];

impl Grid{
  /// Update a grid using the current positions of particles so that neighbours can be queried
  pub fn update_grid(&mut self, pos: &[DVec2], gridsize: f64){
    // instead of defining the grid from a fixed point or in a fixed rectangle, define it from the minimum
    // point of the bounding volume containing all points to allow potentially inifnite grids
    self.min = pos.par_iter().cloned().reduce(|| DVec2::MAX, |a, b|a.min(b)) - DVec2::ONE*gridsize;
    // compute the cell index for each particle, creating a "handle" as defined above
    let curve = GRID_CURVE.load(Relaxed);
    self.handles.par_iter_mut().zip(pos).enumerate().for_each(|(i, (h, p))|
      *h = Handle { index: i, cell: get_key(p, &self.min, gridsize, curve) }
    );
    // sort particles with respect to their cell index
    self.handles.voracious_mt_sort(*THREADS);
    // create a compact list of all filled cells from the sorted list of particles
    self.cells = self.handles.par_iter().enumerate().map(|(i,h)|
      if i==0 || h.cell != self.handles[i-1].cell {Some(Handle{index: i, cell: h.cell})} else {None}
    ).flatten().collect();
  }

  /// Query the for all neighbours within a given radius.
  /// A superset of the actual neighbours is returned, potentially containing non-neighbours.
  /// Since the results of this function are typically used as input to compact kernel functions, 
  /// this is unproblematic.
  pub fn query(&self, p:&DVec2, search_radius:f64)->Vec<usize>{
    let curve = GRID_CURVE.load(Relaxed);
    DIRECTIONS.iter().map(|d|{
      let x = *p+(*d)*search_radius;
      get_key(&x, &self.min, search_radius, curve)
    }).flat_map(|code| 
      if let Ok(cells_index) = self.cells.binary_search_by_key(&code, |h| h.cell) {
        Some(
          self.handles.iter()
            .skip(self.cells[cells_index].index)
            .take_while(|h|h.cell == self.cells[cells_index].cell)
            .map(|h|h.index)
            .collect::<Vec<usize>>()
        )
      } else {None}
    ).flatten().collect()
  }

  /// Create a new grid, allocating space for the given number of particles
  pub fn new(number_of_particles:usize)->Self{
    Self { min: DVec2::NEG_INFINITY, handles: vec![Handle::default(); number_of_particles], cells: vec![] }
  }
}


#[cfg(test)]
mod tests {
  use super::*;
  extern crate test;
  use test::Bencher;
  use rand::{thread_rng, Rng};
  use rayon::prelude::{ParallelIterator, ParallelBridge};

  const REPETITIONS:usize = 10;
  const TEST_RUNS:usize = 1000;
  const PARTICLES:usize = 5000;
  const RANGE:f64 = 10.0;
  const GRIDSIZE:f64 = 1.0;

  const BENCH_PARTICLES:usize = 2500;
  const BENCH_TIMESTEPS_SIMULATED:usize = 1;
  const BENCH_PARTICLE_DX_PER_TIMESTEP:f64 = 0.1;
  const BENCH_QUERIES_PER_STEP:usize = 2;
  
  fn random_vec2(range:f64)->DVec2{
    DVec2::new(
      thread_rng().gen_range(-range..range),
      thread_rng().gen_range(-range..range),
    )
  }

  #[test]
  fn grid_artificial_example(){
    let grid = Grid{ 
      min: DVec2::ZERO, 
      handles: vec![Handle{ index: 0, cell: 0 }, Handle{ index: 1, cell: 0 }, Handle{ index: 2, cell: 3 }, Handle{ index: 3, cell: 8 }], 
      cells: vec![Handle{ index: 0, cell: 0 }, Handle{ index: 2, cell: 3 }, Handle{ index: 3, cell: 8 }, ]
    };
    let answer = grid.query(&(DVec2::ONE*0.5), 1.0);
    let expected = vec![0usize, 1];
    assert!(answer.iter().zip(expected).all(|(a,b)|*a==b))
  }

  #[test]
  fn grid_update_and_query_correctness() {
    let curve = GRID_CURVE.load(Relaxed);
    for _ in 0..REPETITIONS{
      let mut grid = Grid::new(PARTICLES);
      let pos:Vec<DVec2> = (0..PARTICLES).par_bridge().map(|_|random_vec2(RANGE)).collect();
      grid.update_grid(&pos, GRIDSIZE);
      for _ in 0..TEST_RUNS {
        let p = random_vec2(RANGE);
        let answer:Vec<usize> = grid.query(&p, GRIDSIZE);
        let expected:Vec<(usize, DVec2)> = pos.par_iter().enumerate().flat_map(|(i, x)|
          if x.distance(p) <= GRIDSIZE {
            Some((i,*x))
          } else {None}
        ).collect();

        // the answer can't contain less entries than the solution, as it must be a superset
        assert!(
          answer.len()>=expected.len(), 
          "p: {}\nanswer: {:?}\nexpected: {:?}\ngrid: {:?}\npos: {:?}\ncodes: {:?}", 
          p, answer, expected, grid, pos, pos.iter().map(|p|get_key(p, &grid.min, GRIDSIZE, curve)).collect::<Vec<u64>>()
        );

        // all neightbours must be in the answer
        for i in &expected {
          assert!(
            answer.contains(&i.0), 
            "p: {}\nx: {}\ni:{}\nanswer:{:?}\nexpected: {:?}\ngrid:{:?}", 
            p, pos[i.0], i.0, answer, expected, grid
          )
        }
        
      }
    }
  }

  #[bench]
  fn grid_bench_grid(b: &mut Bencher) {
    b.iter(|| {
      let queries:Vec<DVec2> = (0..BENCH_PARTICLES).map(|_|random_vec2(RANGE)).collect();
      let mut pos:Vec<DVec2> = (0..BENCH_PARTICLES).map(|_|random_vec2(RANGE)).collect();
      let mut grid = Grid::new(BENCH_PARTICLES);
      for _ in 0..BENCH_TIMESTEPS_SIMULATED{
        grid.update_grid(&pos, GRIDSIZE);
        for _ in 0..BENCH_QUERIES_PER_STEP{
          queries.par_iter().for_each(|q|{test::black_box( 
            grid.query(q, GRIDSIZE) 
          );});
        }
        pos.par_iter_mut().for_each(|p|{*p+=random_vec2(BENCH_PARTICLE_DX_PER_TIMESTEP);})
      }
    });
  }

}
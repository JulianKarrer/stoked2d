# Stoked2D - 2D SPH Fluid Simulation

This crate implements a state-equation based solver for weakly compressible fluids using Smoothed Particle Hydrodynamics (SPH) methods on the GPU using Rust and OpenCL. 


The solver comes with a GUI for interactive adjustment of simulation parameters. Larger systems with millions of particles can be simulated fully on the GPU and the result of the simulation exported as a MP4 video. 


![Screenshot of the GUI](/screenshot.png)

Both the SPH calculations and the neighbour search using parallel radix sort are implemented on the GPU, keeping the entire data on the device and only transferring data back for visualization after every 60th of a second of simulated time, maximizing GPU usage and avoiding bandwith bottlenecks.




To run and build the project, use:
```
cargo run -r
cargo build -r
```
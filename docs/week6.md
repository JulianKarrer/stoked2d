---
layout: page
title: Milestone
permalink: /milestone/
nav_order: 7
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Features Implemented
- **Solvers**
  - EOS SPH Solver
  - EOS with Splitting
  - Iterative EOS with Splitting
    - Different Equations of State
  - IISPH (2013, Notes)
- **Datastructures**
  - Index search - regular grid
    - XYZ, Morton, Hilbert curves
  - GPU version with parallel radix sort and reductions
- **Experiments**
  - Oscillation frequencies over stiffness
  - Error over stiffness
  - Stability over viscosity, timestep size
  - Aliasing and stability improvements with jitter
- **Initialization**
  - Oblique and regular grid initialization
  - Jittered initialization
    - Non-uniform mass, uniform density solver (-> influence on viscosity?)
- **GUI/Visualization**
  - Off-thread, GPU accelerated rendering
  - Atomic parameters
  - SPH "number density" using OpenGL quads and alpha blending
- **Miscellanous**
  - arbitrary non-uniform boudnary handling (-> tests?)
  - Viscosity-based adhesion
  - GPU implementation of EOS SPH


# IISPH v2
<div style="display: flex;">
  <video style="width:50%;" loop muted autoplay controls>
    <source src="{{ '/assets/week6/notes_col.webm' | relative_url }}" type="video/webm">
  </video>
  <video style="width:50%;" loop muted autoplay controls>
    <source src="{{ '/assets/week6/2013_col.webm' | relative_url }}" type="video/webm">
  </video>
</div>
- [2013 Paper](https://cg.informatik.uni-freiburg.de/publications/2013_TVCG_IISPH.pdf){:target="_blank"} version more numerically stable than Notes version?

### Improving Robustness
- minimum/maximum timestep relevant (stability vs. shock problems)
- higher $$\epsilon$$ for $$A_{ii} \neq 0$$
- maximum compression over average compression as convergence criterion (but higher iteration count)
- lower $$\omega$$
- higher (minimum) iteration count
- jittered boundaries and spacing not a multiple of $$h$$ - less porous

### Observations
- iteration count coupled to compression:

<div style="display: flex;">
    {% include plot_graphs_1718272458.html %}
</div>


### Stability
- problems seem largely related to boundary quality
- occur mostly at the start (better equilibration? remove outliers?)
- also, velocity field is noisy:

<a href="{{ '/assets/week6/vel_field.jpg' | relative_url }}" rel="noopener noreferrer" target="_blank">
  <img src="{{ '/assets/week6/vel_field.jpg'| relative_url }}" alt="Visible noise in the velocity field using density invariance as IISPH source term" style="width:30%"/>
</a> 

### Improvements?
  - other solver?
  - regularization?
  - stochastic behaviour?
  - preconditioning?
  - other initialization?
  - better parameters $$\epsilon, \omega, \text{miniter}, \dots$$



### Other Questions
- meaning of diagonally dominant and non-zero entries
- [micropolar vorticity](https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf){:target="_blank"} model / use vorticity to integrate on curve?

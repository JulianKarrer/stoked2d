---
layout: page
title: Testing Kernels and Datastructures
permalink: /week1a/
nav_order: 0
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Testing the Kernel Function


- Kernel properties are tested
  - Integrals are implemented with Monte Carlo, $$N=10^{6}$$
  - Accepted error is $$\epsilon = 10^{-3}$$
  - All other tests are run $$N$$ times with random $$\mathbf{\vec{x_i}, \vec{x_j}}$$

  $$\begin{align}

  &W(\lVert \mathbf{\vec{x_i} - \vec{x_j}} \rVert, \kappa)  > 0&& \text{Positivity}\\

  \forall r>\kappa:\quad &W(r, \kappa)  = 0,\quad \nabla W(r, \kappa)  = 0&&\text{Compact support}\\

  &W_{ij} = W_{ji}&& \text{Symmetry}\\

  &\nabla W_{ij} = \nabla W_{ji} &&\text{Derivative Antisymmetry}\\

  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa)\,dV = 1&&\text{Kernel Integral}\\

  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&\nabla W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa)\,dV = \mathbf{\vec{0}}&&\text{Kernel Derivative Integral}\\


  &\nabla W_{ii} = 0 &&\text{Kernel Derivative at Zero is Zero}\\

  \end{align}$$

- For ideal sampling, where $$\kappa = 2h$$, $$h$$ is the particle spacing a regular grid is used:

  $$\begin{align}
  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa) = \frac{1}{h^2}&&\text{Integral is Density}\\

  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&\nabla W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa) = \mathbf{\vec{0}}&&\text{Derivative Integral is Zero}\\
  \end{align}$$


- One test fails? For ideal sampling:

  $$\begin{align}
  \sum_j \left(\mathbf{(\vec{x}_i - \vec{x}_j)} \odot \nabla W_{ij} \right) = -\frac{1}{V} \cdot \mathbf{\vec{1}}
  \end{align} $$


- All other tests are successful:

  ![Screenshot of the Kernel test]({{'/assets/week1/tests_screenshot.png' | relative_url}})


## Use of alternative Kernels

- ✅ [2D Cubic Spline Kernel](https://cg.informatik.uni-freiburg.de/course_notes/sim_03_particleFluids.pdf){:target="_blank"} for $$\kappa = 2h,\, \alpha = \frac{5}{14 \pi h^2}, \, q=\lVert\mathbf{\vec{x}}_{ij}\rVert$$ :
  
  $$W(\mathbf{\vec{x}}_{ij}) := \alpha \begin{cases}
  (2-q)^3 - 4(1-q)^3& 0 \leq q < 1\\
  (2-q)^3 & 1 \leq q < 2\\
  0       & q \ge 2
\end{cases}$$

  $$\nabla W(\mathbf{\vec{x}}_{ij}) := \alpha \frac{\mathbf{\vec{x}}_{ij}}{\lVert\mathbf{\vec{x}}_{ij}\rVert h}\begin{cases}
    -3(2-q)^2 +12 (1-q)^2& 0 \leq q < 1\\
    -3(2-q)^2 & 1 \leq q < 2\\
    0       & q \ge 2
    \end{cases}$$


# Improving the Acceleration Datatructure
 <span style="color:lightgreen">+400 FPS@N=4k / +60%</span>
- Filter by $$\lVert \mathbf{\vec{x_i} - \vec{x_j}} \rVert \le \kappa$$ when building neighbour lists instead of saving supersets with all particles in neighbouring grid cells. 
  <span style="color:lightgreen">~15%@N=4k</span>
- Binary search directly in particle handles, don't build a compact list of filled cells like suggested in [Compressed Neighbor Lists for SPH](https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf){:target="_blank"} (???). 
  <span style="color:lightgreen">~30%@N=4k</span>
- Use an XYZ curve that is faster to compute than a Morton Code (?)
  <span style="color:lightgreen">~15%@N=4k</span>

# Testing the Acceleration Datatructure

For $$\kappa \in [0.9, 1.0, 1.1], \, N=5000$$ particles are randomly distributed in $$[-10;10]$$ and the neighbour sets for each compared to the ones computed per brute force in $$\mathcal{O}(n^2)$$. After 10 repetitions, success is reported:


![Screenshot of the Datastructure Test]({{'/assets/week1/datastructure_test_screenshot.png' | relative_url}})

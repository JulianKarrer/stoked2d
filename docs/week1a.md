---
layout: page
title: Testing Kernels and Datastructures
permalink: /week1a/
nav_order: 0
---



# Implemented Kernel Functions

- [2D Cubic Spline Kernel](https://cg.informatik.uni-freiburg.de/course_notes/sim_03_particleFluids.pdf){:target="_blank"} for $$\kappa = 2h,\, \alpha = \frac{5}{14 \pi h^2}, \, r=\lVert\mathbf{\vec{x}}_{ij}\rVert, \, q=\frac{r}{h}$$ :
  
  $$W_{\text{spline}}(\mathbf{\vec{x}}_{ij}) := \alpha \begin{cases}
  (2-q)^3 - 4(1-q)^3& 0 \leq q < 1\\
  (2-q)^3 & 1 \leq q < 2\\
  0       & q \ge 2
\end{cases}$$

  $$\nabla W_{\text{spline}}(\mathbf{\vec{x}}_{ij}) := \alpha \frac{\mathbf{\vec{x}}_{ij}}{\lVert\mathbf{\vec{x}}_{ij}\rVert h}\begin{cases}
    -3(2-q)^2 +12 (1-q)^2& 0 \leq q < 1\\
    -3(2-q)^2 & 1 \leq q < 2\\
    0       & q \ge 2
    \end{cases}$$

  - Vanishing gradients as $$r\to 0$$ problematic for pressure computation? 


- [2D Spiky Kernel](https://cs418.cs.illinois.edu/website/text/sph.html#kernel-functions){:target="_blank"} 
  suggested by [Müller, Charypar, Gross](https://matthias-research.github.io/pages/publications/sca03.pdf){:target="_blank"} 
  after [Desbrun, Cani](http://www.geometry.caltech.edu/pubs/DC_EW96.pdf){:target="_blank"} 
  [normalized for 2D](https://www.diva-portal.org/smash/get/diva2:573583/FULLTEXT01.pdf){:target="_blank"}    for $$\kappa = 2h,\, \alpha = -\frac{10}{\pi\kappa^5}, \, r=\lVert\mathbf{\vec{x}}_{ij}\rVert$$ :
  
  $$W_{\text{spiky}}(\mathbf{\vec{x}}_{ij}) := \alpha \frac{\mathbf{\vec{x}}_{ij}}{\lVert\mathbf{\vec{x}}_{ij}\rVert}\begin{cases}
    (\kappa-r)^3& 0 \leq r \leq  \kappa\\
    0 & \text{otherwise}
    \end{cases}$$

  $$\nabla W_{\text{spiky}}(\mathbf{\vec{x}}_{ij}) := \alpha \frac{\mathbf{\vec{x}}_{ij}}{\lVert\mathbf{\vec{x}}_{ij}\rVert}\begin{cases}
    (\kappa-r)^2& 0 < r \leq  \kappa\\
    0 & \text{otherwise}
    \end{cases}$$

    - Unrealistic viscous behaviour?
    - Wrong density estimation for ideal sampling?

- [2D Double Cosine Kernel](https://www.sciencedirect.com/science/article/pii/S0307904X13007920?ref=pdf_download&fr=RR-2&rr=87c843287c471e6e){:target="_blank"} 
  proposed by [Yang, Peng,Liu]
  for $$\kappa = 2h,\, \alpha = \frac{\pi}{(3\pi^2-16)(\kappa)^2}, \, r=\lVert\mathbf{\vec{x}}_{ij}\rVert, \, s=\frac{r}{h}$$:
  
    $$W_{\text{cos}}(\mathbf{\vec{x}}_{ij}) := \alpha \begin{cases}
    4\cos(\frac{\pi}{2}s)+\cos(\pi s)+3& 0 \leq s \leq  2\\
    0 & \text{otherwise}
    \end{cases}$$


    $$\nabla W_{\text{cos}}(\mathbf{\vec{x}}_{ij}) := \alpha \frac{\mathbf{\vec{x}}_{ij}}{\lVert\mathbf{\vec{x}}_{ij}\rVert}\begin{cases}
    -2\frac{\tau}{\kappa}\sin(\frac{\pi}{2}s)-\frac{\tau}{\kappa}\sin(\pi s)& 0 \leq s \leq  2\\
    0 & \text{otherwise}
    \end{cases}$$


# Kernel Function Plots
### Kernel Functions for $$\kappa = 2h,\, h=1$$
<div style="display: flex">
  {% include plot_kernels.html %}
</div>

- Surprisingly equal integral? Projection to 1D may be at fault, outer regions weigh quadratically more in 2D integral

### Magnitude of Kernel Gradient over $$\mathbf{\vec{x}}_{ij}$$ for Cubic Spline and Spiky Kernel:
<div style="display: flex;">
  <div style="width:50%">
  {% include plot_kernel_derivatives_cubic_gauss_spline.html %}
  </div>
  <div style="width:50%">
  {% include plot_kernel_derivatives_spiky_kernel.html %}
  </div>
</div>


# Testing the Kernel Function

- Kernel properties are tested
  - Integrals are implemented with Monte Carlo, $$N=10^{8}$$ on $$\Omega=[-2.1h; 2.1h]$$ 
  - Accepted error is $$\epsilon = 10^{-3}$$ 
  - They are checked again with Riemann sums of $$n=10^4,\, \epsilon=10^{-9}$$
  - All other tests are run $$N$$ times with random $$\mathbf{\vec{x_i}, \vec{x_j}}$$

  $$\begin{align}

  &W(\lVert \mathbf{\vec{x_i} - \vec{x_j}} \rVert, \kappa)  > 0&& \text{Positivity}\\

  \forall r>\kappa:\quad &W(r, \kappa)  = 0,\quad \nabla W(r, \kappa)  = 0&&\text{Compact support}\\

  &W_{ij} = W_{ji}&& \text{Symmetry}\\

  &\nabla W_{ij} = \nabla W_{ji} &&\text{Gradient Antisymmetry}\\

  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa)\,dV = 1&&\text{Kernel Integral}\\

  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&\nabla W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa)\,dV = \mathbf{\vec{0}}&&\text{Kernel Gradient Integral}\\


  &\nabla W_{ii} = 0 &&\text{Kernel Gradient at Zero is Zero}\\

  \end{align}$$

- For ideal sampling, where $$\kappa = 2h$$, $$h$$ is the particle spacing a regular grid is used:

  $$\begin{align}
  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa) = \frac{1}{h^2}&&\text{Integral is Density}\\

  \int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}&\nabla W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa) = \mathbf{\vec{0}}&&\text{Gradient Integral is Zero}\\

  \sum_j &\left(\mathbf{(\vec{x}_i - \vec{x}_j)} \odot \nabla W_{ij} \right) = -\frac{1}{V} \cdot \mathbf{\vec{1}}&&\text{Density from Gradient}
  \end{align}$$

- On a $$100\times 100$$ regular grid and a random grid, the consistency of each $$\nabla W$$ is tested by comparing to a $$\mathcal{O}(\Delta x^4)$$ [central difference](https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf){:target="_blank"} of $$W$$ with an accepted error of $$1\%$$ 
  - ❌ this fails for Cubic Spline for small $$h$$ - numerical problem at discontinuity? 
  - ❌ always fails for Spiky kernel - smoothness too low?
  - ✅ always passes for Double Cosine kernel

## Failed Tests

❌ Ideal Sampling: $$\sum_j \left(\mathbf{(\vec{x}_i - \vec{x}_j)} \odot \nabla W_{ij} \right) = -\frac{1}{V} \cdot \mathbf{\vec{1}}$$ always fails

| Kernel    | Failed Tests |
| -------- | ------- |
| Cubic Spline  ❓  | $$W$$ and $$\nabla W$$: consistency with finite difference? |
| Spiky Kernel ❌ | Ideal Sampling: $$\int_{\mathbf{x_j}\in[-\kappa;\kappa]^2}W(\lVert \mathbf{\vec{0} - \vec{x_j}} \rVert, \kappa) = \frac{1}{h^2}$$ off by +27.36%|
| Double Cosine  ✅  | None |

# Stability as an interval over stiffness $$k$$
- Viscosity: Double Cosine Kernel
- Pressure and Density Computations: Cubic Spline or Double Cosine
- Instability for $$\nu=0.02, h=0.03, \lambda=0.1$$ is defined as:
  - when a watter column of height $$3$$ in a $$3\times 3$$ box has $$\epsilon_{\rho}>5\%$$ density deviation
  - or starts oscillating at the bottom boundary
- Results:
  - Double Cosine stable for $$k\in[580; 2400]$$
  - Cubic Spline stable for $$k\in[18200; 93000]$$


Double Cosine Kernel for $$k=580$$ and $$k=2000$$ (unstable on the left, stable on the right)
<div style="display: flex;">
  <div style="width:50%">
    <video style="width:100%;" loop muted autoplay>
      <source src="{{ '/assets/week1/w1_double_cos_col_unstable.webm' | relative_url }}" type="video/webm">
    </video>
  </div>
  <div style="width:50%">
    <video style="width:100%;" loop muted autoplay>
      <source src="{{ '/assets/week1/w1_double_cos_col_stable.webm' | relative_url }}" type="video/webm">
    </video>
  </div>
</div>

# Testing Energy Conservation
The Hamiltonian is for now defined as :

  $$\mathcal{H} = \sum_i^N \left( m_i \mathbf{\vec{g}} \cdot \mathbf{\vec{x}}_i + \frac{1}{2} m_i \mathbf{\vec{v}}_i \cdot \mathbf{\vec{v}}_i\right)$$

And normalized as $$ \mathcal{H}_{min} = mgh_{min}$$ and $$\mathcal{H}_{norm} = \frac{\mathcal{H}-\mathcal{H}_{min} }{\lVert \mathcal{H}_{min} \rVert} $$ for plotting

but appears to be missing a term that is proportional to the compression of the fluid ([internal energy potential?](https://www.sciencedirect.com/science/article/pii/S0021999105001865){:target="_blank"}) as indicated by oscillations in a resting water column:

{% include plot_den_ham_1714640585.html %}

- Should $$U =\sum_i^N p_i V_i = p_i\frac{m_i}{\rho_i}$$ be used to compensate? Can the oscillation even be avoided?

# Improving the Acceleration Datatructure
 <span style="color:lightgreen">+400 FPS@N=4k / +60%</span>
- Filter by $$\lVert \mathbf{\vec{x_i} - \vec{x_j}} \rVert \le \kappa$$ when building neighbour lists instead of saving supersets with all particles in neighbouring grid cells. 
  <span style="color:lightgreen">~15%@N=4k</span>
- Binary search directly in particle handles, don't build a compact list of filled cells like suggested in [Compressed Neighbor Lists for SPH](https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf){:target="_blank"} (???). 
  <span style="color:lightgreen">~30%@N=4k</span>
- Use an XYZ curve that is faster to compute than a Morton Code (?)
  <span style="color:lightgreen">~15%@N=4k</span>

# Testing the Acceleration Datatructure

For $$\kappa \in [0.9, 1.0, 1.1], \, N=5000$$ particles are randomly distributed in $$[-10;10]$$ and the neighbour sets for each compared to the ones computed per brute force in $$\mathcal{O}(n^2)$$. This is repeated 10 times. ✅ 


![Screenshot of the Datastructure Test]({{'/assets/week1/datastructure_test_screenshot.png' | relative_url}})

# Current State of the Project
<div style="display: flex;">
  <div style="width:100%">
    <video style="width:100%;" loop muted autoplay>
      <source src="{{ '/assets/week1/w1_dam_break.webm' | relative_url }}" type="video/webm">
    </video>
  </div>
</div>

{% include plot_den_ham_1714580095.html %}


# Questions
- What does $$\sum_j \left(\mathbf{(\vec{x}_i - \vec{x}_j)} \odot \nabla W_{ij} \right) = -\frac{1}{V} \cdot \mathbf{\vec{1}}$$ actually mean? Why does it always fail for ideal sampling?
- Why did the test for the spiky kernel fail?
- Why do the stable parameters differ so wildly from kernel to kernel?
- Why did not producing a compact filled cell list improve neighbour search performance? (size of the system?)
- How can energy conservation accurately be tracked?

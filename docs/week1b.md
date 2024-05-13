---
layout: page
title: Initialization and Boundary Handling
permalink: /week2/
nav_order: 1
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Jittered Initialization

Introducing even just $$0.01h$$ as random jitter when initializing particles can improve stability at the beginning of the simulation:

<div style="display: flex;">
  <div style="width:50%">
    <video style="width:100%;" loop muted autoplay controls>
      <source src="{{ '/assets/week2/jitter0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div style="width:50%">
    <video style="width:100%;" loop muted autoplay controls>
      <source src="{{ '/assets/week2/jitter1p.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

*(example for $$h=0.05, k=2300, \nu=0.03$$ water column height of $$1$$, left is unjittered)*

- not very predictable for low sampling resolutions
- the absolute and the squared density error $$E_{abs}, E_{sqrd}$$ of a simulation are defined as:

$$\begin{align}
E_{abs} &:= \int_0^{10s} \left\lvert \frac{\rho_{avg}}{\rho_{0}}-1 \right\rvert\\
E_{sqrd} &:= \int_0^{10s} \left(\frac{\rho_{avg}}{\rho_{0}}-1 \right)^2\\
\end{align}$$

- plotted against amount of initial random jitter in units of the particle spacing $$h$$ for $$h=0.05, k=2000, \nu=0.03$$:

![Plot of Density Error over initial Jitter]({{'/assets/week2/plot_jitter_error.png' | relative_url}})

- in larger systems, small ($$1\%$$) jitter effectively prevents aliasing artefacts:

<div style="display: flex;">
  <div style="width:50%">
    <video style="width:100%;" loop muted autoplay>
      <source src="{{ '/assets/week2/dam_jitter.webm' | relative_url }}" type="video/webm">
    </video>
  </div>
  <div style="width:50%">
    <video style="width:100%;" loop muted autoplay>
      <source src="{{ '/assets/week2/dam_no_jitter.webm' | relative_url }}" type="video/webm">
    </video>
  </div>
</div>
*(video slowed down x8 with motion interpolation, left is jittered $$N=90000, h=0.01, \lambda = 0.1, k=2000, \nu=0.01$$)*

# Towards a single non-uniform layer

- Following [the 2019 SPH Tutorial](https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf){:target="_blank"}, we define for a particle ideally surrounded by fluid on a single layer of boundary particles:
  - $$\begin{align}
  \gamma_1 &:= \frac{V_i - \sum_{i_f}W_{i,i_f}}{\sum_{i_b} W_{i, i_b}} = \frac{\frac{1}{h^2} - \sum_{i_f}W_{i,i_f}}{\sum_{i_b} W_{i, i_b}}\\
  \\
  \gamma_2 &:= \frac{\sum_{i_f}\nabla W_{i,i_f} \cdot \sum_{i_b} W_{i, i_b}}{\sum_{i_b} W_{i, i_b} \cdot \sum_{i_b} W_{i, i_b}}
  \end{align}$$

- This turns out to be $$\gamma_1, \gamma_2 = 1$$ for a Cubic Spline with $$h=2$$ in 2D
- Higher $$\gamma_2$$ decreases stability
- Higher $$\gamma_1$$ seems to not decrease stability but improve impenetrability

# State of the Project



 <video style="width:100%;" loop muted autoplay controls>
    <source src="{{ '/assets/week2/one_layer_dambreak.mp4' | relative_url }}" type="video/mp4">
  </video>
$$$$
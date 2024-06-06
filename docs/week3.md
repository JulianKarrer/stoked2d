---
layout: page
title: Basic Adhesion
permalink: /basic-adhesion/
nav_order: 2
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Adhesion through viscosity
*($$j$$ are fluid neighbours, $$k$$ are boundary neighbours)*
- previously used viscosity: 
  - $$a_{i,vis} = 2\nu \sum_{j} 
    \frac{m_j}{\rho_{j}} 
    \frac{\vec{v}_{ij} \cdot \vec{x}_{ij}} \nabla W_{ij}
      {\lVert \vec{x}_{ij}\rVert^2 + 0.01h^2}$$
- [correct for dimensionality](https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf){:target="_blank"}: factor $$2(\text{dimensions}+2)$$ instead of $$2$$ ?
- introduce $$\nu_2$$ 
  - projected on boundary surface $$\hat{\vec{n}}\Longrightarrow$$ adhesion without tangential friction
  - $$\hat{\vec{n}} = \begin{cases}
    \frac{\sum_k \nabla W_{ik}}{\lVert\sum_k \nabla W_{ik}\rVert} & \lVert\sum_k \nabla W_{ik}\rVert>0\\
    \vec{0} & \text{otherwise}\\
    \end{cases} $$
  - $$a_{i,vis} = 2(\text{dimensions}+2)\left(
    \nu_1 \sum_{j} 
    \frac{m_j}{\rho_{j}} 
    \frac{\vec{v}_{ij} \cdot \vec{x}_{ij}}
      {\lVert \vec{x}_{ij}\rVert^2 + 0.01h^2} \nabla W_{ij} + 
    \left(\nu_2 \sum_{k} 
    \frac{m_k}{\rho_{0}} 
    \frac{\vec{v}_i \cdot \vec{x}_{ik}}
      {\lVert \vec{x}_{ik}\rVert^2 + 0.01h^2} \nabla W_{ik}\right) \cdot \hat{\vec{n}} \odot \hat{\vec{n}}
    \right)$$




- no adhesion, $$\nu_1 = 0.002, \nu_2=0$$:
  - <video style="width:100%;" loop muted autoplay controls>
      <source src="{{ '/assets/week3/visc_no_adhesion.mp4' | relative_url }}" type="video/mp4">
    </video>
- adhesion, $$\nu_1 = 0.001, \nu_2=0.020$$ *lower viscosity possible without instability!*:
  - <video style="width:100%;" loop muted autoplay controls>
      <source src="{{ '/assets/week3/visc_adhesion.mp4' | relative_url }}" type="video/mp4">
    </video>


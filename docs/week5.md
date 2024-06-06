---
layout: page
title: Implicit Incompressible SPH
permalink: /iisph/
nav_order: 6
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Implicit Incompressible SPH

WORKS! (?)

<div style="display: flex; margin-bottom: 50px;">
  <video style="width:100%;" loop muted autoplay controls>
    <source src="{{ '/assets/week5/iisph-dambreak.webm' | relative_url }}" type="video/webm">
  </video>
</div>

<div style="display: flex; margin-bottom: 50px;">
  <video style="width:100%;" loop muted autoplay controls>
    <source src="{{ '/assets/week5/iisph-slide-crop.webm' | relative_url }}" type="video/webm">
  </video>
</div>

<div style="display: flex; margin-bottom: 50px;">
  <video style="width:100%;" loop muted autoplay controls>
    <source src="{{ '/assets/week5/iisph-poiseuille-sm.webm' | relative_url }}" type="video/mp4">
  </video>
</div>

### Sources 
- IISPH PDF, 
- [2013 Paper](https://cg.informatik.uni-freiburg.de/publications/2013_TVCG_IISPH.pdf){:target="_blank"} (Source)
- [2019 SPH Tutorial](https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf){:target="_blank"} (Error in equation 49? $$W_{ij}$$ or $$W_{ji}$$ in diagonal element?)
- [SPlisHSPHlasH Source](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/IISPH/TimeStepIISPH.cpp){:target="_blank"} (implements [2013 Paper](https://cg.informatik.uni-freiburg.de/publications/2013_TVCG_IISPH.pdf){:target="_blank"} pretty much 1:1 in C++)

# Algorithm

*indices $$j$$ are fluid neighbours, $$k$$ boundary neighbours, $$n$$ neighbours of neighbours*

compute non-pressure accelerations (same as before)
- $$\rho_i = \sum_j W_{ij} m_j + \sum_k W_{ik} m_k$$
- $$\vec{a}^{vis}_i = 2\nu_1(d+2) \sum_j \frac{m_j}{\rho_j} \frac{\vec{v}_{i,j}\cdot \vec{x}_{ij}}{\lVert\vec{x}_{ij}\rVert^2 + 0.01h^2} \nabla W_{ij}$$
- $$\vec{a}^{adh}_i = 2\nu_2(d+2) \sum_k \frac{m_k}{\rho_0} \frac{\vec{v}_{i}\cdot \vec{x}_{ik}}{\lVert\vec{x}_{ik}\rVert^2 + 0.01h^2} \nabla W_{ik}$$
- $$\vec{a}^{\text{non-pressure}}_i = \vec{a}^{vis}_i + \vec{a}^{adh}_i + \left(\begin{array}{c} 0\\-9.81\end{array}\right)$$
- $$ \Delta t = \min\left(0.001, \lambda \frac{h}{v_{max}}\right)$$

predict advection
- $$\vec{v}^{adv}_i = \Delta t \vec{a}^{\text{non-pressure}}_i$$

precompute values and prepare pressure
- $$\vec{d}_{ii} = - \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij} - \sum_k \frac{m_k}{\rho_i^2} \nabla W_{ij}$$
- $$\rho^{adv}_i = \rho_i + \Delta t \sum_j m_j \left(\vec{v}^{adv}_i- \vec{v}^{adv}_j\right) \cdot \nabla W_{ij} + \Delta t \sum_k m_k \left(\vec{v}^{adv}_i- \vec{v}_k\right) \cdot \nabla W_{ik}$$
- $$a_{ii} = \sum_{j} m_j (\vec{d}_{ii} - \underbrace{\frac{m_i}{\rho_i^2}\nabla W_{ij}}_{\vec{d}_{ji}})\cdot \nabla W_{ij} + \sum_k m_k (\vec{d_{ii} - \underbrace{\frac{m_i}{\rho_i} \nabla W_{ik}}_{\vec{d}_{ki}}})\cdot\nabla W_{ik}$$
- $$p_i = \frac{1}{2}p_i$$

loop over $$l \in [0,\infty)$$ while $$(\rho_{err}^{avg} > 0.001 \rho_0) \lor (l<2)$$:

---
> - $$\vec{d}_{ij}p_j = -\sum_j \frac{m_j}{\rho_j^2}p_j\nabla W_{ij}$$
> - $$ \begin{align*} r_i &= \sum_j m_j \left(
  \vec{d}_{ij}p_j - \vec{d}_{jj} \cdot p_j - \underbrace{\left(\vec{d}_{j{n}}p_{n} 
    -\underbrace{\frac{m_i}{\rho_i^2}\nabla W_{ij}  p_i}_{\vec{d}_{ji} p_i}\right)}_{\sum_{n \neq i} \vec{d}_{j{n}}p_{n} }
\right)\cdot\nabla W_{ij}\\
&+ \sum_k m_k \vec{d}_{ij}p_j \cdot \nabla W_{ik}
\end{align*}
$$
> - use $$\omega = 0.5$$
> - $$p_i = \begin{cases}
\max\left(0, (1-\omega)p_i +\frac{\omega}{a_{ii}\Delta t} (\rho_0 - \rho_i^{adv} - \Delta t^2 r_i )\right)  & \text{if } \lvert a_{ii} \Delta t \rvert > 10^{-9}\\
0 & \text{otherwise}\\
\end{cases}$$
> - $$\rho_{err}^{avg} = \frac{1}{N} \sum_i \begin{cases}
\rho_0 \left( \Delta t^2 (a_{ii} p_i + r_i) - (\rho_0 - \rho_i^{adv}) \right)
\end{cases}$$

---

calculate pressure accelerations and integrate them
- $$\vec{a}^{p}_i = - \sum_j m_j \left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}\right) \nabla W_{ij} - \gamma_2 \left(\frac{p_i}{\rho_i^2} + \frac{p_i}{\rho_0^2}\right) \sum_k m_k \nabla W_{ik}$$
- $$\vec{v}_i = \vec{v}^{adv}_i + \Delta t \vec{a}^{p}_i $$
- $$\vec{x}_i = \vec{x}_i + \Delta t \vec{v}_i $$




# Problems and Improvements
### Problem for higher resolutions: artefacts at start
  -  initialize and normalize kernels for regular hexagonal lattice instead of square or oblique hexagonal lattice to account for [denser packing](https://archive.lib.msu.edu/crcmath/math/math/s/s561.htm){:target="_blank"} ?
<div style="display: flex;">
  <video style="width:100%;" loop muted autoplay controls>
    <source src="{{ '/assets/week5/iisph-dambreak-highres.webm' | relative_url }}" type="video/webm">
  </video>
</div>


<a href="{{ '/assets/week5/00007.jpg' | relative_url }}" rel="noopener noreferrer" target="_blank">
  <img src="{{ '/assets/week5/00007.jpg' | relative_url }}" alt="Artefacts in IISPH solver for high particle count"/>
</a> 

### Problem: Divergence
- Might not converge for very high water column, pressure explosion:

<a href="{{ '/assets/week5/00000.jpg' | relative_url }}" rel="noopener noreferrer" target="_blank">
  <img src="{{ '/assets/week5/00000.jpg' | relative_url }}" alt="Explosion in IISPH solver for high particle count"/>
</a> 

- worse if $$\rho_k = \rho_0$$ instead of density mirroring?
- better for larger timesteps? (Division by $$\Delta t^2$$)

### Equlibrate initial mass
  - solve system for mass that creates rest density
  - iteratively measure density with same function as pressure solver uses
  - Solver: $$m_i^{l+1} = m_i^l \cdot \frac{\rho_0}{\rho_i^l(\vec{m}^l)}$$

  - more stable at start, can initialize closer to boundary
  - bonus benefit: less crystalline structure when combined with jitter
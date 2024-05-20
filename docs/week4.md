---
layout: page
title: Analyzing Parameters
permalink: /week4/
nav_order: 3
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Setting
- let $$N=1591, h=0.02, \nu_2 = 0$$
- water height: $$2.96m$$

- <p align="center">
    <img src="{{ '/assets/week3/setting_column_small.png' | relative_url }}" alt="Sublime's custom image" alt="Initial Setting of the water column"/>
  </p>


# Oscillation Frequency over Stiffness
- fixed $$\Delta t=\frac{1}{5000}$$
- time interval $$2s \le t \le 12s$$
- the signal is $$f(t) = \rho_{avg}(t) - \rho_0$$, which measures compression
- use Fourier analysis to find:
  - peak frequency of the oscillation
  - zero frequency (total density error)

## Results:
Total density error as a function of $$k$$:
{% include plot_ks_to_err_1716081442.html %}

Oscillation peak frequency as a function of $$k$$:
{% include plot_ks_to_freq_1716081442.html %}

## Interpretation:
- viscosity doesn't change the frequency or total error in this setting
- error decreases exponentially with $$k$$ (?)
- oscillation frequency increases with $$k$$
  - possible explanation: $$f \propto c$$ since [*the oscillations move at the speed of sound $$c$$*](https://link.springer.com/article/10.1007/s40571-022-00511-8){:target="_blank"}
  - Newton–Laplace equation:  $$c = \sqrt{\left(\frac{\partial P}{\partial \rho}\right)}= \sqrt{\frac{k}{\rho}}$$
  - explains $$\sqrt{x}$$-shaped curve


# Stability over Viscosity, Stiffness and Timestep Size
- time interval: $$10s \le t \le 20s$$
- time integral of kinetic energy (less is better): 
  - $$E_{kin, int} = \int_{10s}^{20s} \frac{1}{N}\sum_{i=1}^N \frac{1}{2} m_i (\vec{v}_i \cdot \vec{v}_i) \, dt$$
- peak average kinetic energy (less is better): 
  - $$E_{kin, peak} = \max_{10s \le t \le 20s} \frac{1}{N}\sum_{i=1}^N \frac{1}{2} m_i \left(\vec{v}_i(t) \cdot \vec{v}_i(t)\right) \, dt$$


## Fixed Timestep
- set $$\lambda = 0.1$$
- $$k$$ on the x-axis, $$\nu$$ on the y-axis

<div style="display: flex;">
  <div style="width:50%">
  $$E_{kin, int}$$
    {% include plot_stability_int_1716168583.html %}
  </div>
  <div style="width:50%">
  $$E_{kin, peak}$$
    {% include plot_stability_peak_1716168583.html %}
  </div>
</div>

## Fixed Viscosity
- set $$\nu = 0.001$$
- $$k$$ on the x-axis, $$\nu$$ on the y-axis

<div style="display: flex;">
  <div style="width:50%">
  $$E_{kin, int}$$
    {% include plot_stability_lambda_int_1716168583.html %}
  </div>
  <div style="width:50%">
  $$E_{kin, peak}$$
    {% include plot_stability_lambda_peak_1716168583.html %}
  </div>
</div>



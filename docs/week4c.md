---
layout: page
title: Better Rendering
permalink: /week4c/
nav_order: 4
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Better Rendering

## Problem
Rendering particles as circles:
- looks noisy
- is physically inaccurate/misleading
- creates aliasing (especially at smaller resolutions)
- clobbers quality when video compression is used

<a href="{{ '/assets/week4/00430_lowres.jpg' | relative_url }}" rel="noopener noreferrer" target="_blank">
  <img src="{{ '/assets/week4/00430_lowres.jpg' | relative_url }}" alt="Rendering with particles as circles"/>
</a> 

## Goal

- Use SPH to interpolate, then visualize
  - storing acceleration datastructure each frame takes too much memory
  - rendering each frame when simulating disallows responsive zooming/panning etc.
  - an SPH query per pixel is expensive

## Solution

- particles are quads with side length $$4h$$
- instanciate a quad at each particle location
- give each quad a white texture with SPH kernel weight in alpha channel
  - Cubic Spline Kernel for $$h=1$$ used: $$W(q) = \frac{5}{14\pi} \max(0, 2-q)^3 -4 \max(0, 1-q)^3$$
  - calculate pixel centre's distance from texture centre: $$d$$ 
  - normalize to $$\hat{d} \in [0;\sqrt{2}]$$
  - for $$h=1$$ in 2D the kernel support is $$2$$ and $$W(0)=0.45$$
  - use $$r=255, g=255, b=255, \alpha = 255*\left\lfloor\frac{W(2\hat{d})}{0.45}\right\rfloor$$

- tint the texture according to particle velocity/density/etc in a material

$$\Longrightarrow$$ abuse OpenGL to perform SPH sums!

<a href="{{ '/assets/week4/00430_highres.jpg' | relative_url }}" rel="noopener noreferrer" target="_blank">
  <img src="{{ '/assets/week4/00430_highres.jpg' | relative_url }}" alt="Rendering with particles as textured quads that approximate kernel sums"/>
</a> 
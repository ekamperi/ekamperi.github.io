---
layout: post
title:  Kernel density estimation
date:   2020-12-08
categories: math
tags: [Mathematica, mathematics, 'statistics']
---


$$
k(u)\text = \frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{u^2}{2}\right)\\
$$

$$
k_\text{scaled}(h, u) = \frac{1}{h} k\left(\frac{u}{h}\right)
$$

$$
\begin{align*}
\frac{1}{6} \left(\frac{e^{-\frac{1}{2} (x-9)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-7)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-4)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-3)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-2)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-1)^2}}{\sqrt{2 \pi }}\right)
\end{align*}
$$

<p align="center">
<img src="{{ site.url }}/kernel_density.gif" /> 
</p>

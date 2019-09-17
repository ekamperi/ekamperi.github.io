---
layout: post
title:  Arcsin transformation gone wrong
date:   2019-08-22
categories: math
tags: ['machine learning', math, 'neural netowrks']
---

Neglecting correlations or assuming independent variables yields a common formula among engineers and experimental scientists to calculate error propagation, the variance formula:[4]

Assuming that you have $$z = f(x, y, \ldots)$$:

$$
s_z = \sqrt{ \left(\frac{\partial f}{\partial x}\right)^2 s_x^2 + \left(\frac{\partial f}{\partial y} \right)^2 s_y^2 + \cdots}
$$

where $$s_z$$ represents the standard deviation of the function $$f$$, $$s_x$$ represents the standard deviation of $$x$$, $$s_y$$ represents the standard deviation of $$y$$, and so forth.

$$
z = \text{arcsin}(\sqrt{x})
$$

$$ \Delta z = \frac{dz}{d x} \Delta x \Rightarrow \Delta z = \frac{1}{2\sqrt{x (1-x)}} \Delta x
$$t

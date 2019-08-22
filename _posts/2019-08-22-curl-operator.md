---
layout: post
title:  The curl operator
date:   2019-08-22
categories: math
tags: math
---

Quoting the wikipedia definition of the [curl vector operator](https://en.wikipedia.org/wiki/Curl_(mathematics)):

> In vector calculus, the curl is a vector operator that describes the infinitesimal rotation of a vector field in three-dimensional Euclidean space. At every point in the field, the curl of that point is represented by a vector. The attributes of this vector (length and direction) characterize the rotation at that point.

The devil in this definiton lies in the word *infinitestimal*. I was under the impression that curl was related to the *macroscopic* rotation, but I couldn't be more wrong! Let me show what I mean. Consider the vector field defined by $$\mathbf{F}(x,y,z) = \left(-y/(x^2+y^2), x/(x^2+y^2), 0\right)$$. 

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/curl_operator.png">
</p>

By looking at these images my first reaction was that this field is most certainly a rotational one. Imagine my surprise when I actually did the math and my intuition proved to be completely wrong:

$$
\begin{align}
\nabla \times \mathbf{F}
&= \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\[5pt]
{\dfrac{\partial}{\partial x}} & {\dfrac{\partial}{\partial y}} & {\dfrac{\partial}{\partial z}} \\[10pt]
F_x & F_y & F_z \end{vmatrix}\\
&=
\left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right) \mathbf{i} + \left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x} \right) \mathbf{j} + \left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} \right) \mathbf{k}\\
&= (0 - 0)\mathbf{i} + (0 - 0) \mathbf{j} + \left[\frac{-x^2+y^2}{(x^2+y^2)^2} - \frac{-x^2+y^2}{(x^2+y^2)^2}\right] \mathbf{k} = \vec{0}
\end{align}
$$

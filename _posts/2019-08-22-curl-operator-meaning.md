---
layout: post
title:  The meaning of curl operator
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

By looking at these images my first reaction was that this field is most certainly a rotational one. I mean look at how "swirly" it is! Imagine my surprise when I actually did the math and my intuition proved to be completely wrong:

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

How can it be that this plot corresponds to an irrotational field? Well, it depends on which rotation you are referring to (macroscopic vs. microscopic or global vs. local). Imagine that this vector field describes the flow of water in a pool with a sink at the bottom that sucks the water out it. If we put a small ball on the surface of the water, then the ball may move in two distinct ways:

1. The general rotation of the flow around the z-axis (z-axis is perpendicular to your monitor) in the *counterclockwise* direction, along the direction of the stream lines.
2. Since the arrows of the field are longer the closer we are to the z-axis, the field tends to push the ball more strongly on the side closest to the z-axis, rather than the opposite side. The "differential" push on the two sides of the ball would tend to make it rotate in the *clockwise* direction.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/curl_rotation.png">
</p>

These two opposite effects may cancel out (as in our case) and then the curl is zero. The ball still moves inside the pool around the z-axis, but it doesn't rotate *around itself*, which is what the curl operator measures.

By now, it shouldn't come as a surprise that the curl of a vector field calculated at some point $$O$$, is related to the angular velocity of a rotating object with its center fixed at $$O$$. Let's do the math!

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/angular_velocity_curl.png">
</p>

Let us calculate the curl of $$\mathbf{v}$$:

$$
\nabla \times \mathbf{v}
= \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\[5pt]
{\dfrac{\partial}{\partial x}} & {\dfrac{\partial}{\partial y}} & {\dfrac{\partial}{\partial z}} \\[10pt]
\nu_x & \nu_y & \nu_z \end{vmatrix}
$$

The $$x$$ component of $$\nabla \times \mathbf{v}$$ is:

$$
\left( \nabla \times \mathbf{v} \right)_x = {\partial_y \nu_z} - {\partial_z \nu_y}
$$

Recall though that $$\mathbf{v} = \mathbf{\omega} \times \mathbf{r} \Rightarrow \nu_z = \omega_x y-\omega_y x $$ and similarly $$\nu_y = \omega_x z - \omega_z x$$. Therefore (for brevity we write $$\partial_x$$ instead of $$\frac{\partial}{\partial_x}$$):

$$
\begin{align}
\left( \nabla \times \mathbf{v} \right)_x
&= {\partial_y \nu_z} - {\partial_z \nu_y}\\
&= \partial_y (\omega_x y - \omega_y x)- \partial_z (\omega_x z - \omega_z x)\\
&= \omega_x + \omega_x = 2\omega_x
\end{align}
$$

Similarly it is $$\left( \nabla \times \mathbf{v} \right)_y = 2 \omega_y$$ and $$\left( \nabla \times \mathbf{v} \right)_z = 2\omega_z$$. Therefore:

$$
\nabla \times \mathbf{v} = 2 \mathbf{\omega}
$$

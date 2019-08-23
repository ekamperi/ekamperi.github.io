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

How can it be that this vector plot corresponds to an irrotational field? It depends on which rotation you are referring to. Imagine this is a water pool and there's a sink that sucks the water. If we put a small ball on the surface of the water, then the ball may  move in two distinct ways:

1. The general rotation of the flow around the z-axis (z-axis is perpendicular to your monitor) in the *counterclockwise* direction, along the direction of the stream lines. This is similar to the motion of the earth around the sun.
2. Since the arrows of the field are longer the closer we are to the z-axis, the field tends to push the ball more strongly on the side closest to the z-axis, rather than the opposite side. The "differential" push on the two sides of the ball would tend to make it rotate in the *clockwise* direction. This motion is similar to earth spinning around its own axis, *while* it also moves around the sun.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/curl_rotation.png">
</p>

These two opposite effects may cancel out (as in our case) and then the curl is zero. The ball still moves inside the pool around the z-axis, but it doesn't rotate *around itself*, which is what the curl operator measures.

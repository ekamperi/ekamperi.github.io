---
layout: post
title:  "Norms and machine learning"
date:   2019-10-19
categories: [machine learning]
tags: ['mathematics', 'neural networks']
---

### Introduction
A vector space, known also as a linear space, is a collection of objects (the vectors),
which may be added together and multiplied by some scalar (a number). In machine learning
we use vectors all the time, for instance we have *feature vectors* that are collections
of numbers that we group them together when representing an object. In image processing,
the features' values may be the pixels of the image, so assuming a 128 x 128 grayscale image,
you get a 16384 vector. Feature vectors are equivalent to the vectors of independent variables
in linear regression, but usually are much larger.

Informally, a norm is a function that accepts as input a vector from our vector space V and spits out a real
number that tells us how big a vector is. In order for a function to qualify as a norm,
it must first fulfill some properties, so that the results of this metrization process kind of
"make sense". These properties are the following. For all $$u, v$$ in the vector space $$V$$
and $$\alpha$$ in $$\mathbb{R}$$:

* $$\|v\| \ge 0$$ and $$\|v\| = 0 \Leftrightarrow v = 0$$ (positive/definite)
* $$\| \alpha v \| = \|\alpha\| \| v \|$$ (absolutely scalable)
* $$\|u+v\| \le \|u\|+\|v\|$$ (Triangle inequality)

### The $$L^p$$ norm
One of the most widely known family of norms is the $$L^p$$ norm, which is defined as:

$$
\ell_p = \left( \sum_{i=1}^N |x_i|^p \right)^2, \text{for } p \ge 1
$$

For $$p = 1$$ you get, $$\ell_1 = \vert x_1 \vert + \vert x_2 \vert + \ldots + \vert x_n \vert$$

For $$p = 2$$, $$\ell_2 = \sqrt{x_1^2 + x_2^2 + \ldots + x_n^2}$$

For $$p = 3$$, $$\ell_3 = \sqrt[3]{x_1^3 + x_2^3 + \ldots + x_n^3}$$

For $$p \to \infty$$, $$\ell_\infty = \max_i (\vert x_1 \vert, \vert x_2 \vert, \ldots, \vert x_n \vert)$$

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/lp_norms_2d.png">
</p>

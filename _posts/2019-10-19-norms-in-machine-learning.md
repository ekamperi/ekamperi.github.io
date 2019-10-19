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
we use vectors all the time. Here are some examples:

* *Feature vectors* that are collections of numbers that we group them together when representing
an object. In image processing, the features' values may be the pixels of the image, so assuming a
$$128 \times 128$$ grayscale image, we get a $$16384$$ long vector. Feature vectors are equivalent
to the vectors of independent variables (the $$x$$-s) in linear regression, but usually are much larger.
* The *output* of a machine learning model, say a neural network that is trained to identify hand-written
digits, may be represented as a vector, e.g. $$y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]$$ for representing $$2$$
as the correct output.
* The *loss function*, i.e. the function that tells us how good or how bad are predictions are, is also
the norm of a particular vector space. For example, the mean squared error is $$\text{MSE} = \frac{1}{N} \sum_i (y_\text{true} - y_\text{predicted})^2$$, which (as we shall see) is the $$\ell_2$$ norm of vectors $$y_\text{true} - y_\text{predicted}$$.

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

For $$p = 3$$, $$\ell_3 = \sqrt[3]{\vert x_1 \vert ^3 + \vert x_2 \vert ^3 + \ldots + \vert x_n \vert ^3}$$

For $$p \to \infty$$, $$\ell_\infty = \max_i (\vert x_1 \vert, \vert x_2 \vert, \ldots, \vert x_n \vert)$$

In the following image we can see the shape of the $$L^p$$ norm for various values of $$p$$. The vector space
that we are operating is $$R^2$$, i.e. vectors with two components, $$x$$ and $$y$$.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/lp_norms_2d.png" alt="The lp norm for various values of p">
</p>



---
layout: post
title:  "Norms and machine learning"
date:   2019-10-19
categories: [machine learning]
tags: ['mathematics', 'neural networks']
---

### Introduction
A vector space, known also as a linear space, is a collection of objects (the vectors), which may be added together and multiplied by some scalar (a number). In machine learning we use vectors all the time. Here are some examples:

* *Feature vectors* that are collections of numbers that we group them together when representing an object. In image processing, the features' values may be the pixels of the image, so assuming a 128 $$\times$$ 128 grayscale image, we get a 16384 long vector. Feature vectors are equivalent to the vectors of independent variables (the $$x$$-s) in linear regression, but usually are much larger.
* The *output* of a machine learning model, say a neural network that is trained to identify hand-written digits, may be represented as a vector, e.g. $$y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]^T$$ for representing "2" as the correct output. By the way this representation is called *one hot encoding* and the vector *one hot vector* :D
* The *loss function*, i.e. the function that tells us how good or how bad are predictions are, is also directly related to the norm of a particular vector space. For example, the mean squared error is defined as $$\text{MSE} = \frac{1}{N} \sum_i (y_{\text{true,}i} - y_{\text{predicted,}i})^2$$, which (as we shall see) is connected to the $$\ell_2$$ norm of vectors $$y_i = y_{\text{true,}i} - y_{\text{predicted,}i}$$.

Let's see what does it mean for a vector space to have a norm.

### Norms
Informally, a norm is a function that accepts as input a vector from our vector space $$V$$ and spits out a real number that tells us how big that vector is. In order for a function to qualify as a norm, it must first fulfill some properties, so that the results of this metrization process kind of "make sense". These properties are the following. For all $$u, v$$ in the vector space $$V$$ and $$\alpha$$ in $$\mathbb{R}$$:

* $$\|v\| \ge 0$$ and $$\|v\| = 0 \Leftrightarrow v = 0$$ (positive/definite)
* $$\| \alpha v \| = \|\alpha\| \| v \|$$ (absolutely scalable)
* $$\|u+v\| \le \|u\|+\|v\|$$ (Triangle inequality)

### The $$\ell_p$$ norm
One of the most widely known family of norms is the $$\ell_p$$ norm, which is defined as:

$$
\ell_p = \left( \sum_{i=1}^N |x_i|^p \right)^{1/p}, \text{for } p \ge 1
$$

For $$p = 1$$, we get $$\ell_1 = \vert x_1 \vert + \vert x_2 \vert + \ldots + \vert x_n \vert$$

For $$p = 2$$, $$\ell_2 = \sqrt{x_1^2 + x_2^2 + \ldots + x_n^2}$$

For $$p = 3$$, $$\ell_3 = \sqrt[3]{\vert x_1 \vert ^3 + \vert x_2 \vert ^3 + \ldots + \vert x_n \vert ^3}$$

For $$p \to \infty$$, $$\ell_\infty = \max_i (\vert x_1 \vert, \vert x_2 \vert, \ldots, \vert x_n \vert)$$

Every $$\ell_p$$ attaches a different "size" in vectors and the answer to the question on what's the best norm to use, depends on the problem you are solving. For example, if you are building an application for taxi drivers in Manhattan that needs the minimal distance between two places, then using the $$\ell_1$$ norm would make more sense than $$\ell_2$$.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/manhattan_distance.jpg" alt="Example of Manhattan distance and Euclidean distance">
</p>
Image taken from quora.com.

In the following image we can see the shape of the $$\ell_p$$ norm for various values of $$p$$. The vector space that we are operating is $$\mathbb{R}^2$$. In specific, we see the boundary of $$\ell_p = 1$$, i.e. all those vectors $$v = (x,y)$$ whose $$\ell_p$$ norm equals $$1$$.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/lp_norms_2d.png" alt="The lp norm for various values of p">
</p>

At this point the careful reader might have noticed that $$p$$ should be a real number greater than or equal to 1. So is $$\ell_{1/2}$$ a norm? The answer is no, because it violates the triangle equality. Let $$u = (x_1, y_1), v = (x_2, y_2)$$ then $$u+v=(x_1+x_2, y_1+y_2)$$.

$$
\|u+v\| \le \|u\|+\|v\| \Leftrightarrow \left(\sqrt{x_1+x_2} + \sqrt{y_1+y_2} \right)^2 \le \left(\sqrt{x_1} + \sqrt{y_1}\right)^2 + \left( \sqrt{x2} + \sqrt{y_2}\right)^2
$$

If you expand the squares and simplify the inequality, you will end up in a false statement.

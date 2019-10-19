---
layout: post
title:  "Norms and machine learning"
date:   2019-10-19
categories: [machine learning]
tags: ['mathematics', 'neural networks']
---

A vector space, known also as a linear space, is a collection of objects (the vectors),
which may be added together and multiplied by some scalar (a number). Informally, a norm
is a function that accepts as input a vector from our vector space V and spits out a real
number that tells us how big a vector is. In order for a function to quantify as a norm,
it must first fulfill some properties, so that the results of this metrization process "make sense".
These properties are the following:

* $$\forall u   \in V: \|v\| \ge 0$$ and $$\|v\| = 0 \Leftrightarrow v = 0$$ (positive/definite)
* $$\forall u   \in V, \alpha \in \mathbb{R}: \| \alpha v \| = \|\alpha\| \| v \|$$ (absolutely scalable)
* $$\forall u, v\in V: \|u+v\| \le \|u\|+\|v\|$$ (Triangle inequality)

$$
\ell_p = \left( \sum_{i=1}^N |x_i|^p \right)^2, p \ge 1
$$

$$
l_1 = x_1 + x_2 + \ldots + x_n
$$

$$
l_2 = \sqrt{x_1^2 + x_2^2 + \ldots + x_n^2}
$$

$$
l_3 = \sqrt[3]{x_1^3 + x_2^3 + \ldots + x_n^3}
$$

$$
l_\infty = \max_i(|x_1|, |x_2|, \ldots, |x_n|)
$$

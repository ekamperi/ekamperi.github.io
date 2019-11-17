---
layout: post
title:  "Dual spaces"
date:   2019-11-17
categories: [mathematics]
tags: ['general relativity', 'linear algebra', 'mathematics']
---

### Dual spaces
Given a vector space $$V$$, we define its dual space $$V^*$$ to be the set of all *linear transformations* $$\varphi: V \to \mathbb{F}$$, where $$\varphi$$ is a linear functional. Here is a list of examples:

* <ins>Example 1</ins>: Let $$V = \mathbb{R}^3$$ and $$\varphi: \mathbb{R}^3 \to \mathbb{R}$$, $$\varphi(x,y,z) = 2x+3y+4z$$

* <ins>Example 2</ins>: Let $$V = P_n$$ (the set of polynomials with degreee $$n$$) and $$\varphi: P_n \to \mathbb{R}$$, $$\varphi(p) = p(1)$$. E.g., $$\varphi(1 + 2x + 3x^2) = 1 + 2\cdot 1 + 3\cdot 1^2 = 6$$.

* <ins>Example 3</ins>: Let $$V = M_{m\times n}$$ (the set of matrices with dimensions $$m\times n$$) and $$\varphi: M_{m\times n} \to \mathbb{R}$$, $$\varphi(A) = \text{Trace}(A)$$. E.g.,

$$
\varphi\left(\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}\right) = 1+ 5 = 6
$$


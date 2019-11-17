---
layout: post
title:  "Dual spaces, dual basis, dual vectors"
date:   2019-11-17
categories: [mathematics]
tags: ['general relativity', 'linear algebra', 'mathematics']
---

### Dual spaces
So, it turns out that *dual spaces* and *dual vectors* sneak into general relativity and, therefore, I decided to take a closer look at them. The videos from Dr Peyam on YouTube are also very informative.

Given a vector space $$V$$, we define its dual space $$V^*$$ to be the set of all *linear transformations* $$\varphi: V \to \mathbb{F}$$. The $$\varphi$$ is called a linear functional. In other words, $$\varphi$$ is something that accepts a vector $$v \in V$$ as input and spits out an element of $$\mathbb{F}$$ (lets just assume that $$\mathbb{F} = \mathbb{R}$$, meaning that it spits out real numbers). If you take all the possible (linear) ways that a $$\varphi$$ can eat such vectors and produce real numbers, you get $$V^*$$. Here is a list of examples of dual spaces:

* <ins>Example 1</ins>: Let $$V = \mathbb{R}^3$$ and $$\varphi: \mathbb{R}^3 \to \mathbb{R}$$, then $$\varphi(x,y,z) = 2x+3y+4z$$ is a member of $$V^*$$.

* <ins>Example 2</ins>: Let $$V = P_n$$ (the set of polynomials with degreee $$n$$) and $$\varphi: P_n \to \mathbb{R}$$, then $$\varphi(p) = p(1)$$ is a member of $$V^*$$. Concretely, $$\varphi(1 + 2x + 3x^2) = 1 + 2\cdot 1 + 3\cdot 1^2 = 6$$.

* <ins>Example 3</ins>: Let $$V = M_{m\times n}$$ (the set of matrices with dimensions $$m\times n$$) and $$\varphi: M_{m\times n} \to \mathbb{R}$$, then $$\varphi(A) = \text{Trace}(A)$$ is a member of $$V^*$$. In specific,

$$
\varphi\left(\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}\right) = 1+ 5 = 6
$$

* <ins>Example 4</ins>: Let $$V = C([0,1])$$ (the set of all continuous function on the interval $$[0,1]$$) and $$\varphi: C[(0,1)] \to \mathbb{R}$$, then $$\varphi(g) = \int_0^1 g(x) \mathrm{d}x$$ is a member of $$V^*$$. For instance, $$\varphi(e^x) = \int_0^1 e^x \mathrm{d}x = e^1 - 1 = e -1$$.

### Dual basis example
If $$b = \{v_1, v_2, \ldots, v_n\}$$ is a basis of vector space $$V$$, then $$b^* = \{ \varphi_1, \varphi_2, \ldots, \varphi_n\}$$ is a basis of $$V*$$. So, how do you calculate the dual basis vectors? You set $$\varphi_i(v_j) = \delta_{ij}$$. Then any functional $$\varphi$$ can be written as a linear combination of the dual basis vector, i.e. $$\varphi = \varphi(v_1) \varphi_1 + \varphi(v_2) \varphi_2 + \ldots + \varphi(v_n) \varphi_n$$.

Let's see a concrete example. Assume $$V = \mathrm{R}^2$$ and a vector basis $$b = \{ (2,1), (3,1) \}$$, then what is the dual basis $$b^*$$? 

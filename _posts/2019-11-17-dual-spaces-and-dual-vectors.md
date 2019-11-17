---
layout: post
title:  "Dual spaces, dual vectors and dual basis"
date:   2019-11-17
categories: [mathematics]
tags: ['general relativity', 'linear algebra', 'mathematics']
---

So, it turns out that *dual spaces* and *dual vectors* sneak into general relativity and, therefore, I decided to take a closer look at them. The videos from Dr Peyam on YouTube are also very informative and the examples I'm listing here are from his lectures. First, we will go through the math and then we will invoke some intuition to make sense of all these.

### Dual spaces
Given a vector space $$V$$, we define its dual space $$V^*$$ to be the set of all *linear transformations* $$\varphi: V \to \mathbb{F}$$. The $$\varphi$$ is called a *linear functional*. In other words, $$\varphi$$ is something that accepts a vector $$v \in V$$ as input and spits out an element of $$\mathbb{F}$$ (lets just assume that $$\mathbb{F} = \mathbb{R}$$, meaning that it spits out a  real number). If you take all the possible (linear) ways that a $$\varphi$$ can eat such vectors and produce real numbers, you get $$V^*$$. Here is a list of examples of dual spaces:

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

As it turns out, the elements of $$V^*$$ satisfy the axioms of a vector space and therefore $$V^*$$ is indeed a vector space itself.

### Dual basis
If $$b = \{\mathbf{v_1}, \mathbf{v_2}, \ldots, \mathbf{v_n}\}$$ is a basis of vector space $$V$$, then $$b^* = \{ \varphi_1, \varphi_2, \ldots, \varphi_n\}$$ is a basis of $$V^*$$. If you define $$\varphi$$ via the following relations, then the basis you get is called the *dual basis*:

$$
\varphi_i \underbrace{(a_1 \mathbf{v_1} + \cdots + a_n \mathbf{v_n})}_{\text{A vector } \mathbf{v}\in V, a_i \in \mathbb{F}} = a_i, \quad i=1,\ldots,n
$$

It is as if the functional $$\varphi_i$$ acts on a vector $$\mathbf{v}\in V$$ and returns the $$i$$-th component $$a_i$$. Another way to write the above relations is if you set $$\varphi_i(\mathbf{v_j}) = \delta_{ij}$$.

Then any functional $$\varphi$$ can be written as a linear combination of the dual basis vector, i.e.

$$
\varphi = \varphi(\mathbf{v_1}) \varphi_1 + \varphi(\mathbf{v_2}) \varphi_2 + \ldots + \varphi(\mathbf{v_n}) \varphi_n
$$

#### Example
Let's see a concrete example. Assume $$V = \mathbb{R}^2$$ and a vector basis $$b = \{ (2,1), (3,1) \}$$, then what is the dual basis $$b^*$$?

By definition, it's $$\varphi_i(\mathbf{v_j}) = \delta_{ij}$$, therefore:

$$
\begin{align*}
\varphi_1 (\mathbf{v_1}) &= \delta_{11} = 1 \Leftrightarrow \varphi_1 (2,1) = 1 \Leftrightarrow \varphi_1 \left[ 2(1,0) + 1(0,1) \right] = 1 \Leftrightarrow 2\varphi_1(1,0) + 1\varphi_1(0,1) = 1 \\
\varphi_1 (\mathbf{v_2}) &= \delta_{12} = 0 \Leftrightarrow \varphi_1 (3,1) = 0 \Leftrightarrow \varphi_1 \left[ 3(1,0) + 1(0,1) \right] = 0 \Leftrightarrow 3\varphi_1(1,0) + 1\varphi_1(0,1) = 0
\end{align*}
$$

If you solve the system:

$$
\begin{bmatrix}
2 & 1 \\
3 & 1
\end{bmatrix}
\begin{bmatrix}
\varphi_1(1,0)\\
\varphi_1(0,1)
\end{bmatrix}=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

You get $$ \varphi_1(1,0) = -1, \quad \varphi_1(0,1) = 3$$. Therefore:

$$
\varphi_1(x, y) = x \varphi_1(1,0) + y\varphi_1(0,1) = -x + 3y
$$

Similarly one can prove that:

$$
\varphi_2(x, y) = x \varphi_2(1,0) + y\varphi_2(0,1) = x - 2y
$$

Therefore the dual basis $$b^*$$ is equal to $$\{ \varphi_1, \varphi_2 \} = \{ -x + 3y, x - 2y\}$$. Now here comes the magic. Suppose that you have a function $$\varphi = 8x - 7y$$ and you would like to write it as a linear combination of the dual basis. How would you do? 

$$
\varphi = \varphi(\mathbf{v_1}) \varphi_1 + \varphi(\mathbf{v_2}) \varphi_2 + \ldots + \varphi(\mathbf{v_n}) \varphi_n
$$

Where $$\mathbf{v_1} = (2,1)$$ and $$\mathbf{v_2} = (3,1)$$. Let us do the math:

$$
\begin{align*}
8x - 7y &= \overbrace{\varphi(2,1)}^{\varphi(\mathbf{v_1})} \cdot \underbrace{(-x + 3y)}_{\varphi_1} + \overbrace{\varphi(3,1)}^{\varphi(\mathbf{v_2})} \cdot \underbrace{(x-2y)}_{\varphi_2}\\
8x - 7y &= (8 \cdot 2 - 7\cdot 1) \cdot (-x + 3y) + (8 \cdot 3 - 7\cdot 1) \cdot (x-2y)\\
8x - 7y &= 9 (-x + 3y) + 17 (x-2y)\\
8x - 7y &= 8x -7y
\end{align*}
$$

### The dual of a dual space

Recall that the dual of space is a vector space on its own right, since the linear functionals $$\varphi$$ satisfy the axioms of a vector space. But if $$V^*$$ is a vector space, then it is perfectly legitimate to think of its dual space, just like we do with any other vector space. This might feel too recursive, but hold on. The double dual space is $$(V^*)^* = V^{**}$$ and is the set of all *linear transformations* $$\varphi: V^* \to \mathbb{F}$$.

In other words, $$\varphi$$ is something that accepts a vector $$\hat{v} \in V^*$$ as input and spits out an element of $$\mathbb{F}$$ (again, we just assume that $$\mathbb{F} = \mathbb{R}$$, meaning that it spits out a real number). In plain English language, a double dual vector is a creature that eats "a creature that eats a vector and spits a real number" and spits a real number. Think about them as carnivore animals (double dual vectors) eating herbivore animals (dual vectors) eating plants (vectors), while at the same time they both produce feces (real numbers).

In several areas of mathematics *isomorphism* appears as a very general concept. The word derives from the Greek *iso*, meaning "equal", and *morphosis*, meaning "to form" or "to shape." Informally, an isomorphism is a map that preserves sets and relations among elements. When this map or this correspondence is established with no choices involved, it is called *canonical isomorphism*.


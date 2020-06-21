---
layout: post
title:  "A simple example of perturbation theory"
date:   2020-06-21
categories: [mathematics]
tags: ['mathematics', 'perturbation theory']
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

I was looking at the video lectures of Carl Bender at YouTube on mathematical physics. What a great teacher Carl Bender is! The first lectures are an introduction to the perturbation theory. They start with a straightforward problem, where we want to find the real root of the following quintic equation:

$$
x^5 + x = 1
$$

This equation cannot be solved exactly, like the quadratic, cubic, or quartic equations. However, we will see how the perturbation theory allows us to solve it with arbitrarily high precision.

The **first step** when doing perturbation theory is to introduce the perturbation factor $$\epsilon$$ into our problem. This is, to some degree, an art, but the general rule to follow is this. We put $$\epsilon$$ into our problem in such a way, that when we set $$\epsilon = 0$$, that is when we consider the unperturbed problem, we can solve it exactly. For instance, if we put $$\epsilon$$ as $$x^5 + \epsilon x = 1$$, then for $$\epsilon = 0$$, we get $$x^5 = 1$$, that we can solve exactly ($$x = 1$$).

The **second step** is to assume that the solution to the perturbed problem can be described by an infinite power series of $$\epsilon$$:

$$
x(\epsilon) = \sum_{n=0}^\infty a_n \epsilon^n
$$

In this particular example, let us consider only the first 4 terms:

$$
x(\epsilon) = a_0 + a_1 \epsilon + a_2 \epsilon^2 + a_3 \epsilon^3 = 1 + a_1 \epsilon + a_2 \epsilon^2 + a_3 \epsilon^3
$$

Why did we set $$a_0 = 1$$? Well, $$x(0) = a_0$$ and we already established that $$x(0) = 1$$ when we solved the unperturbed problem. Since $$x(\epsilon)$$ is a solution to the perturbed problem, then it must satisfy the initial equation that we are solving:
$$
x(\epsilon)^5 + x(\epsilon) = 1 \Leftrightarrow
(1+a_1\epsilon + a_2\epsilon^2 + a_3 \epsilon^3)^5 + \epsilon (1+a_1\epsilon+a_2 \epsilon^2 + a_3 \epsilon^3) = 1
$$

Recall also that:
$$
(1+s)^5 = 1 + 5s + 10s^2 + 10 s^3 + \ldots
$$

and let us set $s = a_1\epsilon + a_2\epsilon^2 + a_3 \epsilon^3$

Therefore:
$$
\begin{align*}
&1 + 5a_1\epsilon + 5a_2\epsilon^2 + 5a_3\epsilon^3 + 10(a_1^2\epsilon^2 + 2a_1 a_2 \epsilon^3 + \ldots) + \ldots\\
&=1 + 5a_1\epsilon + \epsilon^2(5a_2+10a_1^2 ) + \epsilon^3(5a_3 + 20a_1 a_2) + \ldots\\
&\epsilon + a_1 \epsilon^2 + a_2 \epsilon^3 + \ldots
\end{align*}
$$

$$1 + 5 a_1 = 0 \Rightarrow a_1 = -\frac{1}{5}$$

$$a_1 + 5a_2 + 10a_1^2 = 0 \Rightarrow
a_2 = \frac{1}{5} \left[-10\left(-\frac{1}{5}\right)^2 - \left(-\frac{1}{5}\right)\right] \Rightarrow
a_2 = -\frac{1}{25}$$

Therefore:

$$
x(\epsilon) = 1 + a_1 \epsilon + a_2 \epsilon^2 + a_3 \epsilon^3 =
1 - \frac{\epsilon}{5} - \frac{\epsilon^2}{25}
$$

---
layout: post
title:  "A simple example of perturbation theory"
date:   2020-06-21
categories: [mathematics]
tags: ['maths', 'perturbation theory']
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

I was looking at the video lectures of Carl Bender at YouTube on mamthematical physics. What a great teacher Carl Bender is! The first lectures are an introduction the perturbation theory. They start with a really simple problem, where we want to solve the following quintic equation:

$$
x^5 + x = 1
$$

This equation cannot be solved exactly, like the quadratic equation. However, we will see how perturbation theory allows us to solve it with arbitratily high precision. The first step when doing perturbation theory is to introduce the perturbation factor $$\epsilon$$ into our problem. This is to some degree an art, but the rule to follow is this. We put $$\epsilon$$ into our problem in such a way, that when we set $$\epsilon = 0$$, that is when we consider the unperturbed problem, we are able to solve it exactly.

For instance, if we put $$\epsilon$$ as $$x^5 + \epsilon x = 1$$, then for $$\epsilon = 0$$ we get $$x^5 + 1 = 0$$ that we can solve exactly.

$$
x(\epsilon) = \sum_{n=0}^\infty a_n \epsilon^n
$$

$$
x(\epsilon) = a_0 + a_1 \epsilon + a_2 \epsilon^2 + a_3 \epsilon^3 = 1 + a_1 \epsilon + a_2 \epsilon^2 + a_3 \epsilon^3
$$

$$
(1+a_1\epsilon + a_2\epsilon^2 + a_3 \epsilon^3)^5 + \epsilon (1+a_1\epsilon+a_2 \epsilon^2 + a_3 \epsilon^3) = 1
$$

Recall that:
$$
(1+s)^5 = 1 + 5s + 10s^2 + 10 s^3 + \ldots
$$

and set $s=a_1\epsilon + a_2\epsilon^2 + a_3 \epsilon^3$

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

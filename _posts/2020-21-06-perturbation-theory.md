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

Suppose that we want to solve the following quintic equation:
$$
x^5 + x = 1
$$

We insert the perturbation factor $\epsilon$:
$$
x^5 + \epsilon x = 1
$$

The unperturbed problem is when $\epsilon = 0$:
$$
\epsilon = 0: x^5 = 1 \Rightarrow x = 1
$$

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

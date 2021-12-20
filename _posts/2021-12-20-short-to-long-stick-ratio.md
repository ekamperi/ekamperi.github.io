---
layout: post
title:  "Short to long stick ratio: a nice little math problem"
date:   2021-12-20
categories: [mathematics]
tags: ['mathematics']
description: How to calculate the average ratio of short to long stick ratio.
---

Hola! Long time no see! In the past months, I've been very busy working at Chronicles Health, a digital health company, on a course to revolutionize the management of inflammatory bowel disease.

In today's blog post, I'm not going to talk about some fancy machine learning or data science topic. Instead, I'll write about a nice little mathematical problem I've stumbled upon on the Internet. And by solving it, I'll demonstrate my thought process. Without further ado:

**Suppose that we crack 10.000 rods at random into two pieces by throwing them against a rock. What is the average ratio of the length of the short piece to the length of the long piece?**

Here's my thought process. We start by modeling the problem, and for that, we need to assign symbols to the various components. So, there's a rod, and two pieces, a short and a long one. Let's say that the rod has length $$L$$. Then, if we agree that the short piece has size $$x$$, the rest will be the long one with length $$L-x$$. Okay, we named things, but we need to constrain the values that our variables assume so that the symbols always "make sense". Since $$x$$ is the short part, it really can't be larger than half the rod because then it would be the long one! So, $$x\in[0,L/2]$$. Also, $$L>0$$ or there would be any rod, to begin with. So, we are interested in the average ratio of the short to long pieces, but that is $$\left<x/(L-x)\right>$$.

At this point, we need to invoke the **expected value**notion. The expected value of a random variable $$x$$, often denoted $$\operatorname {E} [X]$$, can be thought of as a generalized version of the weighted average, where the weights are given by the probabilities. If we have a fair die, then the probability of each outcome is $$p=1/6$$ and the expected value after one throw is given by $$1 \times 1/6 + 2 \times 1/6 + \ldots + 6 \times 1/6 = 7/2$$. So, if we throw the die 10.000 times and take the mean of the outcomes, we *expect* to get a value of 3.5.


{% highlight mathematica %}
{% raw %}
Mean@RandomInteger[{1, 6}, 1000] // N
3.58
{% endraw %}
{% endhighlight %}

Alright, back to our problem! Here we don't throw dice and look at the number we get. Instead, we crack rods and look at the number $$x/(L-x)$$. To calculate the *expected value* of this ratio, we write:

$$
\begin{align*}
E(x/(L-x)) = \int_{0}^{L/2} \frac{x}{L-x} p(x)d x
\end{align*}
$$

Where $$x\frac{L-x}$$ is the *value of the ratio* when the rod breaks at short length $$x$$, and $$p(x)$$ is the *probability* of this particular break happening. We assume that a rod is equally probable to break at a point $$x$$ since the problem doesn't state any particular probability distribution. Therefore, $$p(x) = 1/(L/2)=2/L$$. Is this intuitive? Yes, because the longer the rod, the less probable it is for a particular break to happen. Imagine if we had a die with 1.000.000 faces; what would be the probability of getting the number "6" after a throw? 1/1.000.000.

$$
\begin{align*}
E(x/(L-x)) = \int_{0}^{L/2} \frac{x}{L-x} \cdot\frac{2}{L} \mathrm{d}x = 
2\int_{0}^{L/2} \frac{x}{L(L-x)} \mathrm{d}x 
\end{align*}
$$

From this point forward, it's just about calculating an integral. Such integrals are usually calculated by breaking up the fraction into a sum of simple fractions, e.g.,

$$
\frac{x}{L(L-x)}=\frac{A}{L} + \frac{B}{L-x}
$$

and solving for $$A, B$$. Since this is a simple one, we could just see that:

$$
\frac{x}{L(L-x)}=-\frac{1}{L} + \frac{1}{L-x}
$$

Therefore:

$$
\begin{align*}
E(x/(L-x)) =
2\int_{0}^{L/2} \left( -\frac{1}{L} + \frac{1}{L-x} \right) \mathrm{d}x=\\
-\frac{2}{L} \left(\frac{L}{2}-0\right) - 2\left[\ln{(L-x)}\right]_{0}^{L/2}=\\
-1 - 2\left[\ln\left({L}-\frac{L}{2}\right) - \ln{L}\right]
\end{align*}=\\
-1-2(\ln{1/2})= -1+\ln{4}
$$

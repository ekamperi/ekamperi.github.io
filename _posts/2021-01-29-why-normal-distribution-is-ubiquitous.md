---
layout: post
title:  "Why normal distribution is so ubiquitous"
date:   2020-01-29
categories: [mathematics]
tags: ['machine learning', 'Mathematica', 'mathematics', 'statistics']
description: A summary of explanations and insights on why the normal distribution shows up so often in real world phenomena
---

## As simple as it can get
Have you ever wondered why normal distributions are encountered so frequently? Some examples include the height, birth weight of newborns, the sum of dice, and others.

Let's look at a couple of examples.

Suppose we throw a fair dice. Every number has an equal probability of showing up, i.e., $$p=1/6$$. For instance, let us assume the following sequence of 30 occurences:

{3, 4, 6, 5, 1, 5, 6, 5, 1, 5, 3, 1, 2, 4, 1, 1, 2, 5, 3, 1, 3, 4, 3, 6, 2, 4, 6, 2, 4, 3}

If we plot the histogram of these frequencies, we get something close to a uniform distribution (that's ok, given we only threw it 30 times). Here comes the magic. Suppose that we throw two dice 15 times, and we get:

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/hist_of_sum1.png" alt="Uniform distribution histogram">
</p>

{{3, 4}, {6, 5}, {1, 5}, {6, 5}, {1, 5}, {3, 1}, {2, 4}, {1, 1}, {2, 5}, {3, 1}, {3, 4}, {3, 6}, {2, 4}, {6, 2}, {4, 3}}

So the first time we got 3 and 4, the second time 6 and 5, and so on. Now take the sum of each throw:

{7, 11, 6, 11, 6, 4, 6, 2, 7, 4, 7, 9, 6, 8, 7}

And plot the histogram of the sums. As you see, the distribution of the sums looks roughly like a gaussian. If we keep going by assuming more dice rolls, the sums' distribution will get closer to a normal distribution. We already may answer why normal distributions are so ubiquitous: because many variables in the real world are the sum of other variables.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/hist_of_sum2.png" alt="Sum of dice histogram">
</p>

For example, consider a person's height. This is determined by the sum of many independent variables:
* The genes (I haven't researched it by I presume several genes contribute to height, not just one)
* Hormones (E.g., growth hormone)
* The environment
** The nutrition in terms of what one eats every day during the developmental phase
** Sports / activity levels
** Pollution
* Other factors

A person's height is akin to the sum of rolling many dice. Each die is similar to each of the above factors. I.e., the genes are the 1st die, the hormones the 2nd, etc.

What follows is the presentation of the same arguments written with mathematical symbols. Do not worry if you find trouble understanding. You won't miss any insights!

## Why convolution is the way to express the sum of two random variables, $$X, Y$$, mathematically

Suppose that each variable of $$X, Y$$, and $$Z=X+Y$$ (their sum) has a probability mass function (pmf). By definition, the pmf for $$Z=X+Y$$ at some number $$z_i$$ gives the proportion of the sum $$X+Y$$ that is equal to $$z_i$$ and is denoted as $$Pr(z_i=X+Y)$$. By applying the Law of Total Probability, we can break up $$Pr(Z=X+Y)$$, into the sum over all possible values of $$x$$, such that $$X=x$$, and $$Y=z-x$$. 

$$
\text{Pr}(z=X+Y) = \sum_{x} \text{Pr}(X=x,Y=z−x)
$$

That's the convolution in the discrete case. For continuous variables, it's just:

$$
(f * g)(t) \stackrel{\text{def}}{=}
 \int_{-\infty}^\infty f(s) \, g(t - s) \, \mathrm{d}s
$$

Therefore, any distribution with finite variance given some time and convolution will morph into a Gaussian.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/conv_1.png" alt="Convolution with itself">
</p>

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/conv_2.png" alt="Convolution with itself">
</p>

{% highlight mathematica %}
{% raw %}
f0[x_] := UnitBox[x];
fg[f_, g_] := Convolve[f, g, x, y] /. y -> x
repcon[n_] := repcon[n] = Nest[fg[#, #] &, f0[x], n]
Grid[
 Partition[#, 2] &@
  (Plot[#, {x, -3, 3}, Exclusions -> None, Filling -> Axis] & /@ 
    Table[repcon[k], {k, 0, 3}])
 ]
{% endraw %}
{% endhighlight %}



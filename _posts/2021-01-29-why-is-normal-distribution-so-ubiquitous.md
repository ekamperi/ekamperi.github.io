---
layout: post
title:  "Why is normal distribution so ubiquitous"
date:   2020-01-29
categories: [mathematics]
tags: ['machine learning', 'Mathematica', 'mathematics', 'statistics']
description: A summary of explanations and insights on why the normal distribution shows up so often in real-world phenomena
---

## Introduction
Have you ever wondered why [normal distributions](https://en.wikipedia.org/wiki/Normal_distribution) are encountered so frequently in everyday life? Some examples include the height of people, newborns' birth weight, the sum of two dice, and many others. Also, in our statistical models, we often assume that some quantity is modeled by a normal distribution. Is there some fundamental reason Gaussians are all over the place? Yes, there is!

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/normal_dist/bell.png" alt="Bell curve artistic">
</p>

[Image taken from here](https://www.impactlab.com/2011/10/16/top-10-photos-of-the-week-200/)

## As simple as it can get
### The case of rolling dice

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/dice.jpg" alt="Dice">
</p>


Let's look at a couple of examples, starting with the case of throwing a fair die. Every number on each marked side has an equal probability of showing up, i.e., $$p=1/6$$. For instance, let us assume the following sequence of 30 occurences:

{% highlight mathematica %}
{% raw %}
{3, 4, 6, 5, 1, 5, 6, 5, 1, 5, 3, 1, 2, 4, 1, 1, 2, 5, 3, 1, 3, 4, 3, 6, 2, 4, 6, 2, 4, 3}
{% endraw %}
{% endhighlight %}

If we plot the frequency histogram of these numbers, we get something close to a [uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution) (that's ok, given we only threw the die 30 times). 

<p align="center">
 <img style="width: 40%; height: 40%" src="{{ site.url }}/images/normal_dist/hist_sum_of_1.png" alt="Uniform distribution histogram">
</p>

Here comes the magic. Suppose that we throw two dice 15 times, and we get:

{% highlight mathematica %}
{% raw %}
{{3, 4}, {6, 5}, {1, 5}, {6, 5}, {1, 5}, {3, 1}, {2, 4}, {1, 1},
 {2, 5}, {3, 1}, {3, 4}, {3, 6}, {2, 4}, {6, 2}, {4, 3}}
{% endraw %}
{% endhighlight %}

So the first time we got 3 and 4, the second time 6 and 5, and so on. Now let us take the sum of each throw:

{% highlight mathematica %}
{% raw %}
{7, 11, 6, 11, 6, 4, 6, 2, 7, 4, 7, 9, 6, 8, 7}
{% endraw %}
{% endhighlight %}

And plot the frequency histogram of the sums.

<p align="center">
 <img style="width: 40%; height: 40%" src="{{ site.url }}/images/normal_dist/hist_sum_of_2.png" alt="Sum of dice histogram">
</p>

As you see, the sums' distribution shifted from a uniform to something that looks vaguely like a gaussian. Just think about it for a second. Why does the number $$7$$ appear so often as the sum of two dice? Well, because many combinations could end up having a sum of $$7$$. E.g., $$2 + 5$$, $$5 + 2$$, $$1 + 6$$, $$6 + 1$$, $$3 + 4$$, $$4 + 3$$. The same logic applies to $$6$$ as well. However, to get some very large sum, say $$12$$, there is only one combination, namely $$6 + 6$$. The same applies to small sums, like $$2$$, which is realized only by $$1 + 1$$.

If we keep going by assuming the sum of more dice, say 3 dice, the sums' distribution will get even closer to a normal distribution. The reasoning is the same as previously. There will be many more combinations of dice summing up to some value in the middle, rather than summing in some extreme value. We already may answer why normal distributions are so ubiquitous: because many variables in the real world are the sum of other independent variables. And, when independent variables are added together, their sum converges to a normal distribution. Neat?

In the following histograms, we observe the sums' distribution evolution, starting with only one die and going up to 20 dice!

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/hist_of_sum.png" alt="Sum of dice histogram">
</p>

### The case of random walk
This example was reproduced by the excellent book "Statistical Rethinking: A Bayesian Course with Examples in R and Stan; Second Edition". Suppose we place 1000 folks on a field and then ask them to flip a coin, and depending on the outcome, to take a step from 0 to 1 meter in the indicated by the coin direction. Each person takes, say, a total of 16 such steps. At the end of this game, we can't really tell each person's position, but we can say something about the distribution of their distances. Arguably, **the distribution will be normal simply because there are vastly more sequences of left-right steps whose sum ends up being zero than sequences of steps that end up being non-zero.** For example, to end up with a distance of 16, one needs to take 16 consecutive left steps or 16 successive right steps. That's just very unlikely to happen (remember that the direction people move is determined by a flip coin, so they should have 16 heads or 16 tails in a row). The following code generates the plot below, but feel free to skip it.

{% highlight mathematica %}
{% raw %}
randomWalks[reps_, steps_] :=
 Module[{},
  Table[
   Accumulate[
    Flatten@Append[{0}, 
      RandomVariate[UniformDistribution[{-1, 1}], steps]]],
   {k, reps}]
  ]

Show[
 ListPlot[walks, Joined -> True, PlotStyle -> Directive[Blue, Opacity[0.02]],
  AspectRatio -> 1/3, Frame -> {True, True, True, True}, FrameLabel -> {"Steps", "Position"}, 
  FrameTicksStyle -> Directive[Bold], FrameTicks -> {{{-6, -3, 0, 3, 6}, None},
   {{0, 4, 8, 16, 32}, None}},
  GridLines -> {{4, 8, 16, 32}, None}, GridLinesStyle -> Dashed, ImageSize -> Large],
 ListPlot[walks[[1]], Joined -> True, PlotStyle -> Red]]
 {% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/random_walk.png" alt="Random walk">
</p>

These are the distributions of the positions (distances) after 4, 8, and 16 steps. Notice how the distribution converges on a Gaussian (the red one) as the number of steps increases. 

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/hist_vs_steps.png" alt="Random walk">
</p>

**The mind-blowing thing is that it doesn't matter what the underlying distribution processes are**. They might be uniform like in our examples, or almost anything else. In the end, the sums will converge on a Gaussian. The only thing affected is the speed of convergence, which is high when the underlying distribution is uniform and slower otherwise.

### So why are normal distributions so ubiquitous?
**Because many things in our world emerge as the sum of smaller independent parts.** For example, consider a person's height. This is determined by the sum of many independent variables:
* The **genes** (I haven't researched it by I presume several genes contribute to height, not just one)
* Hormones (E.g., [growth hormone](https://en.wikipedia.org/wiki/Growth_hormone))
* The **environment**
  * The **nutrition** in terms of what one eats every day during the developmental phase
  * **Sports / activity levels**
  * **Pollution**
* **Other** factors

A person's height is akin to the sum of rolling many dice. Each die is similar to each of the above factors. I.e., the genes are the 1st die, the hormones the 2nd, etc. So, to end up being very tall, you need them all in favor of you: the genes, the hormones, the environment, the activity, etc. It's as if achieving 16 heads in a row in the random walk setting. That's why there are few very tall people. The same logic applies to very short people. In this case, you need to have everything against you: bad genes, poor nutrition, no activity, etc.

What follows is the presentation of the same arguments written with mathematical symbols. Do not worry if you find them confusing. You won't have missed any insights!

## Why is convolution the natural way to express the sum of two random variables, $$X, Y$$

Suppose that each variable of $$X, Y$$, and $$Z=X+Y$$ (their sum) has a probability mass function (pmf). By definition, the pmf for $$Z=X+Y$$ at some number $$z_i$$ gives the proportion of the sum $$X+Y$$ that is equal to $$z_i$$ and is denoted as $$\text{Pr}(z_i=X+Y)$$. By applying the [Law of Total Probability](https://en.wikipedia.org/wiki/Law_of_total_probability), we can break up $$\text{Pr}(Z=X+Y)$$, into the sum over all possible values of $$x$$, such that $$X=x$$, and $$Y=z-x$$. 

$$
\text{Pr}(z=X+Y) = \sum_{x} \text{Pr}(X=x,Y=z−x)
$$

That's the convolution in the discrete case. For continuous variables, it's just:

$$
(f * g)(t) \stackrel{\text{def}}{=}
 \int_{-\infty}^\infty f(s) \, g(t - s) \, \mathrm{d}s
$$

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


<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/normal_dist/unit_box.png" alt="Convolution of unit box function with itself">
</p>


Αny distribution with finite variance given some time and convolution will morph into a Gaussian.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/conv_1.png" alt="Convolution with itself">
</p>

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/conv_2.png" alt="Convolution with itself">
</p>


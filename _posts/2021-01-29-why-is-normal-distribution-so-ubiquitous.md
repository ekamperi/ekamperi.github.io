---
layout: post
title:  "Why is normal distribution so ubiquitous?"
date:   2021-01-29
categories: [mathematics]
tags: ['machine learning', 'Mathematica', 'mathematics', 'statistics']
description: A summary of explanations and insights on why the normal distribution shows up so often in real-world phenomena
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
Have you ever wondered why [normal distributions](https://en.wikipedia.org/wiki/Normal_distribution) are encountered so frequently in everyday life? Some examples include the height of people, newborns' birth weight, the sum of two dice, and numerous others. Also, in statistical modeling, we often assume that some quantity is represented by a normal distribution. Particularly when we don't know the actual distribution, but we do know the sample mean and standard deviation. It is as if Gaussian is the "default" or the most generic. Is there some fundamental reason Gaussians are all over the place? Yes, there is!

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/normal_dist/bell.png" alt="Bell curve artistic">
</p>
[Image taken from here](https://www.impactlab.com/2011/10/16/top-10-photos-of-the-week-200/)

## As simple as it can get
### The case of rolling dice

<p align="center">
 <img style="width: 20%; height: 20%" src="{{ site.url }}/images/normal_dist/dice.jpg" alt="Dice">
</p>

Let's take a look at a couple of examples, starting with the case of throwing a fair die. Every number on each marked side has an equal probability of showing up, i.e., p=1/6. Let us examine the following sequence of 30 trials:

{% raw %}
3, 4, 6, 5, 1, 5, 6, 5, 1, 5, 3, 1, 2, 4, 1, 1, 2, 5, 3, 1, 3, 4, 3, 6, 2, 4, 6, 2, 4, 3
{% endraw %}

If we get to plot the frequency histogram of these numbers, we notice something resembling a [uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution). Considering we only threw the die 30 times, the small deviations are ok.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/normal_dist/hist_sum_of_1.png" alt="Uniform distribution histogram">
</p>

**Here comes the magic**. Suppose that we throw two dice 15 times, and we get the follwing pairs:

{% raw %}
{3, 4}, {6, 5}, {1, 5}, {6, 5}, {1, 5}, {3, 1}, {2, 4}, {1, 1},
 {2, 5}, {3, 1}, {3, 4}, {3, 6}, {2, 4}, {6, 2}, {4, 3}
{% endraw %}

So the first time, we get 3 and 4, the second time 6 and 5, and so on. Now let us **take the sum of each pair**:

{% raw %}
{7, 11, 6, 11, 6, 4, 6, 2, 7, 4, 7, 9, 6, 8, 7}
{% endraw %}

And let's plot the frequency histogram **of the sums**.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/normal_dist/hist_sum_of_2.png" alt="Sum of dice histogram">
</p>

**As you notice, the sums' distribution switched from a uniform to something that could end up being a Gaussian.** Just think about it for a second. Why does the number 7 appear so often as the sum of two dice? Well, because many combinations could end up having a sum of 7. E.g., 1 + 6, 6 + 1, 2 + 5, 5 + 2, 3 + 4, 4 + 3. The same logic applies to 6 as well. However, to get some very large sum, say 12, there is only one combination, namely 6 + 6. The same applies to small sums, like 2, which is realized only by 1 + 1.

If we keep going by throwing more dice, say 4 at a time, the sums' distribution will get even closer to a normal distribution. **There will be even more combinations of dice outcomes summing up to a "central" value, rather than in some extreme value.** In the following figure, we plot the number of distinct combinations that yield all possible sums in a "roll 4 dice" scenario. There is an exact correspondence between the number of generating combinations and the frequency a sum appears.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/normal_dist/combinations.png" alt="Sum of dice plot">
</p>

So here are the 146 combinations of dice values that sum to 11.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/raster_combinations.png" alt="Sum of dice plot">
</p>

In the following histograms, we examine the sums' distributions, starting with only one die and going up to 20 dice! By rolling merely three dice, the sum already looks pretty normally distributed. 

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/hist_of_sum.png" alt="Sum of dice histogram">
</p>

We may now answer why bell curves are so ubiquitous: **because many variables in the real world are the sum of other independent variables.** And, when independent variables are added together, their sum converges to a normal distribution. Neat?

### The case of [random walks](https://en.wikipedia.org/wiki/Random_walk)
This example is reproduced (and modified) by the excellent book *"Statistical Rethinking: A Bayesian Course with Examples in R and Stan; Second Edition".* Suppose we place 1000 folks on a field, one at a time. We then ask them to flip a coin and depending on the outcome, they take a step in the left or right direction. The distance of each step is a random number from 0 to 1 meter. Each person takes, say, a total of 16 such steps. The blue lines are the trajectories of the 1000 random walks. The red line is one such representative walk. At the right end of the plot, the grey line is the probability distribution of the position when the random walks have been completed.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/random_walk.png" alt="Random walk">
</p>

At the end of this game, we can't really tell each person's position, but we can say something about the distribution of their distances. Arguably, **the distribution will look normal simply because there are vastly more sequences of left-right steps whose sum ends up being around zero than sequences of steps that end up being far away from zero.** For example, to end up near position 6 or -6, one needs to take 16 consecutive left steps or 16 successive right steps. That's just very unlikely to happen (remember that the direction people move is determined by a flip coin, so they should have 16 heads or 16 tails in a row). The following code generates the plot below, but feel free to skip it.

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

These are the distributions of the positions (distances) after 4, 8, and 16 steps. Notice how the distribution converges on a Gaussian (the red one) as the number of steps increases. 

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/hist_vs_steps.png" alt="Random walk">
</p>

What we have discussed so far is the other side of the **[central limit theorem (CLT)](https://en.wikipedia.org/wiki/Central_limit_theorem)**. CTL establishes that when independent random variables are added, their properly normalized sum converges toward a normal distribution even if the original variables themselves are not normally distributed.

**The mind-blowing thing about this is that it doesn't matter what the underlying distribution processes are**. They might be uniform like in our examples, or (almost) anything else. In the end, the sums will converge on a Gaussian. The only thing that can vary is the speed of convergence, which is high when the underlying distribution is uniform and slower otherwise.

### So why are normal distributions so ubiquitous?
**Because many things in our world emerge as the sum of smaller independent parts.** For instance, consider a person's height. This is determined by the sum of many independent variables:
* The **genes** (I haven't researched it by I presume several genes contribute to height, not just one)
* **Hormones** (E.g., [growth hormone](https://en.wikipedia.org/wiki/Growth_hormone))
* The **environment**
  * The **nutrition** in terms of what one eats every day during the developmental phase
  * **Sports / activity levels**
  * **Pollution**
* **Other** factors

**A person's height is akin to the sum of rolling many dice.** Each die parallels one of the above factors. I.e., the genes match the 1st die, the hormones the 2nd, etc. So, to end up being very tall, you need them all working in favor of you: the genes, the hormones, the environment, the activity, etc. It's as if scoring 16 heads in a row in the random walk setting. That's why there are few very tall people. The same logic applies to very short people. In this case, you need to have everything working against you: bad genes, poor nutrition, no activity, etc. However, most people have an average height because some factors contribute positively and others negatively.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/normal_dist/four_dice.png" alt="Four dice">
</p>

What follows is the presentation of the same arguments with some math. Do not worry if you find them confusing. You won't have missed any insights!

## Why is convolution the natural way to express the sum of two random variables?

Summing random variables is at the heart of our discussion, so it deserves a few thoughts. Suppose that we have two random variables, $$X, Y$$, and we take their sum, $$Z=X+Y$$. Depending on what mathematical tools we use to describe the random variables, their sum is expressed differently. For a full list, feel free to [check this link](https://stats.stackexchange.com/questions/331973/why-is-the-sum-of-two-random-variables-a-convolution). We will discuss the case where $$X, Y, Z=X+Y$$ all have a probability mass function (PMF).

By definition, a PMF is a function that gives the probability that a *discrete* random variable is exactly equal to some value. For a fair die, the PMF is simply $$p(x)=\text{Pr}(X=x_i)=1/6$$, with $$x_i=1,2,\ldots,6$$. Back to our sum $$Z=X+Y$$. By definition, again, the PMF for $$Z=X+Y$$ at some number $$z_i$$ gives the proportion of the sum $$X+Y$$ that is equal to $$z_i$$ and is denoted as $$\text{Pr}(Z=z_i), z_i=X+Y$$. By applying the [Law of Total Probability](https://en.wikipedia.org/wiki/Law_of_total_probability), we can break up $$\text{Pr}(Z=X+Y)$$, into the sum over all possible values of $$x$$, such that $$X=x$$, and $$Y=z-x$$.

$$
\text{Pr}(z=X+Y) = \sum_{x} \text{Pr}(X=x,Y=z−x)
$$

But, that's the definition of the convolution operation in the discrete case. For continuous variables, it's just:

$$
(f \star g)(t) \stackrel{\text{def}}{=}
 \int_{-\infty}^\infty f(x) \, g(t - x) \, \mathrm{d}x
$$

### Convolving a unit box
Let's look at an example where we will convolve the unit box function with itself. The unit box function is equal to 1 for $$\mid x\mid \le 1/2$$ and 0 otherwise.

{% highlight mathematica %}
{% raw %}
f0[x_] := UnitBox[x];
fg[f_, g_] := Convolve[f, g, x, y] /. y -> x
(* Cache the results because symbolic convolution takes time *)
repconv[n_] := repconv[n] = Nest[fg[#, #] &, f0[x], n]
Grid[
 Partition[#, 2] &@
  (Plot[#, {x, -3, 3}, Exclusions -> None, Filling -> Axis] & /@ 
    Table[repconv[k], {k, 0, 3}])
 ]
{% endraw %}
{% endhighlight %}


<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/normal_dist/unit_box.png" alt="Convolution of unit box function with itself">
</p>

The following animation might clear up the operation of convolution between two functions.

<p align="center">
<video id="movie" width="100%" height="100%" preload controls>
   <source id="srcMp4" src="{{ site.url }}/images/normal_dist/unitbox_conv.mp4" />
</video>
</p>

### Convolving noisy data
Ok, so convolving a unit box function with itself, which pertains to rolling two dice and calculating their sums, led us to a Gaussian distribution. Does this have to do with the unit box? No! **Αny distribution with a finite variance will morph into a Gaussian, given some time and repeated convolutions.** In the following examples, we start with some quite noisy initial distributions and convolve them repeatedly with themselves. As you see, the result is again a normal distribution!

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/conv_1.png" alt="Convolution with itself">
</p>

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/conv_2.png" alt="Convolution with itself">
</p>

## Information-theoretic arguments
In 1948 [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) laid out the foundations of information theory. The need this new theory was called upon to meet was the effective and reliable transmission of messages. Although the motive was applied, information theory is deeply mathematical in its nature. A central concept in it is *entropy*, which is used somewhat differently than in thermodynamics. Consider a random variable $$X$$, which assumes the discrete values $$X = {x_i \mid i=1, 2, \ldots, K}$$. Of course, $$0 \le p_i \le 1$$ and $$\sum_{i=1}^{K} p_i = 1$$ must also be met.

Suppose now the extreme case that the value $$X=x_i$$ has a probability of occurring $$p_i=1$$, and $$p_{j\ne i}=0$$. In this scenario, there's no 
"surprise" by observing the value of $$X$$, and there's no message being transmitted. It is as if I told you that it's chilly today in Alaska or that sun raised at east. In this context, we define the information content that we gain by observing $$X$$ as the following function:

$$
I(x_i) = \log\left(\frac{1}{p_i}\right) = -\log p_i
$$

We then define as *entropy* the expected value of $$I(x_i)$$ over all $$K$$ discrete values $$X$$ takes:

$$
H(X) = \mathop{\mathbb{E}} \left(I(x_i)\right) = \sum_{i=1}^{K} p_i I(x_i) = \sum_{i=1}^{K} p_i \log\frac{1}{p_i} = -\sum_{i=1}^{K} p_i \log p_i
$$

Similarly, we define *differential entropy* for continuous variables:

$$
h(X) = -\int_{-\infty}^{\infty} p_X(x) \log p_X(x) \mathrm{d}x
$$

(However, some of the discrete's entropy properties do not apply to differential entropy, e.g., differential entropy can be negative.)

Alright, but what normal distribution has anything to do with these? **It turns out that normal distribution is the distribution that maximizes information entropy under the constraint of fixed mean $$m$$ and standard deviation $$s^2$$ of a random variable $$X$$.** So, if we know the mean and standard deviation of some data, the optimal distribution is the one that maximizes entropy, or, equivalently, that satisfies the least of our assumptions. This principle may be viewed as expressing **epistemic modesty** or **maximal ignorance** because it makes the least strong claim on a distribution.

Let us look at an even simpler case. E.g., consider a coin that comes tails with a probability $$p$$, and heads with a probability $$1-p$$. The entropy of the flip is then given by:

$$
H(X) = - \sum_{i=1}^2 p_i \log p_i = -p \log p - (1-p) \log (1-p)
$$

Here is the plot of entropy $$H(X)$$ *vs.* probability $$p$$. Notice how entropy is maximized when we assume that the coin is fair, i.e., $$p=0.5$$. So, in the absence of any other more strong assumptions, such as that the coin is biased, the most honest position to take is that all outcomes are equally probable. Consider maximum entropy like "presumption of innocence" ;) Beware, however, when applying the maximum entropy principle, you need to define the distribution's support. So, when the support is $$\{0,1\}$$ as in the coin flip experiment, the uniform distribution is the one with maximum entropy in the absence of any other information. When the support is $$(-\infty,+\infty)$$ and we know the sample mean and standard deviation, the normal distribution is the one that maximizes $$h(X)$$.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/normal_dist/h_vs_p_coin.png" alt="Entropy of coin flip">
</p>

[Edwin Thompson Jaynes](https://en.wikipedia.org/wiki/Edwin_Thompson_Jaynes) put it very beautifully, that the max entropy distribution is *"uniquely determined as the one which is maximally noncommittal with regard to missing information, in that it agrees with what is known, but expresses maximum uncertainty with respect to all other matters".* Therefore, this is the most principled choice. Here is a list of probability distributions and their corresponding maximum entropy constraints, taken from Wikipedia.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/normal_dist/max_entropy_examples.png" alt="Distributions with maximum entropy">
</p>

## References
1. Statistical Rethinking: A Bayesian Course with Examples in R and Stan; Second Edition, by Richard McElreath.
2. Neural Networks and Learning Machines 3rd Edition, by Simon Haykin.

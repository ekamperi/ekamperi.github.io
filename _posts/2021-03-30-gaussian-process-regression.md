---
layout: post
title:  "An introduction to Gaussian Processes"
date:   2021-03-30
categories: [mathematics]
tags: ['machine learning', 'Mathematica', 'mathematics', 'regression', 'statistics']
description: An introduction to the Gaussian Processes, particularly in the context of regression analysis
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
One of the recurring statistics topics is given some data points to perform regression analysis and establish a relationship between $$y$$ and $$x$$. This is typically done by assuming some polynomial function and estimating its coefficients via [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares). But what if we don't want to commit ourselves upfront regarding the number of parameters to use? Suppose that we'd like to consider every possible function as a candidate model for matching our data, no matter how many parameters were needed. How could we do that?

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/gaussian_process/various_fits.png" alt="Regression analysis">
</p>

This is the problem that Gaussian Processes (GP) solve. So, the idea is the following. Let's start with a distribution of all possible functions that, conceivably, could have generated our data (without actually looking at the data!). This is depicted in the following plot, where we have drawn 10 such candidate random functions. In principle, the number is infinite, but for brevity, we only drew 10. 

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gaussian_process/prior_functions.png" alt="Prior distribution over functions">
</p>

Then, as we look at the data and more measurements kick in our training dataset, we narrow down the functions that match our incoming data. In the following example, after considering 5 observations, we build-up some pretty strong confidence on how the function that generated our data should look like. The image was taken from the book *Machine Learning A Probabilistic Perspective* by Kevin P. Murphy, which is very nice, by the way.

<p align="center">
 <img style="width: 80%; height: 80%" src="{{ site.url }}/images/gaussian_process/gp_prior_posterior.png" alt="Prior and posterior distribution over functions">
</p>

Having said that, we don't really want to consider every mathematically valid function. Instead, we impose some constraints on the prior distribution over all possible functions. We want our functions to be smooth for starters because that matches our empirical knowledge about how the world generally works. Points that are close to each other in the input space (whether in the time domain, i.e. $$t_1, t_2, \ldots$$ or in the spatial domain, i.e. $$x_1, x_2, \ldots$$) are associated with $$y$$ values that are also close to each other. So, we don't really want our algorithm to favor functions that look like the one to the left.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/gaussian_process/smooth_vs_non_smooth.png" alt="Smooth vs non-smooth functions">
</p>

So, how do we impose smoothness? First, let's say that although we talk about the distribution over functions, in reality, we define the distribution over the function's values at a finite yet arbitrary set of points, say $$x_1, x_2, \ldots, x_N$$. Hence, although we model functions as really long vectors. You also need to understand that every point $$x_1, x_2, \ldots, x_N$$ is treated as a random variable, and the joint probability distribution of $$x_1, x_2, \ldots, x_N$$ is a multivariate normal distribution. Let that sink in for a moment. To generate the following function, we set up a 120-variate normal distribution and took a single 120-variate sample from it, and that was a long $$y$$ vector that corresponds to our function.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gaussian_process/function_as_vector.png" alt="Function as vector">
</p>

Ok, but if we sampled from a 120-variate Gaussian, how can we guarantee the function's smoothness? First, keep in mind that to set up a 120-variate Gaussian, we need a 120x120 covariance matrix. Each entry of the covariance matrix defines how much the $$(x_i, x_j)$$ variables are related. The trick now is to use a covariance matrix such that the values that are close together in the input space, the $$x$$'s, will produce output values that are close together, the $$y$$'s. In the following plot, $$x_1$$ and $$x_2$$ are close together, so we'd expect $$y_1$$ and $$y_2$$ to also be close (this makes the function smooth and not too wiggly). On the contrary, $$x_1$$ and $$x_N$$ are very apart, so the covariance matrix element $$C(1,N)$$ should be some tiny number.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gaussian_process/covariance_distance.png" alt="Gaussian process">
</p>

This is a plausible covariance matrix since variables near the diagonal, i.e., close together in the input space, are assigned a high value (1.0). On the contrary, the rest of the pairs are given a low value (0). Alright, but how do we actually calculate the values of the covariance matrix? We use a so-called specialized function called kernel. 

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gaussian_process/covariance_matrix_plot.png" alt="Prior distribution over functions">
</p>

A kernel function is just a fancy name for a function that accepts as input two points in the input space, i.e., $$x_i$$ and $$x_j$$, and outputs how "similar" they are based on some notion of "distance". For example, the following kernel is the so-called squared exponential that uses the exp of the squared of the Euclidean distance between two points. Therefore, if $$x_i=x_j$$, then $$K(x_i, x_j) = \exp(0)=1$$, whereas if $$\|x_i-x_j\| \to \infty$$, then $$K(x_i, x_j) \to 0$$.

{% highlight mathematica %}
{% raw %}
kernel[a_, b_] := Exp[-0.5*Norm[(a - b), 2]^2]
Plot[kernel[0, x], {x, -5, 5}, PlotRange -> All, Frame -> {True, True, False, False}, FrameLabel -> {"|x_i-x_j|^2", "Kernel value"}, Filling -> Axis]
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gaussian_process/squared_exp_kernel.png" alt="Squared exp kernel">
</p>

## A simple 1d GP prediction example

Let us consider a contrived one-dimensional problem where the response variable $$y$$ is a merely a sinusoid measured at eight equally spaced  
$$x$$ locations in the span of a single period $$[0, 2\pi]$$. This example is [taken from here](https://bookdown.org/rbg/surrogates/chap5.html), and we reimplement it with *Mathematica* (the original implementation is in *R*).

{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];

(* Define a squared exponential kernel *)
  kernel[a_, b_] := Exp[-Norm[(a - b), 2]^2]

(* These are our training data *)
  nTrainingPoints = 8; 
   X = Array[# &, nTrainingPoints, {0, 2 Pi}]; 
   Y = Sin[X];

eps = 10^-6;
Sigma = Outer[kernel, X, X] + eps*IdentityMatrix[nTrainingPoints];

nTestPoints = 100;
XX = Array[# &, nTestPoints, {-0.5, 2 Pi + 0.5}];

SigmaXX = Outer[kernel, XX, XX] + eps*IdentityMatrix[nTestPoints];
SigmaX = Outer[kernel, XX, X];
SigmaInverse = Inverse[Sigma];
Sigmap = SigmaXX - SigmaX . SigmaInverse . Transpose[SigmaX];

(* Although it is positive definite, it isn't symmetric due to small round off errors *)
  {PositiveDefiniteMatrixQ[Sigmap], SymmetricMatrixQ[Sigmap]}

(*{True, False}*)

(* Make it symmetric *)
  Sigmap = (Sigmap + Transpose@Sigmap)/2; 
   {PositiveDefiniteMatrixQ[Sigmap], SymmetricMatrixQ[Sigmap]}

(*{True, True}*)

nsamples = 100;
YY = RandomVariate[MultinormalDistribution[mup, Sigmap], nsamples];
Dimensions@YY

(*{100, 100}*)

(* Calculate 5% and 95% quantiles for uncertainty modeling *)
  quantiles = Transpose@Quantile[RandomVariate[MultinormalDistribution[mup, Sigmap], nsamples], {0.05, 0.95}]; 
   Dimensions@quantiles

(*{2, 100}*)

Show[
  Table[
   ListPlot[Transpose@{XX, YY[[i]]}, AxesLabel -> {"x", "y"}, Joined -> True, PlotStyle -> Opacity[0.1], PlotRange -> {Automatic, {-2, 2}}], {i, 1, nsamples}], 
  ListPlot[Transpose@{X, Y}, PlotStyle -> {Red, AbsolutePointSize[6]}], 
  Plot[Sin[x], {x, -0.5, 2 \[Pi] + 0.5}, PlotStyle -> Black], 
  ListPlot[Transpose@{XX, quantiles[[1]]}, PlotStyle -> {Red, Dashed},Joined -> True], 
  ListPlot[Transpose@{XX, quantiles[[2]]}, PlotStyle -> {Red, Dashed},Joined -> True]]
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gaussian_process/gp_sin_example.png" alt="Prior distribution over functions">
</p>

## References
1. Surrogates: Gaussian process modeling, design and optimization for the applied sciences by Robert B. Gramacy, 2021-02-06. https://bookdown.org/rbg/surrogates/
2. Machine Learning A Probabilistic Perspective by Kevin P. Murphy.

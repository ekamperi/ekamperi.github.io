---
layout: post
title:  "Bayesian connection to LASSO and ridge regression"
date:   2020-08-02
categories: [mathematics]
tags: ['Bayes theorem', 'machine learning', 'mathematics', 'statistics']
---

So, I was reading *"An introduction to Statistical Learning with Applications in R"*, which by the way is [freely available here](http://faculty.marshall.usc.edu/gareth-james/ISL/). In page 227 the authors provide a Bayesian point of view to both Ridge and LASSO regression. However, the mathematical proof is left as an exercise, in page 262.

Concretely, the idea is to assume the usual linear model with normal errors and combine it with a specific prior distribution for the parameters $$\beta$$. For *ridge regression*, the prior is a Gaussian with mean zero and standard deviation a function of $$\lambda$$, whereas for *LASSO*, the distribution is a double exponential (known also as Laplace distribution) with mean zero and a scale parameter a function of $$\lambda$$. As you can see in the following image, taken from the book of Gareth James et al, for LASSO, the prior distribution peaks at zero, therefore LASSO expects (a priori) many of the coefficients $$\beta$$ to be exactly equal to zero. On the other hand, for ridge regression the prior distribution is flatter at zero, therefore it expects coefficients to be normally distributed around zero.

<p align="center">
 <img style="width: 90%; height: 90%" src="{{ site.url }}/images/bayesian_lasso_ridge.png" alt="Bayesian connection to lasso and ridge regression">
</p>

We shall solve the exercise and establish the connection between Bayesian point of view and the two regularization techniques.

**(a) Suppose that $$y_i = \beta_0 + \sum_{j=1}^{p}\beta_j x_{ij} + \epsilon_i$$, where $$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$. Write out the likelihood
for the data.**

The likelihood for the data is:

$$
\mathcal{L}(Y|X,\beta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\epsilon_i^2}{2\sigma^2}\right) =
\left(\frac{1}{\sqrt{2\pi}\sigma}\right)^n \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n\epsilon_i^2\right)
$$

**(b) Assume the following prior for $$\beta: \beta_1, \beta_2, \ldots, \beta_p$$ are i.i.d. according to a double-exponential distribution with mean 0 and common scale parameter according to a double-exponential distribution with mean 0 and common scale parameter $$\beta$$: i.e.,
$$p(\beta) = (1/2b)\exp(-|\beta|/b)$$. Write out the posterior for $$\beta$$ in this setting.**

Multiplying the prior distribution $$p(\beta\|X)$$ with the likelihood $$\mathcal{L}(Y\|X,\beta)$$ we get the *posterior distribution*, up to a proporionality constant:

$$
p(\beta|X,Y) \propto \mathcal{L}(Y|X,\beta) p(\beta|X) =
\mathcal{L}(Y|X,\beta) p(\beta)
$$

Substituting we get:

$$
\mathcal{L}(Y|X,\beta) p(\beta)=
\left(\frac{1}{\sqrt{2\pi}\sigma}\right)^n \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n\epsilon_i^2\right) \left( \frac{1}{2b}\exp\left(-\frac{|\beta|}{b} \right)\right)
$$

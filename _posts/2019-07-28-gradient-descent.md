---
layout: post
title:  "Gradient descent"
date:   2019-07-28
categories: [machine learning]
---

[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for minimizing the value of a function. In the context of machine learning, we typically define some [cost (or loss) function](https://en.wikipedia.org/wiki/Loss_function) $$J(\boldsymbol{\theta})$$, where $$\boldsymbol{\theta} = (\theta_0, \theta_1, \ldots)$$ are the model's parameters that we want to tune (e.g. the weights in a neural network). The update rule for these parameters is:

$$
\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta})
$$

Where the symbol "$$\leftarrow$$" means that the variable to the left is assigned the value of the right side and $$\alpha$$ is the learning rate (how fast we update our model parameters).

This introduction is invariably accompanied by an image like this:
![gradient descent]({{ site.url }}/images/gradient_descent.png)

The intuition is that the sign of the gradient points us to the direction we have to move in order to minimize $$J$$. I'd like to present the same subject from another perspective, though, that doesn't receive much attention.

Recall that a multivariable function $$f(\mathbf{x})$$ can be written as a [Taylor series](https://en.wikipedia.org/wiki/Taylor_series):

$$f(\mathbf{x}+\boldsymbol{\delta x}) = f(\mathbf{x}) + \nabla_x f(\mathbf{x})\boldsymbol{\delta x}  + \mathcal{O}\left(\left\|\boldsymbol{\delta^2 x}\right\|\right)$$

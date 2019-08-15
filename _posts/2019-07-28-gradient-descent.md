---
layout: post
title:  "Gradient descent"
date:   2019-07-28
categories: [machine learning]
---

### Introduction
[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for minimizing the value of a function. In the context of machine learning, we typically define some [cost (or loss) function](https://en.wikipedia.org/wiki/Loss_function) $$J(\boldsymbol{\theta})$$ that informs us how well the model fits our data and $$\boldsymbol{\theta} = (\theta_0, \theta_1, \ldots)$$ are the model's parameters that we want to tune (e.g. the weights in a neural network). The update rule for these parameters is:

$$
\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta})
$$

The symbol "$$\leftarrow$$" means that the variable to the left is assigned the value of the right side and $$\alpha$$ is the learning rate (how fast we update our model parameters or how big steps we take when we change the values of the model's parameters). The algorithm is iterative and stops when convergence is achieved, i.e. when the gradient is so small that $$\boldsymbol{\theta}$$ doesn't change.

This introduction is invariably accompanied by an image like this:

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gradient_descent.png">
</p>

In the above scenario we have only $$1$$ parameter ($$w$$) and we want to minimize the cost function $$J(w)$$. The intuition is that the sign of the gradient points us to the direction we have to move in order to minimize $$J$$. I'd like to present the same subject from another perspective, though, that doesn't receive much attention.

### Connection to Taylor series
Recall that a multivariable function $$f(\mathbf{x})$$ can be written as a [Taylor series](https://en.wikipedia.org/wiki/Taylor_series):

$$f(\mathbf{x}+\delta \boldsymbol{x}) = f(\mathbf{x}) + \nabla_x f(\mathbf{x})\delta \boldsymbol{x} + \mathcal{O}\left(\left\|\delta^2 \boldsymbol{x}\right\|\right)$$

Suppose that we want to minimize $$f(\mathbf{x})$$, by taking a step from $$\mathbf{x}$$ to $$\mathbf{x} + \mathbf{\delta x}$$. This means that we would like $$f(\mathbf{x} + \delta\mathbf{x})$$ to be smaller than $$f(\mathbf{x})$$. If we substitute the formula from the Taylor expansion of $$f(\mathbf{x} + \delta\mathbf{x})$$, we get:

$$f(\mathbf{x}+\delta \boldsymbol{x}) < f(\mathbf{x}) \Leftrightarrow
f(\mathbf{x}) + \nabla_x f(\mathbf{x})\delta \boldsymbol{x} < f(\mathbf{x}) \Leftrightarrow
\nabla_x f(\mathbf{x})\delta \boldsymbol{x} < 0
$$

To restate our problem: what is the optimal $$\delta\mathbf{x}$$ step that we need to take so that the quantity $$\nabla_x f(\mathbf{x})\delta \boldsymbol{x}$$ is minimized?

Keep in mind that $$\nabla_x f(\mathbf{x})$$ and $$\delta \boldsymbol{x}$$ are just vectors, therefore we need to minimize the dot product $$\mathbf{u} \cdot \mathbf{v}$$, with $$\mathbf{u} = \nabla_x f(\mathbf{x})$$ and $$\mathbf{v} = \delta \mathbf{x}$$.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/gradient_descent2.png">
</p>

Since $$\mathbf{u} \cdot \mathbf{v} = \left\|u\right\| \left\|v\right\| \cos(\mathbf{u}, \mathbf{v})$$ it follows that when the angle between $$\mathbf{u}$$ and $$\mathbf{v}$$ is $$\varphi = -\pi$$, then the dot product takes its minimum value. Therefore $$\delta \mathbf{x} = - \nabla_x f(\mathbf{x})$$.
, 
### The Hessian matrix
Previously, when expanding $$f(\mathbf{x})$$ we considered only the first-order terms in the Taylor series. By also taking into account the second-order terms we get:

$$f(\mathbf{x}+\delta \boldsymbol{x}) = f(\mathbf{x}) + \nabla_x f(\mathbf{x})\delta \boldsymbol{x} + \frac{1}{2} \delta\mathbf{x}^T \mathbf{H}\delta\mathbf{x} + \mathcal{O}\left(\left\|\delta^3 \boldsymbol{x}\right\|\right)$$

Where $$\mathbf{H} = \nabla_x^2f(\mathbf{x})$$ is the [Hessian matrix]($https://en.wikipedia.org/wiki/Hessian_matrix).

It's interesting to think what happens when the gradient becomes zero, i.e. $$\nabla_x f(\mathbf{x}) = 0$$. Then:

$$
\begin{align}
f(\mathbf{x}+\delta \boldsymbol{x})
&= f(\mathbf{x}) + \underbrace{\nabla_x f(\mathbf{x})\delta \boldsymbol{x}}_\text{zero} + \frac{1}{2} \delta\mathbf{x}^T \mathbf{H}\delta\mathbf{x} + \mathcal{O}\left(\left\|\delta^3 \boldsymbol{x}\right\|\right)\\
f(\mathbf{x}+\delta \boldsymbol{x}) &= f(\mathbf{x}) + \frac{1}{2} \delta\mathbf{x}^T \mathbf{H}\delta\mathbf{x} + \mathcal{O}\left(\left\|\delta^3 \boldsymbol{x}\right\|\right)
\end{align}
$$

So the gradient is zero, which means that we are not moving towards any direction. At this point we have not yet assumed anything about the "shape" of the function, i.e. whether it was [convex](https://en.wikipedia.org/wiki/Convex_function) or non-convex.
 Are we on a global minimum? On a local minum? Or on a [saddle point](https://en.wikipedia.org/wiki/Saddle_point)?

The answer to this question is hidden in the product $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x}$$, which informs us about the *local curvature* in the neighborhood of $$\mathbf{x}$$.

* $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x} > 0$$: We are stuck on a local minimum.
* $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x} < 0$$: We are stuck on a local maximum.
* $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x}$$ has both positive and negative eigenvalues: We are stuck on a saddle point.
* None of the above: We have no clue. We need even higher-order data to figure it out.

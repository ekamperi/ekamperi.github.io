---
layout: post
title:  "Gradient descent"
date:   2019-07-28
categories: [machine learning]
---

### Introduction
[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for minimizing the value of a function. In the context of machine learning, we typically define some [cost (or loss) function](https://en.wikipedia.org/wiki/Loss_function) $$J(\boldsymbol{\theta})$$ that informs us how well the model fits our data and $$\boldsymbol{\theta} = (\theta_0, \theta_1, \ldots)$$ are the model's parameters that we want to tune (e.g. the weights in a neural network or simply the coefficients in a linear regression problem of the form $$y = \theta_0 + \theta_1 x$$). The update rule for these parameters is:

$$
\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta})
$$

The symbol "$$\leftarrow$$" means that the variable to the left is assigned the value of the right side and $$\alpha$$ is the learning rate (how fast we update our model parameters or how big steps we take when we change the values of the model's parameters). The algorithm is iterative and stops when convergence is achieved, i.e. when the gradient is so small that $$\boldsymbol{\theta}$$ doesn't change.

This introduction is invariably accompanied by an image like this:

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/gradient_descent.png">
</p>

In the above scenario we only have $$1$$ parameter, $$w$$, and we want to minimize the cost function $$J(w)$$. The intuition is that the sign of the gradient points us to the direction we have to move in order to minimize $$J$$. Imagine that we have many parameters, then we are navigating inside a $$D-$$dimensional space. But since it's easier to visualize with $$D=1$$ or $$D=2$$, most people use the above image as an example (or a 2D version of it).

Although the post is not about gradient descent per se, let's just take on a really simple example.

{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];
(* Generate some points along the 5x+7 line plus some noise *)
data = Table[{x + RandomReal[], 7 + 5 x + RandomReal[]}, {x, 0, 10, 
    0.1}];

{% endraw %}
{% highlight %}

I'd like to present the same subject from a slightly different perspective, though, that doesn't receive much attention.

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

### The Hessian matrix

It's interesting to consider what happens when the gradient becomes zero, i.e. $$\nabla_x f(\mathbf{x}) = 0$$. Since it's zero, this means that we are not moving towards any direction. At this point we have not yet assumed anything about the "shape" of the function, i.e. whether it was [convex](https://en.wikipedia.org/wiki/Convex_function) or non-convex. So, are we on a minimum? Are we on a [saddle point](https://en.wikipedia.org/wiki/Saddle_point)?

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/optimization_shape.png">
</p>
Image taken from [here](https://www.offconvex.org/2016/03/22/saddlepoints).

The answer to this question is hidden in the second-order terms of the Taylor series, which inform us about the *local curvature* in the neighborhood of $$\mathbf{x}$$. Previously, when expanding $$f(\mathbf{x})$$ we considered only the first-order terms. By also taking into account the second-order terms we get:

$$f(\mathbf{x}+\delta \boldsymbol{x}) = f(\mathbf{x}) + \nabla_x f(\mathbf{x})\delta \boldsymbol{x} + \frac{1}{2} \delta\mathbf{x}^T \mathbf{H}\delta\mathbf{x} + \mathcal{O}\left(\left\|\delta^3 \boldsymbol{x}\right\|\right)$$

Where $$\mathbf{H} = \nabla_x^2f(\mathbf{x})$$ is the [Hessian matrix]($https://en.wikipedia.org/wiki/Hessian_matrix).

Now we are able to explore what's happening in the case of $$\nabla_x f(\mathbf{x}) = 0$$:

$$
\begin{align}
f(\mathbf{x}+\delta \boldsymbol{x})
&= f(\mathbf{x}) + \underbrace{\nabla_x f(\mathbf{x})\delta \boldsymbol{x}}_\text{zero} + \frac{1}{2} \delta\mathbf{x}^T \mathbf{H}\delta\mathbf{x} + \mathcal{O}\left(\left\|\delta^3 \boldsymbol{x}\right\|\right)\\
f(\mathbf{x}+\delta \boldsymbol{x}) &= f(\mathbf{x}) + \frac{1}{2} \delta\mathbf{x}^T \mathbf{H}\delta\mathbf{x} + \mathcal{O}\left(\left\|\delta^3 \boldsymbol{x}\right\|\right)
\end{align}
$$

The same argument as before can be applied. We want to take a step from $$\mathbf{x}$$ to $$\mathbf{x} + \mathbf{\delta x}$$ and $$f(\mathbf{x} + \delta\mathbf{x})$$ be smaller than $$f(\mathbf{x})$$. Therefore, we need to find a vector $$\delta \mathbf{x}$$ for which $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x} < 0$$ and move along it.

To sum up regarding the product $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x}$$ we have these cases:

* $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x} > 0$$: We are on a local minimum.
* $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x} < 0$$: We are on a local maximum.
* $$\delta\mathbf{x}^T \mathbf{H} \delta \mathbf{x}$$ has both positive and negative eigenvalues: We are on a saddle point.
* None of the above: We have no clue. We need even higher-order data to figure it out.

In the early days of neural networks, it was believed that the proliferation of local minima would be a problem, in the sense that gradient descent would get stuck in them. But it turned out that this was not the case. Instead, the proliferation of saddle points, especially in high dimensional problems (e.g. neural networks), is the core of the problem (Dauphin et al, 2014). Such saddle points are surrounded by plateaus where the error is high and they can dramatically slow down optimization, giving the impression of the existence of a local minimum.

### References
1. Dauphin Y, Pascanu R, Gulcehre C, Cho K, Ganguli S, Bengio Y. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization [Internet]. arXiv [cs.LG]. 2014. Available from: http://arxiv.org/abs/1406.2572

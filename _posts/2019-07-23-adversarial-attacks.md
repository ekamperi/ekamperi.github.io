---
layout: post
title:  "Adversarial attacks on neural networks"
date:   2019-07-23 15:28:56 +0000
categories: jekyll update
---

### Introduction
Adversarial means "involving or characterized by conflict or opposition". In the context of neural networks, "adversarial examples" refer to specially crafted inputs whose purpose is to force the neural network to misclassify them. These examples (or attacks) are grouped into *targeted*, when a legitimate input is changed by some imperceptible amount and the new input is misclassified by the network. E.g.

[![Example of targeted adversarial attack][1]][1]

Source: Goodfellow IJ, Shlens J, Szegedy C. Explaining and Harnessing Adversarial Examples [Internet]. arXiv [stat.ML]. 2014. Available from [here](http://arxiv.org/abs/1412.6572).

And to *non-targeted*, when you just want to come up with some random input that results to a specific output, even if it looks like noise, as in the following network which is trained to recognize digits:

[![Example of non-targeted adversarial attack][2]][2]

Adversarial attacks could potentially pose a security threat for real-world machine learning applications, such as self-driving cars, facial recognition applications, etc. Szegedy et al. (2014) showed that an adversarial example that was designed to be misclassified by a model $$M1$$ is often also misclassified by a model $$M2$$. This "adversarial transferability" property means that it is possible to exploit a machine learning system *without* having any knowledge of its underlying model ($$M2$).

### Generation
#### Targeted example
A fast method for constructing a targeted example is via $$\mathbf{x}_\text{adv} = \mathbf{x} + \overbrace{\epsilon sign\left({\nabla_x J(\mathbf{x})}\right)}^{\text{perturbation factor } \mathbf{\alpha}}$$, where $$\mathbf{x}$$ is the legitimate input you target (e.g. your original "panda" image), $$\epsilon$$ is some small number (whose value you determine by trial-and-error) and $$J(\mathbf{x})$$ is the cost as a function of input $$\mathbf{x}$$. The gradient $$\nabla_x J(\mathbf{x})$$ with respect to input $$\mathbf{x}$$ can be calculated with [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). This method is simple and computationally efficient when compared to other more complex methods (it requires only 1 gradient calculation after all), however it usually has a lower success rate.

Let's see, how we could derive this formula.

Our cost function is $$J(\hat{y}, y) = J(h(x),y)$$, where $$h(x)$$ is the hypothesis function (basically the value of the network's output layer). Normally, when training a network we want to minimize the cost function $$J$$ so that our network's prediction $$\hat{y}$$ is as close to the true value $$y$$ as possible. When constructing a targeted adersarial attack, though, we want the exact opposite. We want to come up with a new input $$\mathbf{x}_\text{adv}$$, such that the predicted value $$\hat{y}$$ is as much away from $$y$$ as possible. The maximization of the cost function $$J$$ is subject to a constraint, namely that the change we introduce (the perturbation factor $$\mathbf{\alpha}$$) is very small (so that it goes unnoticed by a human).

More formally our goal can be expressed as:

$$\max J(\hat{y}, y)=\max_{\left\| a \right\|\le \epsilon} J(h(x+\alpha), y)$$

In order to solve this optimization problem, we will linearize $$J$$ with respect to input $$x$$. Let us recall that a multivariable function $$f(\mathbf{x})$$ can be written as a [Taylor series](https://en.wikipedia.org/wiki/Taylor_series):

$$f(\mathbf{x}+\mathbf{\alpha}) = f(\mathbf{x}) + \mathbf{\alpha} \nabla_x f(\mathbf{x}) + \mathcal{O}\left(\left\|\alpha^2\right\|\right)$$, where $$\mathbf{x}=(x_1, x_2, \ldots)$$, $$\mathbf{\alpha} = (\alpha_1, \alpha_2,\ldots)$$ and $$\nabla_x$$ the gradient with respect to input $$\mathbf{x}$$.

By applying the above formula to the cost function $$J$$ we get:

$$J(h(x+\alpha),y) = \underbrace{J(h(x),y)}_{\text{fixed}}+\alpha \nabla_xJ(h(x),y)+\mathcal{O}(\left\|\alpha^2\right\|)$$

Therefore:

$$\max_{\left\| a \right\|\le \epsilon} J(h(x+a),y) = \max_{\left\| a \right\|\le \epsilon} \alpha \nabla_x J(h(x),y)$$

The value of $$\alpha$$ that maximizes the above quantity, under the [infinity norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Maximum_norm_(special_case_of:_infinity_norm,_uniform_norm,_or_supremum_norm)) is $$\alpha_i = \epsilon sign(\nabla_x J)_i$$. The proof goes as this:

$$
\begin{align}
\max_{\left\| \alpha \right\|\le \epsilon} \nabla_x J(h(x),y) \alpha
&= \max_{\left\| \epsilon a' \right\|\le \epsilon} \nabla_x J(h(x),y) \epsilon \alpha'\\
&=\epsilon \max_{\left\| a' \right\|\le 1} \nabla_x J(h(x),y) \alpha'\\
&=\epsilon \max_{\left\| a \right\|\le 1} \nabla_x J(h(x),y) \alpha\\
&= \epsilon \left\| \nabla_x J(h(x), y)\right\|_{*}
\end{align}
$$

Where $$\left\| \cdot \right\|_{*}$$ is the [dual norm](https://en.wikipedia.org/wiki/Dual_norm#Mathematical_Optimization).

Since we have assumed the infinity norm:
$$\left\| \alpha \right\|_\infty = \max(\left|\alpha_1\right|, \left|\alpha_2\right|,\ldots) \le 1$$, it holds that $$\left\|\cdot\right\|_{*} = \left\|\cdot\right\|_1$$. Therefore:

$$
\max_{\left\|\alpha\right\| \le \epsilon} \nabla_x J(h(x),y) \alpha = \epsilon \left\| \nabla_x J(h(x), y)\right\|_{1}
$$

Finally we solve the following equation:

$$
\begin{align}
\sum_i {\nabla_x J(h(x),y)}_i \alpha_i &= \epsilon \sum_i \left|\nabla_x J(h(x), y)_i\right| \Rightarrow\\
\sum_i {\nabla_x J(h(x),y)}_i \alpha_i &= \epsilon \sum_i  sign\left(\nabla_x J(h(x),y)_i\right)\nabla_x J(h(x), y)_i
\end{align}
$$

Therefore:

$$
\sum_i {\nabla_x J(h(x),y)}_i \left(\alpha_i - \epsilon sign\left(\nabla_x J(h(x),y)_i\right)\right) = 0 \Rightarrow\\
\alpha_i = \epsilon sign\left(\nabla_x J(h(x),y)_i\right)
$$

#### Non-targeted example
For non-targeted attacks, the value of $$\mathbf{x}_\text{adv}$$ can be found via [gradient descent][3] as the one that minimizes the following definition of cost function $$J$$, starting with some random value for $$\mathbf{x}$$.

$$
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
J(\mathbf{x}) = \frac{1}{2}\norm{y(\mathbf{x})-\mathbf{y}_\text{target}}_2^2 
=\frac{1}{2}\norm{h_\Theta(\mathbf{x}) - \mathbf{y}_\text{target}}_2^2
$$

$$\mathbf{y}_\text{target}$$ is the target class value (e.g. $$\mathbf{y}_\text{target} = [0,0,0,1,0,0,0,0,0,0]$$ in the above image), $$h_\Theta(\mathbf{x})$$ is the output of the network for some input $$\mathbf{x}$$ and $$\mathbf{x}$$ gets updated with:

$$
x_{j,\text{new}} = x_{j,\text{old}} - \alpha \frac{\partial }{\partial x_j}J(\mathbf{x})
$$

[Useful link][4] on `NetPortGradient[]`.


  [1]: https://i.stack.imgur.com/NPbEel.png
  [2]: https://i.stack.imgur.com/h7mGDl.png
  [3]: https://en.wikipedia.org/wiki/Gradient_descent
  [4]: https://mathematica.stackexchange.com/questions/185948/netportgradient-output-port-restriction
 
### References
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. ICLR, abs/1312.6199, 2014. URL
http://arxiv.org/abs/1312.6199.

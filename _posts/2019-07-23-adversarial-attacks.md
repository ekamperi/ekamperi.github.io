---
layout: post
title:  "Adversarial attacks on neural networks"
date:   2019-07-23 15:28:56 +0000
categories: jekyll update
---

### Introduction
Adversarial means "involving or characterized by conflict or opposition". In the context of neural networks, "adversarial examples" refer to specially crafted inputs whose purpose is to cause the neural network to misclassify them. These examples (or attacks) are grouped into *targeted*, where a legitimate input is changed by some imperceptible amount and the new input is misclassified by the network. E.g.

[![Example of targeted adversarial attack][1]][1]

Source: Goodfellow IJ, Shlens J, Szegedy C. Explaining and Harnessing Adversarial Examples [Internet]. arXiv [stat.ML]. 2014. Available from [here](http://arxiv.org/abs/1412.6572).

And to *non-targeted*, where you just want to come up with some random input that leads to a specific output, like:

[![Example of non-targeted adversarial attack][2]][2]

### Generation
#### Targeted example
A fast method for constructing a targeted example is $$x_\text{adv} = x + \underbrace{\epsilon sign\left({\nabla_x J(x)}\right)}_{\text{perturbation factor}}$$, where $$x$$ is the legitimate input you target, $$\epsilon$$ some small number and $$J(x)$$ the cost as a function of input $$x$$. Let's see how we could derive to this formula.

Let us recall that a multivariable function $$f(\mathbf{x})$$ can be written as a Taylor series:
$$f(\mathbf{x}+\mathbf{\alpha}) = f(\mathbf{x}) + \mathbf{\alpha} \nabla_x f(\mathbf{x}) + O\left(\left\|\alpha^2\right\|\right)$$, where $$\mathbf{x}=(x_1, x_2, \ldots)$$ and $$\mathbf{alpha} = (\alpha_1, \alpha_2,\ldots)$$

Similarly, we expand our cost function $$J$$:
$$J(h(x+\alpha),y) = \underbrace{J(h(x),y)}_{\text{fixed}}+\alpha \nabla_xJ(h(x),y)+O(\alpha^2)$$

Our goal is to maximize the cost function $$J$$:
$$\max J(\hat{y}, y)=\max_{\left\| a \right\|\le \epsilon} J(h(x+\alpha), y)$$

$$\max_{\left\| a \right\|\le \epsilon} J(h(x+a),y) = \max_{\left\| a \right\|\le \epsilon} \alpha \nabla_x J(h(x),y)$$
#### Non-targeted example
For non-targeted attacks, the value of $$x_\text{adv}$$ can be found via [gradient descent][3] as the one that minimizes the following definition of cost function $$J$$, starting with some random value for $$x$$.

$$
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
J(x) = \frac{1}{2}\norm{y(x)-y_\text{adv}}_2^2 
=\frac{1}{2}\norm{h_\Theta(x) - y_\text{adv}}_2^2
$$

$$y_\text{adv}$$ is the goal value (e.g. $$y_\text{adv} = [0,0,0,1,0,0,0,0,0,0]$$ in the above image), $$h_\Theta(x)$$ is the output of the network for some input $$x$$ and $$x$$ gets updated with:

$$
x_{j,\text{new}} = x_{j,\text{old}} - \alpha \frac{\partial }{\partial x_j}J(x)
$$

**Questions:**

1. Is it true that `net[x0, NetPortGradient["Input"]]` returns the value of $$\frac{\partial }{\partial x}h_\Theta(x)$$ for some value $$x=x_0$$?

2. Is it normal for `NetPortGradient[]` to return values very close to zero as we go deeper into the network? Does this have anything to do with [vanishing gradients][4], although that term usually refers to the gradient of loss function with respect to _weights_ rather than input?

~~~~
        plg[x_, inp_] := ListPlot[#, Joined -> True, PlotRange -> All, 
         InterpolationOrder -> 1, Frame -> {True, True, False, False}, 
         FrameLabel -> {"Layer number", 
           "Total@Abs@Gradient\nwrt to input"}, Filling -> Bottom] &@
        Table[Total@Abs@Flatten@NetTake[x, k][inp, NetPortGradient["Input"]],
        {k, 1, NetInformation[x, "LayersCount"]}];
~~~~

[![enter image description here][5]][5]

  3. If it is typical for the gradient of the last layer with respect to input to be practically zero, how would the update formulas work?

[Useful link][6] on `NetPortGradient[]`.


  [1]: https://i.stack.imgur.com/NPbEel.png
  [2]: https://i.stack.imgur.com/h7mGDl.png
  [3]: https://en.wikipedia.org/wiki/Gradient_descent
  [4]: https://en.wikipedia.org/wiki/Vanishing_gradient_problem
  [5]: https://i.stack.imgur.com/z8PU0l.png
  [6]: https://mathematica.stackexchange.com/questions/185948/netportgradient-output-port-restriction
 

---
layout: post
title:  "Adversarial attacks on neural networks"
date:   2019-07-23 15:28:56 +0000
categories: [machine learning]
---

### Introduction
Adversarial means "involving or characterized by conflict or opposition". In the context of neural networks, "adversarial examples" refer to specially crafted inputs whose purpose is to force the neural network to misclassify them. These examples (or attacks) are grouped into *non-targeted*, when a legitimate input is changed by some imperceptible amount and the new input is misclassified by the network. E.g.

[![Example of targeted adversarial attack][1]][1]

Source: Goodfellow IJ, Shlens J, Szegedy C. Explaining and Harnessing Adversarial Examples [Internet]. arXiv [stat.ML]. 2014. Available from [here](http://arxiv.org/abs/1412.6572).

And to *targeted*, when you want to force the model to predict a *specific output* ($$y_\text{target}$$), even if it looks like noise, as in the following network which is trained to recognize digits:

[![Example of non-targeted adversarial attack][2]][2]

Adversarial attacks could potentially pose a security threat for real-world machine learning applications, such as self-driving cars, facial recognition applications, etc. Szegedy et al. (2014) showed that an adversarial example that was designed to be misclassified by a model $$M1$$ is often also misclassified by a model $$M2$$. This "adversarial transferability" property means that it is possible to exploit a machine learning system *without* having any knowledge of its underlying model ($$M2$$).

### Generation
#### Non-Targeted example
A fast method for constructing a targeted example is via $$\mathbf{x}_\text{adv} = \mathbf{x} + \overbrace{\epsilon sign\left({\nabla_x J(\mathbf{x})}\right)}^{\text{perturbation factor } \mathbf{\alpha}}$$, where $$\mathbf{x}$$ is the legitimate input you target (e.g. your original "panda" image), $$\epsilon$$ is some small number (whose value you determine by trial-and-error) and $$J(\mathbf{x})$$ is the cost as a function of input $$\mathbf{x}$$. The gradient $$\nabla_x J(\mathbf{x})$$ with respect to input $$\mathbf{x}$$ can be calculated with [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). This method is simple and computationally efficient when compared to other more complex methods (it requires only 1 gradient calculation after all), however it usually has a lower success rate.

Let's see, how we could derive this formula.

Our cost function is $$J(\hat{y}, y) = J(h(x),y)$$, where $$h(x)$$ is the hypothesis function (basically the value of the network's output layer). Normally, when training a network we want to minimize the cost function $$J$$ so that our network's prediction $$\hat{y}$$ is as close to the true value $$y$$ as possible. When constructing a targeted adersarial attack, though, we want the exact opposite. We want to come up with a new input $$\mathbf{x}_\text{adv}$$, such that the predicted value $$\hat{y}$$ is as much away from $$y$$ as possible. The maximization of the cost function $$J$$ is subject to a constraint, namely that the change we introduce (the perturbation factor $$\mathbf{\alpha}$$) is very small (so that it goes unnoticed by a human).

More formally our goal can be expressed as:

$$\max J(\hat{y}, y)=\max_{\left\| a \right\|\le \epsilon} J(h(x+\alpha), y)$$

In order to solve this optimization problem, we will linearize $$J$$ with respect to input $$\mathbf{x}$$. Let us recall that a multivariable function $$f(\mathbf{x})$$ can be written as a [Taylor series](https://en.wikipedia.org/wiki/Taylor_series):

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

Finally we solve the following equation. Notice that $$\|x\| = sign(x)x$$:

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

#### Example code in Mathematica

{% raw %}
~~~~
ClearAll["Global`*"];

(* Load the neural network model *)
netOriginalModel = 
 NetModel["SqueezeNet V1.1 Trained on ImageNet Competition Data"];

(* This is an African wild dog image we will use as input X *)
legitX =
 ImageResize[#, {227, 227}] &@
  RemoveAlphaChannel@
   Import[
    "https://farm1.static.flickr.com/159/384015403_25353f2a7d.jpg"];

(* Remove the decoder from the output, so that we get a vector y as the output *)
netM = NetReplacePart[netOriginalModel, "Output" -> None];

(* Find the index of the African wild dog and construct the true output vector ytrue *)
idx = Ordering[netM[legitX]][[-1]];
ytrue = ConstantArray[0, 1000]; ytrue[[idx]] = 1;

(* Calculate the signed gradients *)
dy[x_] := netM[x] - ytrue;
calcGrads[x_] :=
  ArrayReshape[#, {227, 227, 3}] &@
   Sign@netM[<|"Input" -> x, NetPortGradient["Output"] -> dy[x]|>, 
     NetPortGradient["Input"]];

(* new image = old image + epsilon * signed_gradients *)
getAdv[x_, epsilon_] := Image[ImageData@x + epsilon*calcGrads[x]]

tnew[epsilon_] :=
 With[{newImage = getAdv[legitX, epsilon]},
  {{legitX, netOriginalModel[legitX]}, {newImage, 
    netOriginalModel[newImage]}}]

(* epsilon = 0.0815 *)
tnew[0.0815]
~~~~
{% endraw %}

And this is the result:

![adversarial example]({{ site.url }}/images/adversarial.png)

#### Targeted example

When we want to steer the model's output to some specific class, $$y_\text{target}$$, instead of increasing the cost function $$J(\hat{y}, y_\text{true})$$, we instead decrease the cost function between the predicted $$\hat{y}$$ and the target class $$y_\text{target}$$.

Therefore, instead of $$\mathbf{x}_\text{adv} = \mathbf{x} + \overbrace{\epsilon sign\left({\nabla_x J(\mathbf{x},y_\text{true})}\right)}^{\text{perturbation factor } \mathbf{\alpha}}$$, we do $$\mathbf{x}_\text{adv} = \mathbf{x} - \overbrace{\epsilon sign\left({\nabla_x J(\mathbf{x},y_\text{target})}\right)}^{\text{perturbation factor } \mathbf{\alpha}}$$.

Instead of doing just one update, we could use an *iterative* approach where the value of $$\mathbf{x}_\text{adv}$$ is iteratively calculated via [gradient descent][3], as the one that minimizes the following definition of cost function $$J$$ (starting with some random value for $$\mathbf{x}$$):

$$
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
J(\mathbf{x}, \mathbf{y_\text{target}}) = \frac{1}{2}\norm{y(\mathbf{x})-\mathbf{y}_\text{target}}_2^2 
$$

Here $$\mathbf{y}_\text{target}$$ is the target class value (e.g. $$\mathbf{y}_\text{target} = [0,0,0,0,1,0,0,0,0,0]$$,  $$y(\mathbf{x})$$ is the output of the network for some input $$\mathbf{x}$$. The update rule for $$\mathbf{x}$$ is the following:

$$
x_{j,\text{new}} = x_{j,\text{old}} - \alpha \frac{\partial }{\partial x_j}J(\mathbf{x})
$$

Let us perform a targeted attack on the LeNet network, which was developed by [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) and his collaborators while they experimented with machine learning solutions for classification on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). MNIST is a large database of handwritten digits that is commonly used for training image classification systems.

{% raw %}
~~~~
ClearAll["Global`*"];
netOriginalModel = NetModel["LeNet Trained on MNIST Data"]
trainingData = ResourceData["MNIST", "TrainingData"];
Take[RandomSample[trainingData], 10]
~~~~
{% endraw %}
![MNIST example]({{ site.url }}/images/mnist.jpg)

We start with some random image as $$\mathbf{x}$$:
{% raw %}
~~~~
(* Start with some random image *)
randomX = RandomImage[{0, 1}, ImageDimensions@testData[[1, 1]]];

netM = NetReplacePart[netOriginalModel, "Output" -> None]

p1 = DiscretePlot[netM[randomX][[k + 1]], {k, 0, 9}, PlotRange -> All,
   Frame -> {True, True, False, False}, 
   FrameLabel -> {"Class", "Probability"}, 
   FrameTicks -> {Range[0, 9], Automatic}, 
   PlotLabel -> "LeNet output on a random image", PlotStyle -> AbsolutePointSize[5]];
Style[Grid[{{Image[randomX, ImageSize -> Small], p1}}], ImageSizeMultipliers -> 1]
~~~~
{% endraw %}
![LeNet]({{ site.url }}/images/lenet1.png)

As you can see in the above image when given some random input (noise), LeNet outputs some probabilities for each class. Our goal is to come up with an image $$\mathbf{x}$$ such as that the network will classify it as -say- digit $$4$$. Therefore, the ideal output vector $$\mathbf{y_\text{target}}$$ is $$[0,0,0,0,1,0,0,0,0,0]$$.

{% raw %}
~~~~
(* The target output vector, ytarget *)
ytarget = ConstantArray[0, 10]; ytarget[[5]] = 1; ytarget
(* {0, 0, 0, 0, 1, 0, 0, 0, 0, 0} *)

(* Calculate signed gradients *)
dy[x_] := netM[x] - ytarget;
calcGrads[x_] :=
 ArrayReshape[#, Dimensions@ImageData@randomX] &@
  Sign@netM[<|"Input" -> x, NetPortGradient["Output"] -> dy[x]|>, 
    NetPortGradient["Input"]]

(* Run some iterations of gradient descent with a learning rate 0.01 *)
(* Save the values of cost function J so that we plot it *)
errors =
  Reap[
    For[i = 1, i <= 30, i++,
     randomX = Image[ImageData@randomX - 0.01*calcGrads[randomX]];
     Sow@Total[0.5 (netM[randomX] - ytarget)^2]
     ]
    ][[2, 1]];
~~~~
{% endraw %}

Here is the plot of cost function $$J$$ vs. the iterations of gradient descent:
![Cost function]({{ site.url }}/images/cost_function.png)


{% raw %}
~~~~
p2 = DiscretePlot[netM[randomX][[k + 1]], {k, 0, 9}, PlotRange -> All,
    Frame -> {True, True, False, False}, 
   FrameLabel -> {"Class", "Probability"}, 
   FrameTicks -> {Range[0, 9], Automatic}, 
   PlotLabel -> "LeNet output on adversarial input", 
   PlotStyle -> AbsolutePointSize[5]];
Style[Grid[{{p1, p2}}], ImageSizeMultipliers -> 1]
~~~~
{% endraw %}

![LeNet output on random (left) and adversarial (right) example]({{ site.url }}/images/cost_function.png)

Notice how our adversarial input image bears no resemblance to a $$4$$ digit, yet the network is "100% sure" that this is the digit $$4$$.

{% raw %}
~~~~
Style[Grid[{{Image[randomX, ImageSize -> Small], p2}}], 
 ImageSizeMultipliers -> 1]
~~~~
{% endraw %}

[Useful link][4] on `NetPortGradient[]`.

  [1]: https://i.stack.imgur.com/NPbEel.png
  [2]: https://i.stack.imgur.com/h7mGDl.png
  [3]: https://en.wikipedia.org/wiki/Gradient_descent
  [4]: https://mathematica.stackexchange.com/questions/185948/netportgradient-output-port-restriction
 
### References
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. ICLR, abs/1312.6199, 2014. URL
http://arxiv.org/abs/1312.6199.

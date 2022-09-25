---
layout: post
title: "Custom training loops with Pytorch"
date:   2022-09-25
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'neural networks', 'pytorch', 'statistics']
description: How to create custom training loops with Pytorch

---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
[In a previous post](https://ekamperi.github.io/mathematics/2020/12/20/tensorflow-custom-training-loops.html), we saw a couple of examples on how to construct a custom training loop, define a custom loss function, have Tensorflow automatically compute the gradients of the loss function with respect to the trainable parameters, and then update the model. In this post, we will do the same, but this time we are going to use PyTorch. It's been a while that I wanted to switch from Tensorflow to Pytorch, and what a better way than start from the basics?

## Fit linear regression model to data by minimizing MSE
### Generate training data
We are going to generate some data coming from a quadratic model, i.e., $$y = a x^2 + b x + c$$, and we are also going to add some noise, to make the setup look a bit more realistic, as in real world.

{% highlight python %}
{% raw %}
import torch
import matplotlib.pyplot as plt

def generate_dataset(npts=100):
    x = torch.linspace(0, 1, npts)
    y = 20*x**2 + 5*x - 3
    y += torch.randn(npts)  # Add some noise
    return x, y

x, y_true = generate_dataset()

plt.scatter(x, y_true)
plt.xlabel('$x$')
plt.ylabel('$y_{true}$')
plt.grid()
plt.title('Dataset')
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/pytorch_custom_loop/dataset.png" alt="Dataset for regression">
</p>

### Define a model with trainable parameters
In this step, we are defining a model, the $$y = f(x)$$. Given the model's parameters, $$a, b, c$$, and an input $$x$$, $$x$$ being a tensor, we will calculate the output tensor $$y_\text{pred}$$:

{% highlight python %}
{% raw %}
def f(x, params):
    a, b, c = params
    return a * (x**2) + b * x + c
{% endraw %}
{% endhighlight %}


### Define a custom loss function
{% highlight python %}
{% raw %}
def mse(y_pred, y_true):
    """Returns the mean squared error between y_pred and y_true"""
    return ((y_pred - y_true)**2).mean()
{% endraw %}
{% endhighlight %}

{% highlight python %}
{% raw %}
params = torch.randn(3).requires_grad_()
y_pred = f(x, params)
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
def plot_pred_vs_true(title):
    plt.scatter(x, y_true, label='y_true', marker='o', s=50, alpha=0.75)
    plt.plot(x, y_pred.detach().numpy(), label='y_pred', c='r', linewidth=4)
    plt.legend()
    plt.title(title)
    plt.xlabel('x')

plot_pred_vs_true('Before training')
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/pytorch_custom_loop/before_training.png" alt="Regression with Pytorch">
</p>

{% highlight python %}
{% raw %}
def apply_step():
    lr = 1e-3                                   # Set learning rate to 0.001
    y_pred = f(x, params)                       # Calculate the y given x and a set of parameters' values
    loss = mse(y_pred=y_pred, y_true=y_true)    # Calculate the loss between y_pred and y_true
    loss.backward()                             # Calculate the gradient of loss tensor w.r.t. graph leaves
    params.data -= lr * params.grad.data        # Update parameters' values using gradient descent
    params.grad = None                          # Zero grad since backward() accumulates by default gradient in leaves
    return y_pred, loss.item()                  # Return the y_pred, along with the loss as a standard Python number
{% endraw %}
{% endhighlight %}


### Run the custom training loop
{% highlight python %}
{% raw %}
epochs = 15000
history = []
for i in range(epochs):
    y_pred, loss = apply_step()
    history.append(loss)

plt.plot(history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE vs. Epoch')
plt.grid()
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/pytorch_custom_loop/history.png" alt="History of MSE loss">
</p>

### Final results
Finally, we superimpose the dataset with the best quadratic regression model PyTorch converged to:

{% highlight python %}
{% raw %}
plot_pred_vs_true('After training')
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/pytorch_custom_loop/after_training.png" alt="Regression with Pytorch">
</p>

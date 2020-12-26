---
layout: post
title:  "Trainable probability distributions with Tensorflow"
date:   2020-12-20
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'optimization', 'statistics', 'Tensorflow']
description: How to create trainable probability distributions with Tensorflow
---

In the previous post, we fit a Gaussian curve to data with [maximum likelihood estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). For that, we subclassed `tf.keras.layers.Layer` and wrapped up the model's parameters in our custom layer. Then, we used negative log-likelihood minimization to have Tensorflow figure out the optimal values for the distribution's parameters. In today's short post, we will again fit a Gaussian curve to normally distributed data with MLE. However, we will use Tensorflow's trainable probability distributions rather than using a custom layer.

{% highlight python %}
{% raw %}
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

print("TF version:", tf.__version__)
print("TFP version:", tfp.__version__)

#    TF version: 2.4.0
#    TFP version: 0.12.0

import matplotlib.pyplot as plt
import numpy as np
{% endraw %}
{% endhighlight %}

The same as before, we generate some Gaussian data with $$\mu = 2, \sigma = 1$$:

{% highlight python %}
{% raw %}
def generate_gaussian_data(m, s, n=1000):
    x = tf.random.uniform(shape=(n,))
    y = tf.random.normal(shape=(len(x),), mean=m, stddev=s)
    return x, y

# Generate normally distributed data with mean = 2 and std dev = 1
x_train, y_train = generate_gaussian_data(m=2, s=1)

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=20, rwidth=0.8)
plt.xlabel('y')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.plot(x_train, y_train, 'b.')
plt.xlabel('x')
plt.ylabel('y');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/trainable_distributions/gaussian.png" alt="Normally distributed data">
</p>

We now use a `tensorflow_probability.Normal` distribution, with trainable parameters for loc and scale. We do assign some random values to them, which will be updated during the training loop.

{% highlight python %}
{% raw %}
# Instantiate a normal distribution with trainable parameters for loc and scale.
# The initial values we assign will be updated during the training process.
normal_dist = tfd.Normal(loc=tf.Variable(0., name='loc'),
                         scale=tf.Variable(2., name='scale'))

plt.hist(normal_dist.sample(1000), bins=20, rwidth=0.8, label='initial dist')
plt.hist(y_train, bins=20, rwidth=0.8, label='target dist')
plt.legend(loc="upper left")
plt.xlabel('y')
plt.ylabel('Count');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/trainable_distributions/gaussian_initial_vs_target.png" alt="Normally distributed data">
</p>

{% highlight python %}
{% raw %}
def nll(dist, x_train):
    """Calculates the negative log-likelihood for a given distribution
    and a data set."""
    return -tf.reduce_mean(dist.log_prob(x_train))

@tf.function
def get_loss_and_grads(dist, x_train):
    with tf.GradientTape() as tape:
        tape.watch(dist.trainable_variables)
        loss = nll(dist, x_train)
        grads = tape.gradient(loss, dist.trainable_variables)
    return loss, grads

# Instantiate a stochastic gradient descent optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

epochs = 300
nll_loss = []
for _ in range(epochs):
    loss, grads = get_loss_and_grads(normal_dist, y_train)
    optimizer.apply_gradients(zip(grads, normal_dist.trainable_variables))
    nll_loss.append(loss)
 
plt.plot(nll_loss)
plt.xlabel('Epochs')
plt.ylabel('Cost function\n(Negaltive Log-Likelihood)');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/trainable_distributions/nll_vs_epoch.png" alt="Negative log-likelihood vs epoch">
</p>

{% highlight python %}
{% raw %}
plt.hist(normal_dist.sample(1000), bins=20, rwidth=0.8, alpha=0.5, label='predicted dist')
plt.hist(y_train, bins=20, rwidth=0.8, alpha=0.5, label='target dist')
plt.legend(loc="upper left")
plt.xlabel('y')
plt.ylabel('Count');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/trainable_distributions/predicted_vs_target.png" alt="Negative log-likelihood vs epoch">
</p>

We print the final estimates for the distribution's parameters, and we see that they are pretty close to the ones we used when we generated our training data.

{% highlight python %}
{% raw %}
normal_dist.loc, normal_dist.scale
#    (<tf.Variable 'loc:0' shape=() dtype=float32, numpy=1.9555925>,
#     <tf.Variable 'scale:0' shape=() dtype=float32, numpy=1.0020323>)
{% endraw %}
{% endhighlight %}


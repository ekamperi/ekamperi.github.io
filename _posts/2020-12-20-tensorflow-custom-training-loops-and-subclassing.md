---
layout: post
title:  "Custom training loops and subclassing with Tensorflow"
date:   2020-11-20
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'optimization', 'statistics', 'Tensorflow']
description: How to create custom training loops and use subclassing with Tensorflow
---

The most straightforward way to train a model is to use the `model.fit()` and `model.fit_generator()` Keras functions. These functions also accept callbacks that allow for early stopping, save the model to the disk periodically, log for TensorBoard after every batch, accumulate statistics, and so on. However, it may be the case that one needs even finer control of the training loop. A central component of the training loop is automatic differentiation. In this post, we will see a couple of examples on how to construct a custom training loop, define a custom loss function, have Tensorflow compute the gradients of the loss function with respect to the trainable parameters, and then update the latter.

## Fit linear regression model to data by minimizing MSE

In the first example, we will generate some noisy data and then fit a linear regression model of the form $$y = m x + b$$. The model's parameters are $$m, b$$, and we will have Tensorflow figure out their optimal values.

{% highlight python %}
{% raw %}
import tensorflow as tf
tf.__version__

# '2.4.0'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def generate_noisy_data(m, b, n=100):
    """ Generates (x, y) points along the line y = m * x + b
    and adds some gaussian noise in the y coordinates.
    """
    x = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(len(x),), stddev=0.15)
    y = m * x + b + noise
    return x, y

x_train, y_train = generate_noisy_data(m=1, b=2)
plt.plot(x_train, y_train, 'b.');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/custom_training_loops/output_4_0.png">
</p>

We then proceed by subclassing the `tf.keras.layers.Layer` class to create a new layer. The new layer accepts as input a one dimensional tensor of $$x$$'s and outputs a tensor of $$y$$'s, after mapping the input to $$m x + b$$. This layer's trainable parameters are $$m, b$$, which are initialized to random values drawn from the normal distribution and to zeros, respectively. 

{% highlight python %}
{% raw %}
class LinearRegressionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LinearRegressionLayer, self).__init__()
        self.m = self.add_weight(shape=(1,), initializer='random_normal')
        self.b = self.add_weight(shape=(1,), initializer='zeros')

    def call(self, inputs):
        return self.m * inputs + self.b

linear_regression_layer = LinearRegressionLayer()
linear_regression_layer(x_train)

#    <tf.Tensor: shape=(100,), dtype=float32, numpy=
#    array([6.32419065e-03, 1.72153376e-02, 6.32639334e-04, 5.57286246e-03,
#           1.25469696e-02, 1.44133652e-02, 9.44772546e-05, 1.12606948e-02,
#           5.26433578e-03, 1.16873141e-02, 1.17568867e-02, 1.61101166e-02,
#           4.66661761e-03, 1.65831670e-02, 1.31081808e-02, 5.73157147e-03,
#           ...
#           5.68887964e-03, 1.30544147e-02, 1.70514826e-02, 9.80825396e-04],
#          dtype=float32)>
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
def MSE(y_pred, y_true):
    """Calculates the Mean Squared Error between y_pred and y_true vectors"""
    return tf.reduce_mean(tf.square(y_pred - y_true))
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
# Calculate the MSE of the initial m, b values
MSE(linear_regression_layer(x_train), y_train)

    <tf.Tensor: shape=(), dtype=float32, numpy=6.283869>
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
# Custom training loop
learning_rate = 0.05
epochs = 30

mse_loss = []
for i in range(epochs):
    with tf.GradientTape() as tape:
        predictions = linear_regression_layer(x_train)
        current_mse_loss = MSE(predictions, y_train)

    gradients = tape.gradient(current_mse_loss, linear_regression_layer.trainable_variables)
    linear_regression_layer.m.assign_sub(learning_rate * gradients[0])
    linear_regression_layer.b.assign_sub(learning_rate * gradients[1])
    mse_loss.append(current_mse_loss)
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
# Print optimal values for the parameters m, b.
# The ground truth values are m = 1, b = 2.
linear_regression_layer.m, linear_regression_layer.b

    (<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.053719], dtype=float32)>,
     <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.911512], dtype=float32)>)
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
# Generate evenly spaced numbers over the initial x interval plus some margin
x = np.linspace(min(min(x_train), -0.15), max(max(x_train), 1.15), 50)

# Plot the optimal y = m * x + b regression line superimposed with the data
plt.plot(x, linear_regression_layer.m * x + linear_regression_layer.b, 'r')
plt.plot(x_train, y_train, 'b.');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/custom_training_loops/output_10_0.png">
</p>


## Fit Gaussian curve to data with maximum likelihood estimation


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
plt.hist(y_train, bins=20, rwidth=0.9)
plt.xlabel('y')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.plot(x_train, y_train, 'b.')
plt.xlabel('x')
plt.ylabel('y');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/custom_training_loops/output_12_0.png">
</p>


{% highlight python %}
{% raw %}
class GaussianFitLayer(tf.keras.layers.Layer):
    def __init__(self, ivs):
        super(GaussianFitLayer, self).__init__()
        self.m = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(ivs[0]))
        self.s = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(ivs[1]))

    def call(self, inputs):
        y = tf.random.normal(shape=(len(inputs),), mean=self.m, stddev=self.s)
        return y

# Come up with some initial values (they are off on purpose),
# so that we can check whether the optimization algorithm converges.
m0 = 0.7 * tf.reduce_mean(y_train)
s0 = 1.3 * tf.math.reduce_std(y_train)
gaussian_fit_layer = GaussianFitLayer([m0, s0])
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
def NLL(y_true, params):
    """Calculates the Negative Log-Likelihood for a given set of parameters"""
    N = len(y_true)
    m, s = params
    return (N/2.) * tf.math.log(2. * np.pi * s**2) + (1./(2.*s**2)) * tf.math.reduce_sum((y_true - m)**2)
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
# Custom training loop
learning_rate = 0.0005
epochs = 50

nll_loss = []
for i in range(epochs):
    with tf.GradientTape() as tape:
        current_nll_loss = NLL(y_train, [gaussian_fit_layer.m, gaussian_fit_layer.s])
    gradients = tape.gradient(current_nll_loss, gaussian_fit_layer.trainable_variables)
    gaussian_fit_layer.m.assign_sub(learning_rate * gradients[0])
    gaussian_fit_layer.s.assign_sub(learning_rate * gradients[1])
    nll_loss.append(current_nll_loss)
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
plt.plot(nll_loss)
plt.xlabel('Epochs')
plt.ylabel('Cost function\n(Negaltive Log-Likelihood)');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/custom_training_loops/output_16_0.png">
</p>



{% highlight python %}
{% raw %}
# Print optimal values for the parameters m, s.
# The ground truth values are m = 2, s = 1.
gaussian_fit_layer.m, gaussian_fit_layer.s

    (<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([2.0243182], dtype=float32)>,
     <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.0158775], dtype=float32)>)
{% endraw %}
{% endhighlight %}


{% highlight python %}
{% raw %}
fig = plt.figure(figsize=(15, 4))
fig.suptitle('Training data vs. generated data after fitting')

plt.subplot(1, 2, 1)
x = np.linspace(min(x_train), max(x_train), 1000)
y = tf.random.normal(shape=(len(x_train),), mean=gaussian_fit_layer.m, stddev=gaussian_fit_layer.s)
plt.hist(y, rwidth=0.9, alpha=0.5)
plt.hist(y_train, rwidth=0.9, alpha=0.5)
plt.xlabel('y')
plt.ylabel('Count');

plt.subplot(1, 2, 2)
plt.plot(x_train, y_train, 'r.')
plt.plot(x, y, 'b.');
plt.xlabel('x')
plt.ylabel('y');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/custom_training_loops/output_18_0.png">
</p>


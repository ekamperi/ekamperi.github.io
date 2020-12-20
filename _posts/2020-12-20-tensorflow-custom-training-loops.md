---
layout: post
title:  "Custom training loops with tensorflow"
date:   2020-11-20
categories: [mathematics]
tags: ['Tensorflow', 'machine learning', 'mathematics', 'optimization', 'statistics']
description: How to create custom training loops with Tensorflow
---


{% highlight python %}
{% raw %}
import tensorflow as tf
tf.__version__
{% endraw %}
{% endhighlight %}




    '2.4.0'




```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## Fit linear regression model to data by minimizing MSE


```python
def generate_noisy_data(m, b, n=100):
    """ Generate (x, y) points along the line y = m * x + b
    and add some gaussian noise in the y coordinates.
    """
    x = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(len(x),), stddev=0.15)
    y = m * x + b + noise
    return x, y

x_train, y_train = generate_noisy_data(m=1, b=2)
plt.plot(x_train, y_train, 'b.');
```

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/custom_training_loops/output_4_0.png">
</p>


```python
class LinearRegressionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LinearRegressionLayer, self).__init__()
        self.m = self.add_weight(shape=(1,), initializer='random_normal')
        self.b = self.add_weight(shape=(1,), initializer='zeros')

    def call(self, inputs):
        return self.m * inputs + self.b

linear_regression_layer = LinearRegressionLayer()
linear_regression_layer(x_train)
```




    <tf.Tensor: shape=(100,), dtype=float32, numpy=
    array([6.32419065e-03, 1.72153376e-02, 6.32639334e-04, 5.57286246e-03,
           1.25469696e-02, 1.44133652e-02, 9.44772546e-05, 1.12606948e-02,
           5.26433578e-03, 1.16873141e-02, 1.17568867e-02, 1.61101166e-02,
           4.66661761e-03, 1.65831670e-02, 1.31081808e-02, 5.73157147e-03,
           1.20661929e-02, 3.33428243e-03, 1.69532336e-02, 7.82631990e-03,
           3.23826540e-03, 1.32078435e-02, 2.05542613e-03, 1.45999836e-02,
           1.39767965e-02, 7.82642607e-03, 1.83836627e-03, 1.62130874e-02,
           1.02055387e-03, 1.54200243e-02, 8.62369128e-03, 1.76197048e-02,
           3.65552935e-03, 1.64051559e-02, 1.18581038e-02, 7.72755174e-03,
           4.66154050e-03, 1.33532621e-02, 9.35327914e-03, 1.71481911e-02,
           3.58462660e-03, 1.39624244e-02, 2.42970046e-03, 1.79871935e-02,
           1.38472775e-02, 1.52761415e-02, 6.94182469e-03, 4.68987226e-03,
           1.39027147e-03, 3.87467328e-03, 1.39327552e-02, 1.59199238e-02,
           5.15068090e-03, 1.59113687e-02, 4.92062652e-03, 2.69877305e-03,
           2.57297466e-03, 1.73339471e-02, 7.80327013e-03, 1.36877559e-02,
           1.25807161e-02, 1.58167221e-02, 1.35237388e-02, 1.06976945e-02,
           9.10426676e-03, 4.90641640e-03, 7.51545001e-03, 1.61607354e-03,
           7.12127285e-03, 5.54429926e-03, 1.75650325e-02, 1.52561103e-03,
           1.50490431e-02, 1.62839666e-02, 6.58374326e-03, 1.14207575e-02,
           5.19145466e-03, 1.08724795e-02, 4.76105168e-04, 7.95099535e-04,
           1.11031523e-02, 1.24944123e-02, 2.71888776e-03, 5.24104200e-03,
           9.35510173e-03, 3.21108871e-03, 5.74930850e-03, 1.60326157e-02,
           1.43502010e-02, 7.36440346e-03, 1.13349603e-02, 8.14282615e-03,
           1.12893572e-02, 8.59997422e-03, 4.59470646e-03, 1.06322216e-02,
           5.68887964e-03, 1.30544147e-02, 1.70514826e-02, 9.80825396e-04],
          dtype=float32)>




```python
def MSE(y_pred, y_true):
    """Calculate Mean Squared Error between y_pred and y_true vectors"""
    return tf.reduce_mean(tf.square(y_pred - y_true))
```


```python
# Calculate the MSE of the initial m, b values
MSE(linear_regression_layer(x_train), y_train)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=6.283869>




```python
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
```


```python
# Print optimal values for the parameters m, b.
# The ground truth values are m = 1, b = 2.
linear_regression_layer.m, linear_regression_layer.b
```




    (<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.053719], dtype=float32)>,
     <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.911512], dtype=float32)>)




```python
# Generate evenly spaced numbers over the initial x interval plus some margin
x = np.linspace(min(min(x_train), -0.15), max(max(x_train), 1.15), 50)

# Plot the optimal y = m * x + b regression line superimposed with the data
plt.plot(x, linear_regression_layer.m * x + linear_regression_layer.b, 'r')
plt.plot(x_train, y_train, 'b.');
```

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/custom_training_loops/output_10_0.png">
</p>


 ## Fit Gaussian curve to data with maximum likelihood estimation


```python
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
```

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/custom_training_loops/output_12_0.png">
</p>


```python
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
```


```python
def LL(y_true, params):
    N = len(y_true)
    m, s = params
    return (N/2.) * tf.math.log(2. * np.pi * s**2) + (1./(2.*s**2)) * tf.math.reduce_sum((y_true - m)**2)
```


```python
# Custom training loop
learning_rate = 0.0005
epochs = 50

nll_loss = []
for i in range(epochs):
    with tf.GradientTape() as tape:
        current_nll_loss = LL(y_train, [gaussian_fit_layer.m, gaussian_fit_layer.s])
    gradients = tape.gradient(current_nll_loss, gaussian_fit_layer.trainable_variables)
    gaussian_fit_layer.m.assign_sub(learning_rate * gradients[0])
    gaussian_fit_layer.s.assign_sub(learning_rate * gradients[1])
    nll_loss.append(current_nll_loss)
```


```python
plt.plot(nll_loss)
plt.xlabel('Epochs')
plt.ylabel('Cost function\n(Negaltive Log-Likelihood)');
```

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/custom_training_loops/output_16_0.png">
</p>



```python
# Print optimal values for the parameters m, s.
# The ground truth values are m = 2, s = 1.
gaussian_fit_layer.m, gaussian_fit_layer.s
```




    (<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([2.0243182], dtype=float32)>,
     <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.0158775], dtype=float32)>)




```python
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
```

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/custom_training_loops/output_18_0.png">
</p>


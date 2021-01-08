---
layout: post
title: "Probabilistic regression with Tensorflow"
date:   2021-01-07
categories: [machine learning]
tags: [algorithms, 'machine learning', Python, Tensorflow]
description: Implementation of probabilistic regression with Tensorflow
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
You probably have heard the saying, *"If all you have is a hammer, everything looks like a nail"*. This proverb applies to many cases, deterministic classification neural networks not being an exception. Consider, for instance, a typical neural network that classifies images from the [CIFAR-10 dataset](https://en.wikipedia.org/wiki/CIFAR-10). This dataset consists of 60.000 color images, all of which belong to 10 classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Naturally, no matter what image we feed this network, say a pencil, it will always assign it to one of the 10 known classes.

However, it would be handy if the model conveyed its uncertainty for the predictions it made. So, given a "pencil" image, it would probably label it as a bird or ship or whatever. At the same time, we'd like it to assign a large uncertainty to this prediction. To reach such an inference level, we need to rethink the traditional deterministic neural network paradigm and take a leap of faith towards probabilistic modeling. **So, instead of having a model parameterized by its point weights, each weight will now be sampled from a posterior distribution whose parameters will be trained during the training process.**

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/probabilistic_regression/probabilistic_vs_deterministic_nn.png" alt="Probabilistic vs. deterministic neural networks">
</p>
<p align="center">
Image taken from Blundell, et al. Weight Uncertainty in Neural Networks. arXiv (2015)
</p>

## Aleatoric and epistemic uncertainty
Probabilistic modeling is intimately related to the concept of uncertainty. The latter is sometimes divided into two categories, aleatoric (also known as statistical) and epistemic (also known as systematic). **Aleatoric** is derived from the Latin word "alea" which means die. You might be familiar with the phrase ["alea iact est"](https://en.wikipedia.org/wiki/Alea_iacta_est), meaning "the die has been cast". Hence, aleatoric uncertainty relates to the data itself and captures the inherent randomness when running the same experiment or performing the same task. For instance, if a person draws the number "4" repeatedly, its shape will be slightly different every time. Aleatoric uncertainty is irreducible in the sense that no matter how much data we collect, there will always be some noise in them.

**Epistemic uncertainty**, on the other hand, refers to a model's uncertainty. I.e., there is uncertainty regarding which model's parameters accurately model the experimental data, which is decreased as we collect increasingly more training examples. The modeling of epistemic uncertainty is realized by enabling a neural network's weights to be probabilistic rather than deterministic.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/probabilistic_regression/aleatoric_vs_epistemic.png" alt="Aleatoric vs. epistemic uncertainty">
</p>
<p align="center">
Kendall, A. & Gal, Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? arXiv [cs.CV] (2017)
</p>

## Tensorflow example
### Summary objective
In the following example, we will generate some non-linear noisy training data, and then we will develop a probabilistic regression neural network to fit the data. To do so, we will provide appropriate prior and posterior trainable probability distributions. This blog post is inspired by a weekly assignment of the course "Probabilistic Deep Learning with TensorFlow 2" from Imperial College London.

{% highlight python %}
{% raw %}
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers 

import numpy as np
import matplotlib.pyplot as plt
{% endraw %}
{% endhighlight %}

### Data generation
We generate some training data $$\mathcal{D}=\{(x_i, y_i)\}$$ using the equation $$y_i = x_i^5 + 0.4 \, x_i \,\epsilon_i$$ where $$\epsilon_i \sim \mathcal{N}(0,1)$$ means that the noise $$\epsilon_i$$ is sampled from a normal distribution with *zero* mean and standard deviation equal to *one*.

{% highlight python %}
{% raw %}
# Generate some non-linear training data
n_points = 500
x_train = np.linspace(-1, 1, n_points)[:, np.newaxis]
y_train = np.power(x_train, 5) + 0.4 * x_train * np.random.randn(n_points)[:, np.newaxis]
plt.scatter(x_train, y_train, alpha=0.2);
plt.xlabel('x')
plt.ylabel('y')
plt.show();
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 65%; height: 65%" src="{{ site.url }}/images/probabilistic_regression/training_data.png" alt="Non-linear probabilistic regression data">
</p>

### Setup prior and posterior distributions
#### Bayes' rule
At the core of probabilistic predictive modeling lies the [Bayes' rule](https://en.wikipedia.org/wiki/Bayes%27_theorem). To estimate a full posterior distribution of the parameters $$\mathbf{w}$$, given some training data $$\mathcal{D} = \{(x_i, y_y)\}$$, the Bayes rule assumes the following form:

$$
p(\mathbf{\mathbf{w}|\mathcal{D}}) = \frac{p(\mathcal{D}|\mathbf{w})p(\mathbf{w})}{p(\mathcal{D})}
$$

In the following image, you see a sketch of the various probability distributions that the Bayes' rule entangles. In plain terms, it holds that $$\text{Prior beliefs} \oplus \text{Evidence} = \text{Posterior beliefs}$$, i.e., we start with some assumptions inscribed in the prior distribution, then we observe the "Evidence", and we update our prior beliefs accordingly, to yield the posterior distribution. Subsequently, the posterior distribution acts as the next iteration's prior distribution, and the whole cycle is repeated.

<p align="center">
 <img style="width: 65%; height: 65%" src="{{ site.url }}/images/probabilistic_regression/prior_posterior_evidence.png" alt="Prior, posterior and evidence distributions in Bayes rule">
</p>

To let all these sink, let us elaborate on the essence of posterior distribution, by marginalizing the model's parameters. The probability of predicting $$y$$ given an input $$\mathbf{x}$$ and the training data $$\mathcal{D}$$ is:

$$
p(y\mid \mathbf{x},\mathcal{D})= \int p(y\mid \mathbf{x},\mathbf{w}) \, p(\mathbf{w}\mid\mathcal{D}) \mathop{\mathrm{d}\theta}
$$

This is equivalent to having an ensemble of models with different parameters $$\mathbf{w}$$, and taking their average weighted by the posterior probabilities of their parameters, $$p(\mathbf{w}\mid \mathcal{D})$$. Neat?

There are two problems with this approach, however. First, it is computationally intractable to calculate an exact solution for the posterior distribution. Second, the averaging implies that our equation is not differentiable, so we can't use good old backpropagation to update the model's parameters! The answer to these hindrances is **variational inference**, a method of formulating inference as an optimization problem! We won't dive deep into the theoretical background, but the inquiring reader may google for the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). I promise to blog about all the juicy mathematical details of the KL divergence concept in a future post.

#### Prior distribution
We start by defining a prior distribution for our model's weights. I haven't researched the matter a lot, but in the absence of any evidence, adopting a normal distribution as a prior is a fair way to initialize a probabilistic neural network. After all, the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) asserts that a properly normalized sum of samples will approximate a normal distribution no matter the actual underlying distribution. We use the `DistributionLambda()` function to inject a distribution into our model, which you can think of as the "lambda function" analog for distributions. The distribution we use is a multivariate normal with a diagonal covariance matrix:

$$
\Sigma = \left(
\begin{matrix}
\sigma_1^2 & 0 & 0 & \ldots \\
0 & \sigma_2^2 & 0 & \ldots \\
0 & 0 & \sigma_3^2 & \ldots\\
\vdots & \vdots & \vdots & \ddots
\end{matrix}
\right)
$$

The mean values are initialized to zero and the $$\sigma_i^2$$ to one.

{% highlight python %}
{% raw %}
def get_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential([
        tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
        loc=tf.zeros(n), scale_diag=tf.ones(n)))
    ])
    return prior_model
{% endraw %}
{% endhighlight %}

#### Posterior distribution
The case of the posterior distribution is a bit more complex. We again use a multivariate Gaussian distribution, and there are three ways to parameterize it. First, in terms of a positive definite covariance matrix $$\mathbf{\Sigma}$$, second via a positive definite precision matrix $$\mathbf{\Sigma}^{-1}$$, and last with a lower-triangular matrix $$\mathbf{L}\mathbf{L}^⊤$$ with positive-valued diagonal entries, such that $$\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^⊤$$. This triangular matrix can be obtained via, e.g., [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) of the covariance matrix. In our case, we are going for the last method with `MultivariateNormalTriL()`. "TriL" stands for "triangular lower". So, instead of parameterizing the neural network with point weights $$\mathbf{w}$$, we will instead parameterize it with $$\mathbf{\mu}$$'s and $$\sigma$$'s. Notice that for a lower triangular matrix there are $$(n^2 - n)/2 + n = n(n+1)/2$$ non-zero elements.

{% highlight python %}
{% raw %}
def get_posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])
    return posterior_model
{% endraw %}
{% endhighlight %}

By the way, let us create some prior and posterior distributions, print the number of their trainable variables, and sample from them. Note that every time we run this cell block, we get different results for the samples.

{% highlight python %}
{% raw %}
# The prior distribution has no trainable variables
prior_model = get_prior(3, 1)
print('Trainable variables for prior model: ', prior_model.layers[0].trainable_variables)
print('Sampling from the prior distribution:\n', prior_model.call(tf.constant(1.0)).sample(5))

# The posterior distribution for kernel_size = 3, bias_size = 1, is expected to
# have (3 + 1) + ((4^2 - 4)/2 + 4) = 14 parameters. Note that the default initializer
# according to the docs is 'zeros'.
posterior_model = get_posterior(3, 1)
print('\nTrainable variables for posterior model: ', posterior_model.layers[0].trainable_variables)
print('Sampling from the posterior distribution:\n', posterior_model.call(tf.constant(1.0)).sample(5))

#    Trainable variables for prior model:  []
#    Sampling from the prior distribution:
#     tf.Tensor(
#    [[ 1.3140054   0.93301576 -2.3522265   0.5879774 ]
#     [-2.6143072   0.39889303  0.72736305 -0.06531376]
#     [-1.1271048   0.4480154  -1.389969    0.87443566]
#     [-0.6140247   0.3008949   0.91000426  0.1832995 ]
#     [ 0.39756483  0.4414646  -1.025012    0.21117625]], shape=(5, 4), dtype=float32)
#    
#    Trainable variables for posterior model:  [<tf.Variable 'constant:0' shape=(14,) dtype=float32, numpy=
#    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          dtype=float32)>]
#    Sampling from the posterior distribution:
#     tf.Tensor(
#    [[-0.0099524   1.107596   -0.34787297  0.1307174 ]
#     [-0.7565929  -0.08078367  0.1275031   0.80345786]
#     [ 0.75810474  0.12409975  0.11558666  0.54518634]
#     [-0.5074226   0.11740679  0.86849195 -0.33246624]
#     [ 0.01261052  0.44296038  0.61944205  0.4496125 ]], shape=(5, 4), dtype=float32)
{% endraw %}
{% endhighlight %}

### Define the model, loss function, and optimizer
To define probabilistic layers in a neural network, we use the `DenseVariational()` function, specifying the input and output shape, along with the prior and posterior distributions that we have previously defined. We use a sigmoid activation function to enable the network model non-linear data, along with an `IndependentNormal()` output layer, with an event shape equal to 1 (since our $$y$$ is just a scalar). Regarding the `kl_weight` parameter, you may refer to the original paper "Weight Uncertainty in Neural Networks" for further information. For now, just take for granted that it is a scaling factor.

{% highlight python %}
{% raw %}
# Define the model, negative-log likelihood as the loss function
# and compile the model with the RMSprop optimizer
model = tf.keras.Sequential([
    tfpl.DenseVariational(input_shape=(1,), units=8,
                          make_prior_fn=get_prior,
                          make_posterior_fn=get_posterior,
                          kl_weight=1/x_train.shape[0],
                          activation='sigmoid'),
    tfpl.DenseVariational(units=tfpl.IndependentNormal.params_size(1),
                          make_prior_fn=get_prior,
                          make_posterior_fn=get_posterior,
                          kl_weight=1/x_train.shape[0]),
    tfpl.IndependentNormal(1)
])

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

model.compile(loss=nll, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005))
model.summary()

#    Model: "sequential_2"
#    _________________________________________________________________
#    Layer (type)                 Output Shape              Param #   
#    =================================================================
#    dense_variational (DenseVari (None, 8)                 152       
#    _________________________________________________________________
#    dense_variational_1 (DenseVa (None, 2)                 189       
#    _________________________________________________________________
#    independent_normal (Independ multiple                  0         
#    =================================================================
#    Total params: 341
#    Trainable params: 341
#    Non-trainable params: 0
#    _________________________________________________________________
{% endraw %}
{% endhighlight %}

Let's calculate by hand the model's parameters. The **first dense variational layer** has 1 input, 8 outputs and 8 biases. Therefore, there are $$1\cdot 8 + 8 = 16$$ weights. Since each weight is going to be modelled by a normal distribution, we need 16 $$\mu$$'s, and $$(16^2 - 16)/2 + 16 = 136$$ $$\sigma$$'s. The latter is the number of elements of a lower triangular matrix $$8\times 8$$. Therefore, in total we need $$16 + 132 = 152$$ parameters.

What about the **second variational layer**? This one has 8 inputs (since the previous had 8 outputs), 2 outputs (the $$\mu, \sigma$$ of the independent normal distribution), and 2 biases. Therefore, it has $$8\times 2 + 2 = 18$$ weights. For 18 weights, we need 18 $$\mu$$'s and $$(18^2 - 18)/2 + 18 = 171$$ $$\sigma$$'s. Therefore, in total we need $$18 + 171 = 189$$ parameters. The `tfpl.MultivariateNormalTriL.params_size(n)` static function calculates the number of parameters needed to parameterize a multivariate normal distribution, so we don't have to bother with it.

### Train the model and make predictions
We train the model for 1000 epochs and plot the loss function *vs.* to confirm that the algorithm has converged.

{% highlight python %}
{% raw %}
# Train the model for 1000 epochs
history = model.fit(x_train, y_train, epochs=1000, verbose=0)
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss');
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 65%; height: 65%" src="{{ site.url }}/images/probabilistic_regression/loss_vs_epoch.png" alt="Loss vs. epochs">
</p>

Indeed RMSprop converged, and now we proceed by making some predictions:

{% highlight python %}
{% raw %}
plt.scatter(x_train, y_train, marker='.', alpha=0.2, label='data')
for _ in range(5):
    y_model = model(x_train)
    y_hat = y_model.mean()
    y_hat_minus_2sd = y_hat - 2 * y_model.stddev()
    y_hat_plus_2sd = y_hat + 2 * y_model.stddev()
    plt.plot(x_train, y_hat, color='red', label='model $\mu$' if _ == 0 else '')
    plt.plot(x_train, y_hat_minus_2sd, color='blue', label='$\mu - 2SD$' if _ == 0 else '')
    plt.plot(x_train, y_hat_plus_2sd, color='green', label='$\mu + 2SD$' if _ == 0 else '')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 65%; height: 65%" src="{{ site.url }}/images/probabilistic_regression/regression1.png" alt="Non-linear probabilistic regression data">
</p>

The following plot was generaged by taking the average of 100 models:

<p align="center">
 <img style="width: 65%; height: 65%" src="{{ site.url }}/images/probabilistic_regression/regression2.png" alt="Non-linear probabilistic regression data">
</p>

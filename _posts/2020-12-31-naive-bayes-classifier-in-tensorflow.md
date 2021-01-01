---
layout: post
title:  "How to implement a Naive Bayes classifier with Tensorflow"
date:   2020-12-01
categories: [machine learning]
tags: [algorithms, 'machine learning', Python, Tensorflow]
description: Implementation of a Naive Bayes classifier with Tensorflow's trainable distributions for the iris dataset
---

** WORK IN PROGRESS **

## Introduction
A [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is a simple probabilistic classifier based on the [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) along with some strong (naive) assumptions regarding the independence of features. Others have suggested the name "independent feature model" as more fit. For example, a pet may be considered a dog if it has 4 legs, a tail, and barks. These features (presence of 4 legs, a tail, and barking) may depend on each other. However, the naive Bayes classifier assumes that these properties contribute independently to the probability that a pet is a dog. Naive Bayes classifier is used heavily in text classification, e.g., assigning topics on text, detecting spam, identifying age/gender from text, performing sentiment analysis. Given that there are many well-written introductory articles on this topic, we won't spend much time in theory. 

## The mathematical formulation

Given a set of features $$x_1, x_2, \ldots, x_n$$ and a set of classes $$C$$, we want to build a model that yields the value of $$P(C_k\mid x_1, x_2, \ldots, x_n)$$. Then, by taking the maximum probability over this range of probabilities, we come up with our best estimate for the correct class:

$$
\begin{align*}
C_\text{predicted} &= \underset{c_k \in \mathcal{C}}{\text{arg max}} \,P(C_k | x_1, x_2, \ldots, x_n)\\
&= \underset{c_k \in \mathcal{C}}{\text{arg max}} \,\frac{P(x_1, x_2, \ldots, x_n|C_k) P(C_k)}{P(x_1, x_2, \ldots, x_n)}\\
&= \underset{c_k \in \mathcal{C}}{\text{arg max}} \, P(x_1, x_2, \ldots, x_n|C_k) P(C_k)
\end{align*}
$$

In the second line we applied the [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) $$P(A\mid B) = P(B\mid A) P(A) / P(B)$$. In the last line, we omitted the denominator since it is the same across all classes, i.e., acts merely as a scaling factor. The estimation of $$P(C_k)$$ is straightforward; we just compute each class's relative frequency in the training set. However, the calculation of $$P(x_1, x_2, \ldots, x_n\mid C_k)$$ is more demanding. Here comes the "naive" part of the Naive Bayes classifier. We make the assumption that $$x_1, x_2, \ldots, x_n$$ features are independent. Then, it holds that $$P(x_1, x_2, \ldots, x_n\mid C_k) = P(x_1\mid C_k)P(x_2\mid C_k)\ldots P(x_n\mid C_k)$$ or just $$\prod_{i=1}^n P(x_i\mid C_k)$$. This greatly reduces the number of the model's parameters and simplifies their estimation. So, to sum up, the naive Bayes classifier is the solution to the following optimization problem:

$$
C_\text{predicted} = \underset{c_k \in \mathcal{C}}{\text{arg max}} \, P(C_k) \prod_{i=1}^n P(x_i|C_k)
$$

In the pet example, assuming we had two classes, $$C_\text{dog}$$ and $$C_\text{monkey}$$, we would write:

$$
\begin{align*}
P\left(\text{dog} \mid \text{4-legs}, \text{tail}, \text{barks}\right) &= P(\text{4-legs}) P(\text{tail}) P(\text{barks}) \,\, P(C_\text{dog})\\
P\left(\text{monkey} \mid \text{4-legs}, \text{tail}, \text{barks}\right) &= \underbrace{P(\text{4-legs}) P(\text{tail}) P(\text{barks})}_{\text{feature distributions}} \,\, \underbrace{P(C_\text{monkey})}_{\text{prior}}
\end{align*}
$$

Finally, we would compare the two calculated probabilities to infer whether the pet was a dog or a monkey.

All the model parameters (the priors for each class and the feature probability distributions) need to be approximated from the training set. The priors can be calculated by the relative frequency of each class in the training set, e.g. $$P(C_k) = \frac{\text{# of samples in class }C_k}{\text{total # of samples}}$$. The feature probability distributions can be approximated with [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). In this post, we will create some trainable Gaussian distributions for the features and have Tensorflow estimate their parameters ($$\mu, \sigma$$) by minimizing the negative log-likelihood, which is equivalent to maximizing of log-likelihood. We already did this in [a previous minimal post](https://ekamperi.github.io/mathematics/2020/12/26/tensorflow-trainable-probability-distributions.html). However, the feature distributions need not be Gaussian. For instance, in Mathematica's current implementation, the feature distributions are modeled using a piecewise-constant function:

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/naive_bayes/naive_bayes_piecewise.png" alt="Naive Bayes classifier with piecewise-constant feature distributions">
</p>

## Pros and cons of naive Bayes classifier
Advantages of naive Bayes classifier:
* Works quite well in real-world applications.
* Requires only a small amount of training data.
* With each training example, prior and likelihood can be updated in real-time.
* Since it assumes independent variables, only the class variables' variances need to be estimated and not the entire covariance matrix (i.e., fewer parameters to calculate).
* Fast training and fast inference.
* It gives a probability distribution over all classes (i.e., not just a classification).
* Multiple classifiers may be combined, e.g., by taking the product of their predicted probabilities.
* May be used as a first-line "punching bag" before other smarter algorithms kick in the problem.

Disadvantages
* More sophisticated models outperform them.

## Tensorflow example with the iris dataset


{% highlight python %}
{% raw %}
def learn_parameters(x, y, mus, scales, optimiser, epochs):
    """
    Set up the class conditional distributions as a MultivariateNormalDiag
    object, and update the trainable variables in a custom training loop.
    """
    @tf.function
    def nll(dist, x_train, y_train):
        log_probs = dist.log_prob(x_train)
        L = len(tf.unique(y_train)[0])
        y_train = tf.one_hot(indices=y_train, depth=L)
        return -tf.reduce_mean(log_probs * y_train)

    @tf.function
    def get_loss_and_grads(dist, x_train, y_train):
        with tf.GradientTape() as tape:
            tape.watch(dist.trainable_variables)
            loss = nll(dist, x_train, y_train)
            grads = tape.gradient(loss, dist.trainable_variables)
        return loss, grads

    nll_loss = []
    mu_values = []
    scales_values = []
    x = tf.cast(np.expand_dims(x, axis=1), tf.float32)
    dist = tfd.MultivariateNormalDiag(loc=mus, scale_diag=scales)
    for epoch in range(epochs):
        loss, grads = get_loss_and_grads(dist, x, y)
        optimiser.apply_gradients(zip(grads, dist.trainable_variables))
        nll_loss.append(loss)
        mu_values.append(mus.numpy())
        scales_values.append(scales.numpy())
    nll_loss, mu_values, scales_values = np.array(nll_loss), np.array(mu_values), np.array(scales_values)
    return (nll_loss, mu_values, scales_values, dist)

{% endraw %}
{% endhighlight %}

---
layout: post
title:  "How to implement a Naive Bayes classifier with Tensorflow"
date:   2020-12-01
categories: [machine learning]
tags: [algorithms, 'machine learning', Python, Tensorflow]
description: Implementation of a Naive Bayes classifier with Tensorflow's trainable distributions for the iris dataset
---

** WORK IN PROGRESS **

A Naive Bayes classifier is a simple probabilistic classifier based on the Bayes theorem along with some strong (naive) assumptions regarding the independence of features. Others have suggested the name "independent feature model" as more fit. For example, a pet may be considered a dog if it has 4 legs, a tail, and barks. These features (presence of 4 legs, a tail, and barking) may depend on each other. However, the naive Bayes classifier assumes that these properties contribute independently to the probability that a pet is a dog. Given that there are many well-written introductory articles on this topic, we won't spend much time in theory.

$$
\begin{align*}
C_\text{predicted} &= \underset{c_k \in \mathcal{C}}{\text{arg max}} \,P(C_k | x_1, x_2, \ldots, x_n)\\
&= \underset{c_k \in \mathcal{C}}{\text{arg max}} \,\frac{P(x_1, x_2, \ldots, x_n|C_k) P(C_k)}{P(x_1, x_2, \ldots, x_n)}\\
&= \underset{c_k \in \mathcal{C}}{\text{arg max}} \, P(x_1, x_2, \ldots, x_n|C_k) P(C_k)
\end{align*}
$$

The estimation of $$P(C_k)$$ is straightforward; we just compute the relative frequency of each class in the training set. However, the calculation of $$P(x_1, x_2, \ldots, x_n\|C_k)$$ is more demanding. Here comes the "naive" part of the Naive Bayes classifier. We make the assumption that $$x_1, x_2, \ldots, x_n$$ features are independent. Then, it holds that $$P(x_1, x_2, \ldots, x_n\mid C_k) = P(x_1\|C_k)P(x_2\|C_k)\ldots P(x_n\|C_k) = \prod_{i=1}^n P(x_i\|C_k)$$. This greatly reduces the number of model's parameters and simplifies their estimation. So, to sum up the Naive Bayes classifier is the solution to:

$$
C_\text{predicted} = \underset{c_k \in \mathcal{C}}{\text{arg max}} \, P(C_k) \prod_{i=1}^n P(x_i|C_k)
$$

Advantages of naive Bayes classifier:
* Works quite well in real-world applications
* Requires only a small amount of training data
* With each training example, prior and likelihood can be updated in real-time
* Since it assumes independent variables, only the variances of the class variables need to be estimated and not the entire covariance matrix (i.e., there are fewer parameters to estimate)
* Fast training and fast inference
* It gives a probability distribution over all classes (i.e., not just a classification)
* Multiple classifiers may be combined, e.g., by taking the product of their predicted probabilities
* May be used as a first-line "punching bag" before other smarter algorithms kick in the problem

Disadvantages
* More sophisticated models outperform them


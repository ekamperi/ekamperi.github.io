---
layout: post
title:  "Gradient descent"
date:   2019-07-28
categories: [machine learning]
---

Gradient descent is an optimization algorithm for minimizing the value of a function. In the context of machine learning, we typically define some cost (or loss) function $$J(\mathbf{\theta})$$, where $$\mathbf{\theta} = (\theta_0, \theta_1, \ldots)$$ are the model's parameters that we want to tune (e.g. the weights in a neural network). The update rule for these parameters is:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\mathbf{\theta})
$$

Where $$\alpha$$ is the learning rate (how fast we update our model parameters).

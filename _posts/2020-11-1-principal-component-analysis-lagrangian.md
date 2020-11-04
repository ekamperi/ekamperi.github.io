---
layout: post
title:  "Principal component analysis with lagrangian multiplier"
date:   2020-11-01 20:47:56 +0000
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'optimization', 'statistics']
---

## The motivation
We often come up with datasets consisting of many variables because our ability to measure things has improved significantly over the years. However, we are interested in distilling meaning, knowledge, and insights out of the data. In this context, many variables might be correlated with each other, allowing us to exploit their connectedness and reduce their numbers needed to describe the data. In the following plots, we see a set of data points that span the whole 3D space uniformly in the first row. In this case, there are no correlations between the $$x,y,z$$ variables to exploit. However, in the second row, we see a set of points that are lying, more or less, on a plane. In this case, we could use 2 variables and still locate the points without losing much accuracy. This is the so-called dimensionality reduction. Working with fewer dimensions has many advantages, e.g., being able to visualize the data in 2 or 3 dimensions.

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/pca_motivation.png" alt="Principal component analysis">
</p>

Suppose that we have $$\mathbf{x}_1,\mathbf{x}_2,…,\mathbf{x}_n$$ centered points in $$m$$ dimensional space. Let $$q$$ denote the unit vector along which we project our $$\mathbf{x}$$'s. The length of the projection $$y_i$$ of $$\mathbf{x}_i$$ is $$y_i = x_i^⊤ mathbf{q}$$. The mean squared projection is the variance $$V$$ summed over all points $$\mathbf{x}_i$$:

$$
\begin{align*}
Var &= \frac{1}{n} \sum_{i=1}^n y_i^2 = \frac{1}{n}\sum_{i=1}^n\left(\mathbf{x}_i^⊤ \mathbf{q}\right)^2\\ 
&=\frac{1}{n}\sum_{i=1} \mathbf{x}_i^⊤ \mathbf{q} \cdot\mathbf{x}_i^⊤ \mathbf{q} = 
\frac{1}{n}\sum_{i=1} \mathbf{q}^⊤ \mathbf{x}_i \cdot\mathbf{x}_i^⊤ \mathbf{q}\\
&= \mathbf{q}^⊤ \underbrace{\left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^⊤\right)}_{\text{Covariance matrix}}\mathbf{q} = \mathbf{q}^⊤ C \mathbf{q}
\end{align*}
$$

Maximize Variance $$Var$$ subject to the constraint $$\|\mathbf{q}\|=1$$:

$$
\mathcal{L}(\mathbf{x},\lambda) = f(\mathbf{x})+\lambda(g(\mathbf{x})-c)
$$


Maximize Variance $Var$ subject to the constraint $\|\mathbf{q}\|=1$:

$$
\mathcal{L}(\mathbf{q},\lambda) = \mathbf{q}^⊤ C \mathbf{q} +\lambda(\mathbf{q}^⊤ \mathbf{q}-1)
$$


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{q}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda}
$$

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_many_vecs.png" alt="Principal component analysis">
</p>

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_single_vec.png" alt="Principal component analysis">
</p>


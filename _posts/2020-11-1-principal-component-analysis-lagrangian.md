---
layout: post
title:  "Principal component analysis with lagrangian multiplier"
date:   2020-11-01 20:47:56 +0000
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'optimization', 'statistics']
---

## The motivation
We often come up with datasets consisting of many variables because our ability to measure things has improved significantly over the years. The ultmate goal though it to distill meaning, knowledge, and insights out of the data. In this context, many of the variables might be correlated with each other, allowing us to exploit their relations and reduce their number needed to describe the data. In the following plots, we see a set of data points that span the whole 3D space uniformly in the first row. In this case, there are no correlations between the $$x,y,z$$ variables to exploit. Hence, we could not possibly compress our description of the dataset and get away using only two variables. However, in the second row, we see a set of points that are lying, more or less, on a plane. In this case, we could use just two variables and still locate the points in the 3D space without losing much accuracy. The intrinsic dimension of our data is the plane, not the 3D space. This transformation of data from a high-dimensional space into a low-dimensional one, which preserves some essential qualities of the original data, is the so-called **dimensionality reduction**. Working with fewer dimensions has many advantages, such as being able to create visualizations in 2 or 3 dimensions, that we are comfortably with.

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/pca_motivation.png" alt="Principal component analysis">
</p>

## The formulation
Suppose that we have $$\mathbf{x}_1,\mathbf{x}_2,…,\mathbf{x}_n$$ centered points in $$m$$ dimensional space. Let $$\mathbf{q}$$ denote the unit vector along which we project our $$\mathbf{x}$$'s. The length of the projection $$y_i$$ of $$\mathbf{x}_i$$ is $$y_i = x_i^⊤ \mathbf{q}$$. The mean squared projection is the variance $$V$$ summed over all points $$\mathbf{x}_i$$:

$$
\begin{align*}
Var &= \frac{1}{n} \sum_{i=1}^n y_i^2 = \frac{1}{n}\sum_{i=1}^n\left(\mathbf{x}_i^⊤ \mathbf{q}\right)^2\\ 
&=\frac{1}{n}\sum_{i=1} \mathbf{x}_i^⊤ \mathbf{q} \cdot\mathbf{x}_i^⊤ \mathbf{q} = 
\frac{1}{n}\sum_{i=1} \mathbf{q}^⊤ \mathbf{x}_i \cdot\mathbf{x}_i^⊤ \mathbf{q}\\
&= \mathbf{q}^⊤ \underbrace{\left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^⊤\right)}_{\text{Covariance matrix}}\mathbf{q} = \mathbf{q}^⊤ C \mathbf{q}
\end{align*}
$$

Our objective is to maximize Variance $$Var$$ subject to the constraint $$\|\mathbf{q}\|=1$$. Such problems of constrained optimization might be reformulated as problems of unconstrained optimization via the use of Lagrangian multipliers.

$$
\mathcal{L}(\mathbf{x},\lambda) = f(\mathbf{x})+\lambda(g(\mathbf{x})-c)
$$

Maximize Variance $Var$ subject to the constraint $$\|\mathbf{q}\|=1$$:

$$
\mathcal{L}(\mathbf{q},\lambda) = \mathbf{q}^⊤ C \mathbf{q} +\lambda(\mathbf{q}^⊤ \mathbf{q}-1)
$$


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{q}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda}
$$

We drew various candidate vectors $$\mathbf{q}$$ pointing into different directions in the following image. Which is the one vector $$\mathbf{q}$$ that when our 2D data points are projected on the line it defines, their variance is maximized? I.e., they are as much spread out as possible?

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_many_vecs.png" alt="Principal component analysis">
</p>

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_single_vec.png" alt="Principal component analysis">
</p>


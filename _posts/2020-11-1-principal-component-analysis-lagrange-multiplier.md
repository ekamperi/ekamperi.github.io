---
layout: post
title:  "Principal component analysis with Lagrange multiplier"
date:   2020-11-01 20:47:56 +0000
categories: [mathematics]
tags: ['Lagrange multiplier', 'machine learning', 'mathematics', 'optimization', 'statistics']
---

## The motivation
We often come up with datasets consisting of many variables because our ability to measure things has improved significantly over the years. The ultimate goal, though, is to distill meaning, knowledge, and insights out of the data. In this context, we seek for variables that are linearly correlated with each other, to exploit their relations and reduce their number needed to describe the data. We do so by constructing new variables, called *principal components*, that are linear combinations of the original ones.

In the following plots, we see a set of data points that span the whole 3D space uniformly in the first row. In this case, there are no correlations between the $$x,y,z$$ variables to exploit. Hence, we could not possibly compress our description of the dataset and get away using only two variables. However, in the second row, we see a set of points that are lying, more or less, on a plane. In this case, we could use just two variables and still locate the points in the 3D space without losing much accuracy. The intrinsic dimension of our data is the plane, not the 3D space.

This transformation of data from a high-dimensional space into a low-dimensional one, which preserves some essential qualities of the original data, is the so-called [**dimensionality reduction**](https://en.wikipedia.org/wiki/Dimensionality_reduction). Working with fewer dimensions has many advantages, such as being able to create visualizations in 2 or 3 dimensions that we are comfortable to work with.

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/pca_motivation.png" alt="Principal component analysis">
</p>

## The formulation
Suppose that we have $$\mathbf{x}_1,\mathbf{x}_2,…,\mathbf{x}_n$$ centered points in $$m$$ dimensional space. Let $$\mathbf{v}$$ denote the unit vector along which we project our $$\mathbf{x}$$'s. The length of the projection $$y_i$$ of $$\mathbf{x}_i$$ is $$y_i = \mathbf{x}_i^⊤ \mathbf{v}$$.

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_variance_intro.png" alt="Principal component analysis">
</p>

The mean squared projection is the variance $$V$$ summed over all points $$\mathbf{x}_i$$:

$$
\begin{align*}
Var &= \frac{1}{n} \sum_{i=1}^n y_i^2 = \frac{1}{n}\sum_{i=1}^n\left(\mathbf{x}_i^⊤ \mathbf{v}\right)^2\\ 
&=\frac{1}{n}\sum_{i=1} \mathbf{x}_i^⊤ \mathbf{v} \cdot\mathbf{x}_i^⊤ \mathbf{v} = 
\frac{1}{n}\sum_{i=1} \mathbf{v}^⊤ \mathbf{x}_i \cdot\mathbf{x}_i^⊤ \mathbf{v}\\
&= \mathbf{v}^⊤ \underbrace{\left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^⊤\right)}_{\text{Covariance matrix}}\mathbf{v} = \mathbf{v}^⊤ C \mathbf{v}
\end{align*}
$$

Mind that the covariance matrix is defined as:

$$
C = \frac{1}{n} \sum_{i=1}^n \left(\mathbf{x}_i - m\right)\left(\mathbf{x}_i -m\right)^⊤
$$

Where $$m$$ is the mean of $$\mathbf{x}_i, i = 1,2,\ldots,n$$. Which is why we asked for the variables $$\mathbf{x}_i$$ to be centered. So that $$m=0$$ and the formula for covariance simplifies to $$C = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T$$.

Our objective is to maximize Variance $$Var$$ subject to the constraint $$\|\mathbf{v}\|=1$$. Such problems of constrained optimization might be reformulated as unconstrained optimization problems via the use of [**Lagrangian multipliers**](https://en.wikipedia.org/wiki/Lagrange_multiplier). If we'd like to maximize $$f(\mathbf{x})$$ subject to $$g(\mathbf{x})=c$$, we introduce the Lagrange multiplier $$\lambda$$ and construct the Lagrangian $$\mathcal{L}(\mathbf{x},\lambda)$$:

$$
\mathcal{L}(\mathbf{x},\lambda) = f(\mathbf{x}) - \lambda(g(\mathbf{x})-c)
$$

The sign of $$\lambda$$ doesn't make any difference. We then solve the system of equations:

$$
\left\{\frac{\partial\mathcal{L} (\mathbf{x}, \lambda)}{\partial \mathbf{x}} = 0, \, \frac{\partial \mathcal{L}(\mathbf{x}, \lambda)}{\partial \lambda} = 0\right\}
$$

In our case, we want to maximize Variance $$V$$ subject to the constraint $$\|\mathbf{v}\|=1 \Leftrightarrow \mathbf{v}^⊤\mathbf{v} = 1$$. Our Lagrangian is:

$$
\mathcal{L}(\mathbf{v},\lambda) = \mathbf{v}^⊤ C \mathbf{v} - \lambda(\mathbf{v}^⊤ \mathbf{v}-1)
$$

We solve for the stationary points:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{v}} &= 2\mathbf{v}^⊤ \mathbf{C} \mathbf{v} - 2\lambda v^⊤ = 0\\
\frac{\partial \mathcal{L}}{\partial \lambda} &= \mathbf{v}^⊤ \mathbf{v} - 1 = 0
\end{align*}
$$

And we end up with the following eigenvector equation:

$$
C \mathbf{v} = \lambda \mathbf{v}
$$

## Example with one principal component
Let us play with the simplest possible scenario, where we have two variables, $$x_1$$ and $$x_2$$, and we'd like to calculate a single principal component. In the graph below, we plot the data along with various candidate vectors $$\mathbf{v}$$ pointing into different directions. Our goal is to find the one vector $$\mathbf{v}$$, which will maximize the variance of the data points when projected on the line the vector defines.

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_many_vecs.png" alt="Principal component analysis">
</p>


If we plot the variance as a function of angle of $$v$$ with the $$x$$ axis, we get the following:

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_variance_vs_angle.png" alt="Principal component analysis">
</p>

In the following image, we see the values of variance $$V$$ as a function of the angle of vector $$v$$ and the $$x$$ axis.

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_single_vec.png" alt="Principal component analysis">
</p>


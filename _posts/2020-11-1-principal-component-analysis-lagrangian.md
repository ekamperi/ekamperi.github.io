---
layout: post
title:  "Principal component analysis with lagrangian multiplier"
date:   2020-11-01 20:47:56 +0000
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'optimization', 'statistics']
---

$$
y_i = \mathbf{x}_i^T \mathbf{q}
$$

$$
\begin{align*}
Var &= \frac{1}{n} \sum_{i=1}^n y_i^2 = \frac{1}{n}\sum_{i=1}^n\left(\mathbf{x}_i^⊤ \mathbf{q}\right)^2\\ 
&=\frac{1}{n}\sum_{i=1} \mathbf{x}_i^⊤ \mathbf{q} \cdot\mathbf{x}_i^⊤ \mathbf{q} = 
\frac{1}{n}\sum_{i=1} \mathbf{q}^⊤ \mathbf{x}_i \cdot\mathbf{x}_i^⊤ \mathbf{q}\\
&= \mathbf{q}^⊤ \underbrace{\left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^⊤\right)}_{\text{Covariance matrix}}\mathbf{q} = \mathbf{q}^⊤ C \mathbf{q}
\end{align*}
$$

Maximize Variance $Var$ subject to the constraint $\|\mathbf{q}\|=1$:

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
<img style="width: 100%; height: 60%" src="{{ site.url }}/images/pca_many_vecs.png" alt="Principal component analysis">
</p>

<p align="center">
<img style="width: 100%; height: 60%" src="{{ site.url }}/images/pca_single_vec.png" alt="Principal component analysis">
</p>


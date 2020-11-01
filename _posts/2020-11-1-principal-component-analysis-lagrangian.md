---
layout: post
title:  "Principal component analysis with lagrangian multiplier"
date:   2020-11-01 20:47:56 +0000
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'optimization', 'statistics']
---

$$
y_i = \mathbf{x}_i^T \mathbf{v}
$$

$$
\begin{align*}
Var &= \frac{1}{n} \sum_{i=1}^n y_i^2 = \frac{1}{n}\sum_{i=1}^n\left(\mathbf{x}_i^T \mathbf{v}\right)^2\\ 
&=\frac{1}{n}\sum_{i=1} \mathbf{x}_i^T \mathbf{v} \cdot\mathbf{x}_i^T \mathbf{v} = 
\frac{1}{n}\sum_{i=1} \mathbf{v}^T \mathbf{x}_i \cdot\mathbf{x}_i^T \mathbf{v}\\
&= \mathbf{v}^T \underbrace{\left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T\right)}_{\text{Covariance matrix}}\mathbf{v} = \mathbf{v}^T C \mathbf{v}
\end{align*}
$$

Maximize Variance $Var$ subject to the constraint $\|\mathbf{v}\|=1$:

$$
\mathcal{L}(\mathbf{x},\lambda) = f(\mathbf{x})+\lambda(g(\mathbf{x})-c)
$$


Maximize Variance $Var$ subject to the constraint $\|\mathbf{v}\|=1$:

$$
\mathcal{L}(\mathbf{v},\lambda) = \mathbf{v}^T C \mathbf{v} +\lambda(\mathbf{v}^T \mathbf{v}-1)
$$


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{v}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda}
$$

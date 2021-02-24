---
layout: post
title:  "Principal Component Analysis limitations"
date:   2021-02-23
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'statistics']
description: A list of common pitfalls / limitations of Principal Component Analysis
---

We have already discussed principle component analysis [in a previous post](https://ekamperi.github.io/mathematics/2020/11/01/principal-component-analysis-lagrange-multiplier.html), where we viewed PCA as a constrained optimization problem solved with Lagrange multipliers. PCA is arguably a widely used data dimensionality reduction technique. In the era of big data, where our measuring capabilities have exponentially increased, it is often the case that we end up with very high dimensional datasets that we want to "summarize" with low dimensionality projections. However, there are some limitations of PCA that someone should be familiar with.

0. In order to apply PCA and produce meaningful results, we must first **check whether some assumptions hold**, like the presence of linear correlations or sampling adequacy. This step is often overlooked, and strictly speaking is not a limitation of PCA, but of the person running the analysis. 
1. Related to the previous point, the method relies on **linear relationships** between the variables in a dataset. So if our data are not linearly correlated, then PCA will not perform well. However, there is the so-called called kernel PCA version that allows PCA to also work with nonlinear data. Vanilla PCA computes the covariance matrix of the dataset:
$$
C = \frac{1}{n}\sum_{i=1}^n{\mathbf{x}_i\mathbf{x}_i^\mathsf{T}}
$$. Kernel PCA, on the other hand, first transforms the data into an even higher-dimensional space where:
$$
C = \frac{1}{n}\sum_{i=1}^n{\Phi(\mathbf{x}_i)\Phi(\mathbf{x}_i)^\mathsf{T}}
$$. And only then projects the data onto the first $$k$$ eigenvectors of that matrix, just like PCA. The kernel trick refers to performing the computation without actually computing $$\Phi(\mathbf{x})$$. This is possible only if $$\Phi$$ is chosen such that it has a known corresponding kernel. KPCA doesn't always cut it, so depending on your dataset you may need to look at other non-linear dimensionality reduction techniques, such as LLE, isomap, or t-SNE.
2. Another limitation is the **assumption of orthogonality**, so the principal components are *by design* orthogonal to each other. Depending on the data, there may exist far "better" basis vectors to summarize the data *that are not orthogonal*. The following image shows an extreme such case taken [from here](https://arxiv.org/pdf/1404.1100.pdf):
    <p align="center">
    <img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_pitfall2.png" alt="PCA pitfall">
    </p>
3. The next gotcha is that large variance is used as a criterion for the existence of structure in the data. However, **sometimes structure is found in places with low variance**, as we see in the following image. If we kept only the first principal component, out of 2, in the right case we would be absolutely fine, but in the left case we would perform badly, in a classification context. 
    <p align="center">
    <img style="width: 100%; height: 100%" src="{{ site.url }}/images/pca_pitfall.png" alt="PCA pitfall">
    </p>
4. Next, PCA is **scale variant**. This means that if the variables in our dataset have different units, some variables will dominate the others simply because they assume bigger values, and therefore contribute more to the overall variance. That's why we typically normalize our data so that they have a unit standard deviation.
5. One very important issue is that of **interpretability**. Once we have replaced our original variables with the principal components, it's not always entirely trivial to interpret the results. Sometimes, depending on data's structure and the research question, one might apply a rotation *after PCA* to simplify the interpretation of the components. Such rotations include *Varimax* and *oblique* rotations (however, non-orthogonal rotations have their own set of limitations). 
6. Last, PCA has a hard time working with **missing data and outliers**. With respect to outliers, there is [Robust PCA](https://en.m.wikipedia.org/wiki/Robust_principal_component_analysis). 

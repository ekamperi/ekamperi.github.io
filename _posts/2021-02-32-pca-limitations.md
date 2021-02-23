---
layout: post
title:  "Principal Component Analysis limitations"
date:   2021-02-23
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'statistics']
description: A list of common pitfalls / limitations of Principal Component Analysis
---

We have already discussed principle component analysis in a previous post. PCA is a widely used data dimensionality reduction technique. However, there are some limitations of PCA that someone should be familiar with.

1. First, the method relies on linear relationships between the variables in the dataset. So if our data are not linear, then PCA will not perform well. However, there is the so-called called kernel PCA trick that allows PCA to also work with nonlinear data.
2. The second limitation is the assumption of orthogonality, so the principal components are orthogonal to each other.
3. The next limitation is that large variance is used as a criterion for the existence of structure in the data. However, sometimes structure is found in places with low variance, as we see in the following image.
<p align="center">
<img style="width: 60%; height: 60%" src="{{ site.url }}/images/pca_pitfall.png" alt="PCA pitfall">
</p>
4. Next, PCA is scale variant. This means that if the variables in our dataset have different units, some variables will dominate the others simply because they assume bigger values. That's why we typically normalize our data so that they have a unit standard deviation.
5. One very important issue is that of interpretability. Once we have replaced our original variables with our principal components, it's not always entirely trivial to interpret the principal components.
6. Last, PCA has a hard time working with missing data.

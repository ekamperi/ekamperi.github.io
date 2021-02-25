---
layout: post
title:  "Principal Component Analysis limitations"
date:   2021-02-23
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'statistics']
description: A list of common pitfalls/limitations of Principal Component Analysis
---

In the era of big data, our measuring capabilities have exponentially increased. It is often the case that we end up with very high dimensional datasets that we want to "summarize" with low dimensionality projections. PCA is arguably a widely used data dimensionality reduction technique. We have already discussed principle component analysis [in a previous post](https://ekamperi.github.io/mathematics/2020/11/01/principal-component-analysis-lagrange-multiplier.html), where we viewed PCA as a constrained optimization problem solved with Lagrange multipliers.  However, there are some limitations of PCA that someone should be familiar with.

0. To apply PCA and produce meaningful results, we must first **check whether some assumptions hold**, like the presence of linear correlations (e.g., Bartlett's test of sphericity) or sampling adequacy (e.g., Kaiser-Meyer-Olkin test). This step is often overlooked, and strictly speaking, is not a limitation of PCA but of the person running the analysis. 
1. **Principal components must be unique** otherwise lack meaning (they are essentially random axes). We already [have demonstrated this](https://ekamperi.github.io/mathematics/2020/11/01/principal-component-analysis-lagrange-multiplier.html#weak-linear-correlation). Therefore, the distinctiveness of the eigenvalues is a fundamental assumption of PCA that the analyst must test. The following image was taken [from here](https://onlinelibrary.wiley.com/doi/pdf/10.1111/evo.13835):
    <p align="center">
        <img style="width: 100%; height: 100%" src="{{ site.url }}/images/pca_pitfall3.png" alt="PCA pitfall">
    </p>
2. Τhe method relies on **linear relationships** between the variables in a dataset. So if our data are not linearly correlated, then PCA will not perform well. However, there is the so-called called kernel PCA version that allows PCA to also work with non-linear data. Vanilla PCA computes the covariance matrix of the dataset:
$$
C = \frac{1}{n}\sum_{i=1}^n{\mathbf{x}_i\mathbf{x}_i^\mathsf{T}}
$$. Kernel PCA, on the other hand, first transforms the data into an even higher-dimensional space where:
$$
C = \frac{1}{n}\sum_{i=1}^n{\Phi(\mathbf{x}_i)\Phi(\mathbf{x}_i)^\mathsf{T}}
$$. And only then projects the data onto the eigenvectors of that matrix, just like regular PCA. The kernel trick refers to performing the computation without actually computing $$\Phi(\mathbf{x})$$. This is possible only if $$\Phi$$ is chosen such that it has a known corresponding kernel. KPCA doesn't always cut it, so depending on your dataset, you may need to look at other non-linear dimensionality reduction techniques, such as LLE, isomap, or t-SNE.
3. Another limitation is the **assumption of orthogonality**, since the principal components are *by design* orthogonal to each other. Depending on the data, far "better" basis vectors may exist to summarize the data *that are not orthogonal*. The following image shows an extreme such case that was taken [from here](https://arxiv.org/pdf/1404.1100.pdf):
    <p align="center">
    <img style="width: 50%; height: 50%" src="{{ site.url }}/images/pca_pitfall2.png" alt="PCA pitfall">
    </p>
4. The next gotcha is that large variance is used as a criterion for the existence of structure in the data. However, **sometimes structure is found in places with low variance**, as we see in the following image. If we kept only the first principal component, we would be absolutely fine in the right case, but in the left case, we would perform badly in a classification context. 
    <p align="center">
    <img style="width: 100%; height: 100%" src="{{ site.url }}/images/pca_pitfall.png" alt="PCA pitfall">
    </p>
5. Next, PCA is **scale variant**. This means that if the variables in our dataset have different units, some variables will dominate the others simply because they assume bigger values, and therefore contribute more to the overall variance. That's why we typically transform our data so that they have a unit standard deviation. (Also, the data absolutely need to be mean-centered, but I think pretty much everyone is aware of this.)
6. One very important issue is that of **interpretability**. Once we have replaced our original variables with the principal components, it's not always entirely trivial to interpret the results. Sometimes, depending on the data's structure and the research question, one might apply a rotation *after PCA* to simplify the components' interpretation. Such rotations include *Varimax* and *oblique* rotations (however, these have their own set of limitations, e.g., they might produce components that don't correspond to maximal variance or components that are correlated). 
7. Last, PCA has a hard time working with **missing data and outliers**. Here is a review paper on how to [impute missing data in the context of PCA](http://pbil.univ-lyon1.fr/members/dray/files/articles/dray2015a.pdf). With respect to handling outliers and corrupted data, there is [Robust PCA](https://en.m.wikipedia.org/wiki/Robust_principal_component_analysis). 

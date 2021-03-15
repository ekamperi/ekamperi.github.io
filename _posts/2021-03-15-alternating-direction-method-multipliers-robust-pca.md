---
layout: post
title:  "Alternating direction method of multipliers and Robust PCA"
date:   2021-03-15
categories: [mathematics]
tags: ['machine learning', 'Mathematica', 'mathematics', 'statistics']
description: An introduction on the Alternating Direction of Method Multipliers and how it can be applied to Robust PCA
---

### Contents
{:.no_toc}

In the [previous blog post](https://ekamperi.github.io/mathematics/2021/02/23/pca-limitations.html),
we discussed some of the limitations of principle component analysis.
One such restriction arises when there exist outliers, corruption in the data, and gross errors.
One method that handles such cases is the so-called robust PCA, which we will talk about today. 

Suppose that we are given a large matrix $$X$$, such that it can be decomposed as a sum of a
low-rank matrix $$\mathbf{L}$$ and a sparse matrix $$\mathbf{S}$$, i.e., $$\mathbf{X} = \mathbf{L} + \mathbf{S}$$.

<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/robst_pca/rpca_example0.png" alt="Robust PCA example">
</p>

In this setup, we do not know the rank of matrix $$\mathbf{L}$$, not even the positions of the zeros in the sparse
matrix $$\mathbf{S}$$ or how many of them there are. The optimization
problem we are called to solve is:

$$
\mathop{\mathrm{arg\,min}}_{L,S} \,\,\mathop{\mathrm{rank}}(\mathbf{L}) + \lambda \left\Vert \mathbf{S}\right\Vert_{\infty}, \,\, s.t. \mathbf{X} = \mathbf{L} + \mathbf{S}
$$

However, in this formulation, both terms are non-convex, and therefore we restate the problem as follows:
$$
\mathop{\mathrm{arg\,min}}_{L,S} \,\, \left\Vert \mathbf{L}\right\Vert_* + \lambda \left\Vert \mathbf{S}\right\Vert_1, \,\, s.t. \mathbf{X} = \mathbf{L} + \mathbf{S}
$$

Where $$\left\Vert \mathbf{L}\right\Vert_*$$ is the nuclear norm of matrix $$\mathbf{L}$$, i.e.,
the sum of else singular values which are used as a proxy for their rank of the matrix:
$$
\left\Vert \mathbf{L}\right\Vert_* \stackrel{\text{def}}{=} \sum_i \sigma_i(\mathbf{L}), \,\,\,\,\,\,
\left\Vert \mathbf{S}\right\Vert_1 \stackrel{\text{def}}{=} \sum_{ij} |S_{ij}|
$$

### When is $$\mathbf{S}$$, i.e., $$\mathbf{X} = \mathbf{L} + \mathbf{S}$$ separation meaningful? 
1. $$\mathbf{L}$$ is not sparse, e.g., its singular values are reasonably spread out.
2. $$\mathbf{S}$$ is not low rank, e.g., it does not have all non zero elements in a column or in a few columns.

Otherwise, the decomposition is simply not feasible. A remarkable fact is that there is no tuning of the algorithm. 
Not even the scalar $$\lambda$$ needs to be tuned most of the time. There is a universal value that works well, $$\lambda = \frac{1}{\sqrt{\mathrm{max}(n_1,n_2)}}$$.
However, if assumptions are only partially valid, the optimal value of Lambda may vary a bit. For example, if the matrix $$\mathbf{S}$$ is very sparse,
we may need to increase $$\lambda$$ to recover matrices $$\mathbf{L}$$ of larger rank.

### Applications 
1.	**Video surveillance**. The background variations of a video are modeled as low rank, and the foreground objects such as pedestrians and cars are modeled as sparse errors which are superimposed on the low-rank background. 
2.	**Latent semantic indexing**. The basic idea here is to generate a document versus term matrix whose entries reflect a term's relevance in a document. Then, the composing X equal L + S, L would capture the common words and S the few terms that would distinguish the documents. 
3.	**Ranking and recommendation systems**.
4.	**Face recognition**. Removing shadows, specularities, and reflections from facial images. 

### Augmented Lagrangian method 

$$
\mathcal{L}(\mathbf{L},\mathbf{S},\mathbf{Y}) = 
\left\Vert\mathbf{L}\right\Vert _*+ \lambda \left\Vert \mathbf{S}\right\Vert_1 + \underbrace{\left<{\mathbf{Y},\mathbf{X}-\mathbf{L}-\mathbf{S}}\right>}_{\text{Lagrange Multipliers}} + \underbrace{\frac{\rho}{2} \left\Vert \mathbf{X} - \mathbf{L} -\mathbf{S}\right\Vert_2^2}_{\text{Augmented Lagrangian}}
$$

Where $$\left<{\mathbf{A},\mathbf{B}}\right>=\text{trace}(\mathbf{A}^*\mathbf{B})$$

A generic ALM algorithm would solve by repeatedly doing the following calculations:

$$
(\mathbf{L}_{k+1}, \mathbf{S}_{k+1}) = \mathop{\mathrm{arg\,min}}_{L,S} \,\,\mathcal{L}(\mathbf{L},\mathbf{S},\mathbf{Y}_k)
$$

And then update the Lagrange multipliers:

$$
\mathbf{Y}_{k+1}=\underbrace{\mathbf{Y}_k + \rho \underbrace{(\mathbf{X}-\mathbf{L}_{k+1} - \mathbf{S}_{k+1})}_{\text{residual error}}}_{\text{running sum of residual errors}}
$$

However, the first step is usually as expensive as solving the initial problem. So we need to do better than this.
In the literature, many methods solved the problem above. One such method is the so-called Alternating Direction Method of Multipliers.
ADM splits the minimization problem into two smaller and easier to tackle subproblems, where $$\mathbf{L}, \mathbf{S}$$ are minimized separately. 

$$
\begin{align*}
\mathbf{L}_{k+1} &= \mathop{\mathrm{arg\,min}}_{L} \,\,\mathcal{L}(\mathbf{L}, \mathbf{S}_k, \mathbf{Y}_k)\\
\mathbf{S}_{k+1} &= \mathop{\mathrm{arg\,min}}_{S} \,\, \mathcal{L}(\mathbf{L}_{k+1}, \mathbf{S}, \mathbf{Y}_k)\\
\mathbf{Y}_{k+1} &= \mathbf{Y}_k + \rho(\mathbf{X} - \mathbf{L}_{k+1} - \mathbf{S}_{k+1}), \,\,\rho>0
\end{align*}
$$

### Example code in Mathematica

{% highlight mathematica %}
{% raw %}
Shrink[t_, x_] := Sign[x]*Map[Max[#, 0] &, Abs[x] - t, {2}]

RobustPCA[X_] :=
 Module[{m, n, \[Rho], \[Lambda], i, U, \[CapitalSigma], V, L, S, Y, 
   error, tolerance},
  L = ConstantArray[0, Dimensions[X]];
  S = ConstantArray[0, Dimensions[X]];
  Y = ConstantArray[0, Dimensions[X]];
  {m, n} = Dimensions[X];
  \[Rho] = m*n/(4 Norm[X, 1]);
  \[Lambda] = 1./Sqrt[Max[Dimensions[X]]];
  tolerance = 10^-17*Norm[X, "Frobenius"];
  Print[tolerance];
  error = Infinity;
  errors =
   Reap[
    For[i = 1, i <= 10^4 && error > tolerance, i++,
     {U, \[CapitalSigma], V} = 
      SingularValueDecomposition[X - S + (1/\[Rho])*Y];
     L = U . Shrink[1/\[Rho], \[CapitalSigma]] . Transpose[V];
     S = Shrink[\[Lambda] /\[Rho], X - L + (1/\[Rho])*Y];
     Y = Y + \[Rho]*(X - L - S);
     error = Norm[X - L - S, "Frobenius"];
     Sow[error];
     ]];
  {{L, S}, errors}
]

{{L, S}, errors} = RobustPCA[X];
{% endraw %}
{% endhighlight %}
  



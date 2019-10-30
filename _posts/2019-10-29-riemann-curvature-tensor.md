---
layout: post
title:  "How to derive the Riemann curvature tensor"
date:   2019-10-29
categories: [mathematics]
tags: ['general relativity', 'mathematics', 'tensors']
---

So, I've decided to bite the bullet and study *general relativity*. I've been postponing it for quite a while, but the idea of my life ending without me having studied one of the most profound and fundamental theories of physics was motivating to say the least. I will be posting random stuff as I go and maybe I'll come back later to edit them, as my understanding of the theory -hopefully- deepens.

I decided to watch the video lectures from Professor Susskind that are publicly available on YouTube. I liked them because Susskind puts an emphasis on the physical aspect of things and less on the formalism of the mathematics. Of course both are required, but for starters I think that it's best to first build the intuition.

Our goal is to come up with a tool to measure the curvature of space. Generally speaking, a change in the direction of a vector *parallel-transported around a closed loop* is a way to measure precisely this. Consider the following vector that is parallel-transported across $$A \rightarrow N \rightarrow B \rightarrow A $$.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/Riemann_curvature.png" alt="Parallel transport of a vector on the surface of a sphere">
</p>
Image taken from [here](https://en.wikipedia.org/wiki/Parallel_transport).

The idea is to compute the commutator of two *covariant derivatives*. Let us consider the action of the operator on a random vector $$V^\rho$$:

$$
\begin{align*}
[\nabla_\mu, \nabla_\nu ] V^\rho
&= \nabla_\mu \nabla_\nu V^\rho - \nabla_\nu \nabla_\mu V^\rho \\
&= \nabla_\mu \left[ \partial_\nu V^\rho + \Gamma_{\nu\sigma}^\rho V^\sigma \right] - (\mu \leftrightarrow \nu)\\
&= \partial_\mu \left[ \partial_\nu V^\rho + \Gamma_{\nu\sigma}^\rho V^\sigma \right]
- \Gamma_{\mu \nu}^\lambda \left[ \partial_\lambda V^\rho + \Gamma_{\lambda \sigma}^\rho V^ \sigma\right]
+\Gamma_{\mu\lambda}^\rho \left[ \partial_\nu V^\lambda + \Gamma_{\nu\sigma}^\lambda V^\sigma\right] - (\mu \leftrightarrow \nu)\\
&= \partial_\mu \partial_\nu V^\rho + \underbrace{\partial_\mu (\Gamma_{\nu\sigma}^\rho) V^\sigma + \Gamma_{\nu\sigma}^\rho \partial_\mu V^\sigma}_{\partial_\mu(\Gamma_{\nu\sigma}^\rho V^\sigma)}
-\Gamma_{\mu\nu}^\lambda \partial_\lambda V^\rho - \Gamma_{\mu\nu}^\lambda \Gamma_{\lambda \sigma}^\rho V^\sigma
+ \Gamma_{\mu\lambda}^\rho \partial_\nu V^\lambda + \Gamma_{\mu\lambda}^\rho \Gamma_{\nu\sigma}^\lambda V^\sigma\\

&-\partial_\nu\partial_\mu V^\rho - \underbrace{\partial_\nu (\Gamma_{\mu\sigma}^\rho) V^\sigma - \Gamma_{\mu\sigma}^\rho \partial_\nu V^\sigma}_{\partial_\nu(\Gamma_{\mu\sigma}^\rho V^\sigma)}
+\Gamma_{\nu\mu}^\lambda \partial_\lambda V^\rho + \Gamma_{\nu\mu}^\lambda \Gamma_{\lambda \sigma}^\rho V^\sigma
- \Gamma_{\nu\lambda}^\rho \partial_\mu V^\lambda - \Gamma_{\nu\lambda}^\rho \Gamma_{\mu\sigma}^\lambda V^\sigma\\
&= \underbrace{\left[
\partial_\mu \Gamma_{\nu\sigma}^\rho - \partial_\nu \Gamma_{\mu\sigma}^\rho
+ \Gamma_{\mu\lambda}^\rho \Gamma_{\nu\sigma}^\lambda - \Gamma_{\nu\lambda}^\rho \Gamma_{\mu\sigma}^\lambda \right]}_{R_{\sigma\mu\nu}^\rho} V^\sigma\\
&= R_{\sigma\mu\nu}^\rho V^\sigma
\end{align*} 
$$

Some useful tips for the above calculation:

* The covariant derivative of a type $$(r,s)$$ tensor field along $$\mu$$ is given by the expression:
$$
\begin{align}
  {(\nabla_\mu T)^{a_1 \ldots a_r}}_{b_1 \ldots b_s} = {}
    &\frac{\partial}{\partial x^\mu}{T^{a_1 \ldots a_r}}_{b_1 \ldots b_s} \\
    &+ \,{\Gamma ^{a_1}}_{\lambda\mu} {T^{\lambda a_2 \ldots a_r}}_{b_1 \ldots b_s} + \cdots + {\Gamma^{a_r}}_{\lambda\mu} {T^{a_1 \ldots a_{r-1}\lambda}}_{b_1 \ldots b_s} \\
    &-\,{\Gamma^\lambda}_{b_1 \mu} {T^{a_1 \ldots a_r}}_{\lambda b_2 \ldots b_s} - \cdots - {\Gamma^\lambda}_{b_s \mu} {T^{a_1 \ldots a_r}}_{b_1 \ldots b_{s-1} \lambda}.
\end{align}
$$

Meaning that you take the ordinary partial derivative of the tensor and then add $$+{\Gamma^{a_i}}_{\lambda \mu}$$ for every upper index $$a_i$$ and $$-{\Gamma^\lambda}_{b_i \mu}$$ for every lower index $$b_i$$.

* There are generalized Riemannian geometries that have *torsion*, in which the symmetry $$\Gamma_{a b}^\lambda = \Gamma_{b a}^\lambda$$ does *not* hold. Those geometries are not widely used in ordinary gravitational theory. The geometry of general relativity is the *Minkowski-Einstein* geometry which is an extension of Riemannian geometry with a non-positive definite metric, but *it doesn’t involve torsion*.

* Quantities that have different summation indices, but otherwise have the same symbols, are equal and cancel each other. For instance, $$\Gamma_{\mu\lambda}^\rho \partial_\nu V^\lambda$$ is equal to $$\Gamma_{\mu\sigma}^\rho \partial_\nu V^\sigma$$, because indices $$\lambda$$ and $$\sigma$$ are used just as dummy indices for the summation.

---
layout: post
title:  "How to derive the Riemann curvature tensor"
date:   2019-10-29
categories: [mathematics]
tags: ['general relativity', 'mathematics', 'tensors']
---

So, I've decided to bite the bullet and study *general relativity*. I've been postponing it for quite a while, but the idea of my life ending without me having studied one of the most profound and fundamental theories of physics was motivating to say the least. I will be posting random stuff as I go and maybe I'll come back later to edit them, as my understanding of the theory -hopefully- deepens.

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
+ \Gamma_{\mu\lambda}^\rho \Gamma_{\nu\sigma}^\lambda - \Gamma{\nu\lambda}^\rho \Gamma_{\mu\sigma}^\lambda \right]}_{R_{\sigma\mu\nu}^\rho} V^\sigma\\
&= R_{\sigma\mu\nu}^\rho V^\sigma
\end{align*} 
$$

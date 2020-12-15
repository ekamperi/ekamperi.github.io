---
layout: post
title:  Example of Cauchy's residue theorem
date:   2002-12-15
categories: math
tags: ['mathematics']
description: An example of Cauchy's residue theorem for the calculation of a difficult integral
---

So, a friend of mine brought to my attention one such integral that combines $$\pi$$ and $$e$$ constants in the following elegant manner:

$$
\int_{-\infty}^{\infty} \frac{\cos{x}}{(x^2+1)^2}\mathrm{d}x = \frac{\pi}{e}
$$

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/cauchy_example_f.png">
</p>


How to compute an improper integral using the residue theorem from complex analysis. Improper integral is the limit of a definite integral where either of its endpoints go to infinity. E.g.

$$
\lim_{a\to\infty} \int_{-a}^a f(x) \mathrm{d}x
$$

Perhaps this integral can be calculated via some substitution or with Feynman's technique of differentiation under the integral sign. However, I decided to use the nuclear bomb of integration arsenal, the Cauchy residue theorem of complex analysis.


$$\oint_\gamma f(z)\, \mathrm{d}z = 2\pi i \sum_{k=1}^n \operatorname{I}(\gamma, a_k) \operatorname{Res}( f, a_k )
$$

Where $$I(\gamma, a_k)$$ is the winding number, which for simple curves is equal to one, and $$\operatorname{Res}(f,a_k)$$ is the k-th residue of the function $$f$$. The $a_k$ points must be finite and $$f$$ be holomorphic on a simply connected open subset of the complex plane. Finally, the curve $$\gamma$$ is a closed curve which does not meet any of the $$a_k$$ points.

We start by considering the function:

$$
f(z)=\frac{\cos{z}}{(z^2+1)^2}
$$

Now here comes the tricky part. If we do a direct contour integral, it won't work
because the cosine in the complex plane blows up for large imaginary numbers, in both half-planes. So we will instead calculate the contour integral of

$$
\begin{align*}
\int_\gamma \frac{e^{i z}}{(z^2+1)^2} \mathrm{d}z &= 
\int_{-R}^R \frac{e^{i z}}{(z^2+1)^2} \mathrm{d}z +
\int_{\gamma_R} \frac{e^{i z}}{(z^2+1)^2} \mathrm{d}z =
2\pi i \sum_{k=1}^n  \operatorname{Res}( f, a_k ) \Rightarrow\\
\int_{-R}^R \frac{e^{i z}}{(z^2+1)^2} \mathrm{d}z &=
2\pi i \sum_{k=1}^n  \operatorname{Res}( f, a_k ) -
\int_\gamma \frac{e^{i z}}{(z^2+1)^2} \mathrm{d}z
\end{align*}
$$

We then apply the residue theorem and take the limit as $$R\to\infty$$:

$$
\text{P.V.} \int_{-\infty}^{\infty} \frac{e^{i x}}{(x^2+1)^2} \mathrm{d}x =
\lim_{R\to\infty}\left( 2\pi i \sum_{j=1}^n \operatorname{Res}(f, a_k)\right) -
\lim_{R\to\infty} \int_{\gamma_R} \frac{e^{i z}}{(z^2+1)^2} \mathrm{d}z
$$

The singular points in the complex plane are when the denominator is zero:

$$
(z^2+1)^2 =0 \Leftrightarrow \left[(z+i)(z-i)\right]^2=0 \Leftrightarrow (z+i)^2 (z-i)^2 = 0 \Leftrightarrow
\left\{z=i,z=-i\right\}
$$

Therefore we have a pole of second order at $$z = i$$. Its residue is:


$$
\begin{align*}
\operatorname{Res}(f, z=i)
&= \frac{\partial }{\partial z} \left[ (z-i)^2\frac{\exp (i z)}{(z+i)^2 (z-i)^2}\right]_{z=i}\\
&= \frac{\partial }{\partial z} \left[ \frac{\exp (i z)}{(z+i)^2 }\right]_{z=i}\\
&= \frac{i \exp(i z) (z+i)^2 - 2\exp(iz)(z+i)}{(z+i)^4}\\
&= \frac{i e^{-1}4i^2 - 2e^{-1}2i}{(2i)^4} = \frac{-4i-4i}{16e} = -\frac{i}{2e}
\end{align*}
$$

Therefore:

$$
\text{P.V.} \int_{-\infty}^{\infty} \frac{e^{i x}}{(x^2+1)^2} \mathrm{d}x =
\lim_{R\to\infty}\left[ 2\pi i \left(-\frac{i}{2e}\right)\right] = \frac{\pi}{e}
$$

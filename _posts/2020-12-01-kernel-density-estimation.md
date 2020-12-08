---
layout: post
title:  Kernel density estimation
date:   2020-12-08
categories: math
tags: [Mathematica, mathematics, 'statistics']
---

In low dimensional data, we usually plot the histograms to get a feeling of how data are distributed. However, some times we want to come up with a smooth estimation of the underlying probability density function (PDF). One of the methods to do that, is the **kernel density estimation**. In the univariate case, that is when we have only ony variable, it's very straightforward. Assuming we have a set of $$N$$ samples $$x_i = (x_1, x_2, \ldots, x_N)$$, the kernel density estimation, $$\hat{f}$$, of the PDF is:

$$
\hat{f}(x,h) = \frac{1}{N} \sum_{i=1}^{N} K_h (x-x_i)
$$


$$
k(u)\text = \frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{u^2}{2}\right)\\
$$

$$
k_\text{scaled}(h, u) = \frac{1}{h} k\left(\frac{u}{h}\right)
$$

Here is an example:
{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];

(* Assume some sample data *)
pts = {1, 2, 3, 4, 7, 9};

(* Define the kernel *)
k[u_] := 1/Sqrt[2 Pi] Exp[-u^2/2]
k[h_, u_] := 1/h k[u/h]

(* Define the kernel density estimate *)
f[x_, h_] := 
 With[{n = Length@pts}, 1/n Sum[k[h, x - pts[[i]]], {i, 1, n}]] // N

(* Get a kernel density estimate for bandwidth equal to one *)
f[x, 1]
{% endraw %}
{% endhighlight %}

And this is the output:
$$
\begin{align*}
\frac{1}{6} \left(\frac{e^{-\frac{1}{2} (x-9)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-7)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-4)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-3)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-2)^2}}{\sqrt{2 \pi }}+\frac{e^{-\frac{1}{2} (x-1)^2}}{\sqrt{2 \pi }}\right)
\end{align*}
$$

<p align="center">
<img width="70%" height="70%" src="{{ site.url }}/images/kernel_density_estimate.gif" /> 
</p>

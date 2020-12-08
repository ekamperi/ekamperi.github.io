---
layout: post
title:  Kernel density estimation
date:   2020-12-08
categories: math
tags: [Mathematica, mathematics, 'statistics']
---

In low dimensional data, we usually plot histograms to get a feeling of how the data are distributed. However, sometimes we want to have a smooth estimation of the underlying probability density function (PDF). One of the methods to do that is the **kernel density estimation** (KDE). In the univariate case, that is when we have only one variable, it's very straightforward. Here is how we'd do it. Assuming we have a set of $$N$$ samples $$x_i = \{x_1, x_2, \ldots, x_N\}$$, the KDE, $$\hat{f}$$, of the PDF is defined as:

$$
\hat{f}(x,h) = \frac{1}{N} \sum_{i=1}^{N} K_h (x-x_i)
$$

So, basically, **in the kernel density estimation approach we center a smooth kernel function at each data point and then we take their average**. One of the most common kernels is the Gaussian kernel:

$$
K(u)\text = \frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{u^2}{2}\right)\\
$$

The $$K_h$$ is the scaled version of the kernel, i.e., $$K_h(u) = \frac{1}{h} K\left(\frac{u}{h}\right)$$. The parameter $$h$$ of the kernel is called the bandwidth, and this little number is a very critical determinant of our estimate's quality. As important as the kernel choice itself! By tweaking its value, we change the width of the kernel as in the next figure:

<p align="center">
<img width="60%" height="60%" src="{{ site.url }}/images/gaussian_kernels_var_width.png"/> 
</p>

Here is a concrete example that sums all the above:

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

In the following figure, we plot both the individual Gaussian kernels, along with the kernel density estimate. The black dots are our data points and notice how they are at the kernels' center.

<p align="center">
<img width="60%" height="60%" src="{{ site.url }}/images/kernel_density_sum.png" /> 
</p>

In the following animation, we plot the output of *Mathematica*'s built-in `SmoothKernelDistribution[]` function and our own kernel density estimation for varying values of the bandwidth parameter $$h$$. The red dots at the bottom represent our sample data, same as before. Notice how for small values of the bandwidth parameter $$h$$ (during the start of the animation), the KDE is rigged. But, as $$h$$ increases, the estimate gets smoother and smoother. The selection of the parameter $$h$$ is, as we have already said, very crucial.

<p align="center">
<img width="70%" height="70%" src="{{ site.url }}/images/kernel_density_estimate.gif" /> 
</p>

*Mathematica* uses the Silverman's rule of thumb for bandwidth estimation, via the following formaula:

$$
h = 0.9\, \min\left(\hat{\sigma}, \frac{\text{IQR}}{1.34}\right)\, n^{-\frac{1}{5}}
$$

In our case is is:

{% highlight mathematica %}
{% raw %}
sigma = StandardDeviation[pts];
iqr = InterquartileRange[pts];
n = Length[pts];
0.9 Min[sigma, iqr/1.34]*n^(-1/5)

(* 1.93513 *)
{% endraw %}
{% endhighlight %}

Unless we have screwed things up, if we pass $$h = 1.93513$$ to our KDE, we should get an identical result compared to *Mathematica*.

{% highlight mathematica %}
{% raw %}
Grid[{
  Plot[Last@#, {x, 0, 10}, Frame -> {True, True, False, False}, 
     FrameLabel -> {"x", "PDF"}, ImageSize -> Medium, 
     PlotLabel -> First@#, Filling -> Axis] & /@ {
    {"Mathematica's\nSmoothKernelDistribution[]", PDF[SmoothKernelDistribution[pts], x]},
    {"Ours kernel density estimation\n(h=1.93513)", f[x, 1.93513]}}
  }]
{% endraw %}
{% endhighlight %}

<p align="center">
<img width="100%" height="100%" src="{{ site.url }}/images/kde_comparison.png" /> 
</p>

In principle, we could use whatever kernel we'd like, as long as it is symmetric, non-negative and integrates to 1. However, our choice should take into consideration the underlying process that generates our data. In the following example, we use a bisquare kernel on the same data as before.

The bisquare kernel is symmetric, non-negative and integrates to 1.

{% highlight mathematica %}
{% raw %}
ind[u_] := If[Abs[u] < 1, 1, 0];
k2[u_] := 3 (1 - u^2)^2*Abs[u] ind[u]
Plot[k2[u], {3, -2, 3}, PlotRange -> All]
k2[h_, u_] := 1/h k2[u/h]

Integrate[k2[u], {u, -Infinity, Infinity}]
(* 1 *)
{% endraw %}
{% endhighlight %}

<p align="center">
<img width="50%" height="50%" src="{{ site.url }}/images/kde_bisquare_kernel.png" /> 
</p>

In the following animation, we plot our own kernel density estimation for varying values of the bandwidth parameter $$h$$. The red dots at the bottom represent our sample data, same as before.

<p align="center">
<img width="70%" height="70%" src="{{ site.url }}/images/kde_bisquare_animation.gif" /> 
</p>

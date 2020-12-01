---
layout: post
title:  An almost one-liner to construct Mandelbrot set with Mathematica
date:   2020-12-01
categories: math
tags: ['complex numbers', 'functional programming', Mathematica, mathematics, programming]
---

Mandelbrot set is the set of all complex numbers $c$ that fulfill the following condition:

$$
f_c(z) = z^2 + c, \text{does not diverge for } z = 0
$$

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/complex_grid_mandel.png" alt="Complex plan for Mandeblor set">
</p>

So for every point in the complex plane $C$, we assume the complex number $c = a + b i$ and then we calculate the infinite series:

$$
f_c(0), f_c(f_c(0)), f_c(f_c(f_c(0))), \ldots
$$

If this series doesn't diverge, then $c$ belongs to the mandelbrot set. If it diverges, it does not belong. In practice, we only calculate a finite number of terms,
e.g. 256 or whatever. And we color the point $c$ according to the number of iterations that we had to go through before we knew that it diverged or not. 

{% highlight mathematica %}
{% raw %}
f[c_] := NestWhileList[#^2 + c &, c, Abs[#] <= 2. &, 1, 255] // Length
mandel[res_] := Transpose@ParallelTable[f[x + y I], {x, -2., 1, res}, {y, -1, 1, res}];
MatrixPlot[mandel[0.01], ColorFunction -> "BrassTones"]
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/single_mandel.png" alt="Mandelbrot set with Mathematica">
</p>


{% highlight mathematica %}
{% raw %}
Table[
    MatrixPlot[mandel[1/k^2], 
        ColorFunction -> Function[x, If[x > 0.6, Black, Hue@x]], FrameTicks -> None],
    {k, 2, 13}]
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/many_mandel.png" alt="Mandelbrot sets with Mathematica">
</p>


---
layout: post
title:  An almost one-liner to construct the Mandelbrot set with Mathematica
date:   2020-12-01
categories: math
tags: ['complex numbers', 'functional programming', Mathematica, mathematics, programming]
---

## The motivation
Benoit Mandelbrot was a mathematician best known for the discovery of fractal geometry and the famous homonymous set. Mandelbrot was born in November 1924, and I was hoping to honor his birthday by writing a short post for the Mandelbrot set. However, I missed the date because I was working on my master thesis. Anyway, even though I am off by a few days, here you are.

## Definition
Mandelbrot set is the set of all complex numbers $$c$$ that fulfill the following condition:

$$
z_{n+1} = z^2_n + c, \text{does not diverge, starting with } z_0 = 0
$$

<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/complex_grid_mandel.png" alt="Complex plan for Mandeblor set">
</p>

So for every point in the complex plane $$\mathbb{C}$$, we assume the complex number $$c = a + b i$$ and then we calculate the infinite series:

$$
\underbrace{c}_{z_0}, \,\,\underbrace{c^2+c}_{z_1}, \,\,\underbrace{(c^2+c)^2+c}_{z_2}, \,\,\underbrace{((c^2+c)^2+c)^2+c}_{z_3}, \ldots
$$

If this series doesn't diverge, then $$c$$ belongs to the Mandelbrot set. If it diverges, then it does not belong. In practice, we only calculate a finite number of terms, e.g., 256 or whatever. And we color the point $$c$$ according to the number of iterations that we had to go through before we knew that it diverged or not.

For example, let us check whether $$c=1+i$$ is an element of the Mandelbrot set or not. We calculate the sequence $$z_0, z_1, \ldots$$ and notice that it is $$z_1 = (1+i)^2 + (1+i) = 1 + 2i + i^2 + 1 + i = 1 + 3i$$. But, $$\|1+3i\|= \sqrt{1^2+3^2} = \sqrt{10} > 2$$. Therefore, the series diverges, and the complex number $$1+i$$ does not belong to the set. We figured this out with only 1 iteration, therefore we would color this point of complex plane with the "1st color" of our palette.

## The almost one-liner
In the following code, $$f[c\_]$$ checks whether $$c$$ belongs to the Mandelbrot set or not. If it does not, it returns the number of iterations needed to reach this conclusion. If we perform 255 iterations, we assume that the series doesn't diverge, and we take $$c$$ to belong to the set.

{% highlight mathematica %}
{% raw %}
f[c_] := NestWhileList[#^2 + c &, c, Abs[#] <= 2. &, 1, 255] // Length
mandel[res_] := Transpose@ParallelTable[f[x + y I], {x, -2., 1, res}, {y, -1, 1, res}];
MatrixPlot[mandel[0.01], ColorFunction -> "BrassTones"]
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/single_mandel.png" alt="Mandelbrot set with Mathematica">
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
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/many_mandel.png" alt="Mandelbrot sets with Mathematica">
</p>


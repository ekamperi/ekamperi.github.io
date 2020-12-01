---
layout: post
title:  An almost one-liner to construct the Mandelbrot set with Mathematica
date:   2020-12-01
categories: math
tags: ['complex numbers', 'functional programming', Mathematica, mathematics, programming]
---

## The motivation
[Benoit Mandelbrot](https://en.wikipedia.org/wiki/Benoit_Mandelbrot) was a mathematician best known for the discovery of fractal geometry and the [famous homonymous set](https://en.wikipedia.org/wiki/Mandelbrot_set). Mandelbrot was born on 20 November 1924, and I was hoping to honor his birthday by writing a short post for the Mandelbrot set. However, I missed the date because I was working on my master thesis. Anyway, even though I am off by a few days, let's do it!.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/mandelbrot.png" alt="Benoit Mandelbrot photo">
</p>
Source: Photo attributed to Soo Jin Lee.

## The definition
Mandelbrot set is the set of all complex numbers $$c$$ that fulfill the following condition:

$$
z_{n+1} = z^2_n + c, \text{does not diverge, starting with } z_0 = 0
$$

<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/complex_grid_mandel.png" alt="Complex plan for Mandeblor set">
</p>

So for every point in the complex plane $$\mathbb{C}$$, we assume the complex number $$c = a + b i$$ and then we calculate the infinite series:

$$
\underbrace{c}_{z_1}, \,\,\underbrace{c^2+c}_{z_2}, \,\,\underbrace{(c^2+c)^2+c}_{z_3}, \,\,\underbrace{((c^2+c)^2+c)^2+c}_{z_4}, \ldots
$$

If this series doesn't diverge, then $$c$$ belongs to the Mandelbrot set. If it diverges, then it does not belong. In practice, we only calculate a finite number of terms, e.g., 256 or whatever. And we color the point $$c$$ according to the number of iterations that we had to go through before we knew that it diverged or not.

For example, let us check whether $$c=1+i$$ is an element of the Mandelbrot set or not. We calculate the sequence $$z_1, z_2, \ldots$$ and notice that $$z_1 = 1+i$$, and $$\|1+i\|=\sqrt{1^2+1^2}=\sqrt{2} < 2$$. We proceed to the next term of the sequence, with $$z_2 = (1+i)^2 + (1+i) = 1 + 2i + i^2 + 1 + i = 1 + 3i$$. But, $$\|1+3i\|= \sqrt{1^2+3^2} = \sqrt{10} > 2$$. Therefore, the series diverges, and the complex number $$1+i$$ does not belong to the set. We figured this out with only 2 iteration2, therefore we would color this point of complex plane with the "2nd color" of our palette.

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

The following code visualizes the set with increasing resolution. As we scan the complex plane $$\mathbb{C}$$ with a finer resolution, the set's boundary gets increasingly intricate.

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

## The explanation
Recall that we want to calculate the infinite series $$f(0), f(f(0)), f(f(f(0))), \ldots$$, where $$f(x) = x^2 + c$$. In Mathematica, there's the function `Nest[]` that applies iteratively a function to some given expression and returns the result:

{% highlight mathematica %}
{% raw %}
Clear[f];
Nest[f,x,5]
(* f[f[f[f[f[x]]]]] *)
{% endraw %}
{% endhighlight %}

However, we would like to accumulate the intermediate results so that we can count them and check if we have reached the maximum number of iterations. Therefore, we would need something like:

{% highlight mathematica %}
{% raw %}
Table[
 Nest[f, x, k],
 {k, 0, 3}]
(* {x, f[x], f[f[x]], f[f[f[x]]]} *)
{% endraw %}
{% endhighlight %}

Fortunately, Mathematica has `NestList[]` that does precisely this!

{% highlight mathematica %}
{% raw %}
NestList[f, x, 3]
(* {x, f[x], f[f[x]], f[f[f[x]]]} *)
{% endraw %}
{% endhighlight %}

We are almost there, but how can we check whether in each iteration the absolute value of the complex number is less than 2? We need a `NestList[]` function that allows us to inject some sort of condition. Enter `NestListWhen[]`!

{% highlight mathematica %}
{% raw %}
With[{c = 1 + I},
 NestWhileList[#^2 + c &, c, Abs[#] <= 2 &, 1, 255]
]
(* {1 + I, 1 + 3 I} *)
{% endraw %}
{% endhighlight %}

It is now evident what $$f[c\_]$$ does in the one-liner. It applies the lambda function `#^2 + c` iteratively to the previous result, starting with $$z_0 = c$$. At each iteration, it checks whether the absolute value of current $$z_{n+1}$$ is less or equal to 2. If it is, it continues the iterative process. If it's not, it terminates the loop and counts the number of intermediate computations we performed with `Length[]`. Isn't functional programming fantastic?

## The (no pun intended) plot twist
Up until know we considered the infinite series of $$z_{n+1} = z_n + c, c\in\mathbb{C}$$, where $$z_0$$ was fixed to zero and $$c$$ scanned the complex plane. What if we fix $$c$$ to some complex number and let $$z$$ scan the complex plane? This is left as an excercise to the reader, but for $$c=i$$, you should get something along the lines of the following plot:

<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/julia_set.png" alt="Mandelbrot set and Julia set">
</p>

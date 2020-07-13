---
layout: post
title:  "How to sum a divergent series with Pade approximation"
date:   2020-07-13
categories: [mathematics]
tags: ['mathematics', 'perturbation theory']
---

So, this is a follow-up post on [my previous post](https://ekamperi.github.io/mathematics/2020/06/21/perturbation-theory.html)
where we used perturbation theory to calculate the real root of $$x^5 + x = 1$$. In today's post we will, again, use perturbation theory, but we will introduce
the $$\epsilon$$ factor in front of $$x^5$$. Concretely, we will try to solve $$\epsilon x^5 + x = 1$$. This change might seem smooth at first sight, but it turns
out very "violent", because by letting $\epsilon = 0$ we vanish the term $$x^5$$ and, therefore, the other 4 complex roots of the equation. Our intervention
drastically changes the underlying structure of the problem and this will show up later on.

Instead, of doing the calculations by hand, we will indulge ourselves with *Mathematica* this time.

{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];
f[x_] := e*x^5 + x
vars = {a, b, c};
ans[e_] = 1 + Sum[vars[[k]]*e^k, {k, 1, Length@vars}]
expanded = f[ans[e]] // Expand;
getCoeff[n_] := CoefficientList[expanded, e][[n]]
sols = Table[getCoeff[k] == 0, {k, 2, Length@vars + 1}] // Solve;
f[e_] = ans[e] //. Flatten@sols
{% endraw %}
{% endhighlight %}

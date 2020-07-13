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
(* Note how epsilon factor is in front of x^5 *)
f[x_] := e*x^5 + x
vars = {a, b, c, r, s, t, u, v};
ans[e_] = 1 + Sum[vars[[k]]*e^k, {k, 1, Length@vars}]
expanded = f[ans[e]] // Expand;
getCoeff[n_] := CoefficientList[expanded, e][[n]]
sols = Table[getCoeff[k] == 0, {k, 2, Length@vars + 1}] // Solve;
f[e_] = ans[e] //. Flatten@sols
{% endraw %}
{% endhighlight %}

By running the code above, we get the following pwoer series representation of the answer $$x(\epsilon)$$:

$$
x(\epsilon) = 1 - e + 5 e^2 - 35 e^3 + 285 e^4 -2530 e^5 + 23751 e^6 -231880 e^7 + 2330445 e^8
$$

Hoever, if we proceed like before and set $$\epsilon = 1$$ to retrieve the answer to our particular problem, we get a result that is way off:

$$
x(1) = 2120041
$$

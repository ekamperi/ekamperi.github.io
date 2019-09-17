---
layout: post
title:  Arcsin transformation gone wrong
date:   2019-08-22
categories: math
tags: ['machine learning', math, 'neural netowrks']
---

So, I was working on a regression problem and my $$y$$ values, in theory, would fall in the range $$[0,1]$$. In reality, though, most of them were crowded between $$0.9$$ and $$1.0$$. I thought that I could apply some transformation and distribute them more evenly, without though thinking about it much.

Therefore, I applied the arc sine square transformation:

$$
z = \text{arcsin}(\sqrt{x})
$$

After running my code to the transformed data set, I noticed that not only the model didn't perform better, but the results I was getting were very bad.

The reason behind this failure is, I guess, that my data were noisy and that this transformation *increased* the error. You can check this wikipedia article on [propagation of uncertainty](https://en.wikipedia.org/wiki/Propagation_of_uncertainty). A common formula to calculate error propagation is the following. Assuming that you have $$z = f(x, y, \ldots)$$:

Then your error in the variable $$z$$ is given by:

$$
s_z = \sqrt{ \left(\frac{\partial f}{\partial x}\right)^2 s_x^2 + \left(\frac{\partial f}{\partial y} \right)^2 s_y^2 + \cdots}
$$

Where $$s_z$$ represents the standard deviation of the function $$f$$, $$s_x$$ represents the standard deviation of $$x$$, $$s_y$$ represents the standard deviation of $$y$$ and so forth.

So, in my case:

$$ s_z = \frac{dz}{d x} s_x \Rightarrow s_z = \frac{1}{2\sqrt{x (1-x)}} s_x
$$

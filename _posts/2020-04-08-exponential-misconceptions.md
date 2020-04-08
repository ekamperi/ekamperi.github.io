---
layout: post
title:  "Common misconceptions about exponential growth"
date:   2020-04-08
categories: [mathematics]
tags: ['exponential', 'medicine']
---
Coronavirus disease 2019 (COVID-19) is an infectious disease caused by Severe Acute Respiratory Syndrome CoronaVirus 2 (SARS-CoV-2). By now Covid-19 is a pandemic. In many countries where lockdown was not decided in time, the cases showed exponential growth. After talking with many people, I came to understand that lots of folks have some misconceptions about how the exponential function behaves. Here are my thoughts regarding this matter.

1. Lack of fast growth, rules out exponential dynamics.

For "small" values of $$x$$, an exponential function can be approximated by a linear function. This means that it can masquerade as a linear function.

Recall that $$\exp(x)$$ can be expressed as a McLaurin series:

$$
\exp(x) = 1 + x + \frac{1}{2}x^2 + \frac{1}{6}x^3 + \ldots
$$

For "small" values of $$x$$, the contributions of higher order terms (those containing $$x^2$$, $$x^3$$, ...) are small. Therefore, in this case $$\exp(x) \simeq 1 + x$$. Similarly, if we take a look at $$\exp(\lambda x)$$, where the constant $$\lambda$$ affects the growth rate, its McLaurin series becomes:

$$
\exp(\lambda x) = 1 + \lambda x + \frac{1}{2}\lambda^2 x^2 + \frac{1}{6}\lambda^3 x^3 + \ldots
$$

Depending on the value of $$\lambda$$, the exponential function may trick us for a short or long time to think that it's linear. In the following image you may see how for $$\lambda \ll 1$$ various exponential function hide their exponential nature for longer "time" (for larger values of $$x$$).

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/exp_linear1.png" alt="Linear approximation of exponential function">
</p>

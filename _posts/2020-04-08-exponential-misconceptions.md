---
layout: post
title:  "Common misconceptions about exponential growth"
date:   2020-04-08
categories: [mathematics]
tags: ['exponential', 'medicine']
---
Coronavirus disease 2019 (COVID-19) is an infectious disease caused by Severe Acute Respiratory Syndrome CoronaVirus 2 (SARS-CoV-2). At the time of writing Covid-19 has become a global pandemic. In many countries where lockdown was not enforced early on, coronavirus cases showed an exponential growth. After having talked to many people the last days, I came to understand that there are some common misconceptions regarding how an exponential function behaves. Here are my thoughts on this matter.

#### Misconception 1: Lack of fast growth, rules out exponential dynamics.

Many people think that exponential increase is a synonym for explosive, fast, immense growth. This may be true *eventually*, but
for "small" values of $$x$$, an exponential function can be, *initially*, approximated by a linear function. This means that an exponential growth can masquerade as a linear function during the initial phase of the phenomenon.

Recall that $$\exp(x)$$ can be expressed as a McLaurin series:

$$
\exp(x) = 1 + x + \frac{1}{2}x^2 + \frac{1}{6}x^3 + \ldots
$$

For "small" values of $$x$$, the contributions of higher order terms (those containing $$x^2$$, $$x^3$$, ...) are minuscule. Therefore, in this case, the approximation $$\exp(x) \simeq 1 + x$$ holds. Similarly, if we take a look at $$\exp(\lambda x)$$, where the constant $$\lambda$$ affects the growth rate, its McLaurin series is:

$$
\exp(\lambda x) = 1 + \lambda x + \frac{1}{2}\lambda^2 x^2 + \frac{1}{6}\lambda^3 x^3 + \ldots
$$

Depending on the value of $$\lambda$$, the exponential function may trick us for a long time to think that it's linear. In the following image you may see how for different values of $$\lambda < 1$$, the respective functions hide their exponential nature for longer time (for larger values of $$x$$).

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/exp_linear1.png" alt="Linear approximation of exponential function">
</p>

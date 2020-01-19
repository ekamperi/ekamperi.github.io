---
layout: post
title:  "Bayes theorem and likelihood ratios for diagnostic tests"
date:   2020-01-219
categories: [mathematics]
tags: ['medicine', 'mathematics', 'statisticis']
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}


$$
\begin{align*}
P(D^+|T^+) &= \frac{P(T^+|D^+) P(D^+)}{P(T^+)} \\
&=\frac{P(T^+|D^+) P(D^+)}{P(T^+|D^+)P(D^+) + P(T^+|D^-)P(D^-)} =\\
&= \frac{P(D^+)}{P(D^+) + \frac{P(T^+|D^-)}{P(T^+|D^+)}P(D^-)}\\
&= \frac{P(D^+)}{P(D^+) + \frac{P(D^-)}{LR^+}}\\
&= \frac{0.1}{0.1 + \frac{0.9}{1.8}}
\end{align*}
$$

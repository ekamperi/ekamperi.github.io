---
layout: post
title:  "Bayes theorem and likelihood ratios for diagnostic tests"
date:   2020-01-19
categories: [mathematics]
tags: ['medicine', 'mathematics', 'statisticis']
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

I was taking a look at a course in medical research methodology and I stumbled upon the following problem. Suppose that the probability of a patient having a disease is 0.1 in the general population. Suppose also that the likelihood ratio for positive result for some diagnostic test is 2. What is then the probability of the patient having the disease if the patient tested positive for this diagnostic test?

$$
LR^+ = \frac{\text{specificity}}{1-\text{sensitivity}} = \frac{\text{TPrate}}{\text{FPrate}} = \frac{P(T^+|D^+)}{P(T^+|D^-)}
$$

$$
LR^- = \frac{1-\text{sensitivity}}{\text{specificity}} = \frac{\text{FNrate}}{\text{TNrate}} = \frac{P(T^+|D^+)}{P(T^+|D^-)}
$$

$$
\begin{align*}
P(D^+|T^+) &= \frac{P(T^+|D^+) P(D^+)}{P(T^+)} \\
&=\frac{P(T^+|D^+) P(D^+)}{P(T^+|D^+)P(D^+) + P(T^+|D^-)P(D^-)} =\\
&= \frac{P(D^+)}{P(D^+) + \frac{P(T^+|D^-)}{P(T^+|D^+)}P(D^-)}\\
&= \frac{P(D^+)}{P(D^+) + \frac{P(D^-)}{LR^+}}\\
&= \frac{0.1}{0.1 + \frac{0.9}{1.8}}
\end{align*}
$$

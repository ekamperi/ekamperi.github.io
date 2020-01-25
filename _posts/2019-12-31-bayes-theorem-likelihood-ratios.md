---
layout: post
title:  "Bayes theorem and likelihood ratios for diagnostic tests"
date:   2020-01-19
categories: [mathematics]
tags: ['Bayes', 'mathematics', 'medicine', 'statistics']
---

I was taking a look at a course in medical research methodology and I stumbled upon the following problem. Suppose that the probability of a patient having a disease is 0.1 in the general population. Suppose also that the likelihood ratio (LR) for positive result for some diagnostic test is 1.8. What is then the probability of a patient having the disease if he tested positive for this diagnostic test?

The likelikood ratio in the context of a diagnostic test is defined in the following way. For positive tests:

$$
LR^+ = \frac{\text{specificity}}{1-\text{sensitivity}} = \frac{\text{TPrate}}{\text{FPrate}} = \frac{P(T^+|D^+)}{P(T^+|D^-)}
$$

And for negative tests:

$$
LR^- = \frac{1-\text{sensitivity}}{\text{specificity}} = \frac{\text{FNrate}}{\text{TNrate}} = \frac{P(T^-|D^+)}{P(T^-|D^-)}
$$

So, an $$LR^+ > 1$$ indicates that the test result is associated with the disease. If a test has an $$LR^+ = 1$$ it hardly has any practical significance, since the post-test probability is pretty much the same as the pre-test probability. The likelihood ratio, in the latter case, does not add any extra information on what we already know for our patient's risk.

Let's see how likelihood ratio $$LR^+$$ affects our prior credence on whether our patient has indeed the disease. When we want to calculate the probability of an event based on prior knowledge of conditions that might be related to the event, we invoke the [Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). Let's use for the probability that the patient has the disease and the diagnostic test is positive.

Then:

$$
\begin{align*}
P(D^+|T^+) &= \frac{P(T^+|D^+) P(D^+)}{P(T^+)} \\
&=\frac{P(T^+|D^+) P(D^+)}{P(T^+|D^+)P(D^+) + P(T^+|D^-)P(D^-)} =\\
&= \frac{P(D^+)}{P(D^+) + \frac{P(T^+|D^-)}{P(T^+|D^+)}P(D^-)}\\
&= \frac{P(D^+)}{P(D^+) + \frac{P(D^-)}{LR^+}}\\
&= \frac{0.1}{0.1 + \frac{0.9}{1.8}} = 1/6
\end{align*}
$$

So, we started with a prior probability of $$10\%$$ and when we took into consideration the likelihood ratio for this particular diagnostic test that our patient tested positive for, we updated the probability into $$\sim 17\%$$.

Here is a plot of the post-test probability (%) as a function of LR for various values of prior-test probability:

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/likelihoodratio.png" alt="Post-test probability as a function of likelihood ratio for various values of prior-test probability">
</p>

And here is another plot of the post-test probability as a function of prior-test probability for various values of LR. Notice how for and LR equal to 1, the line is $$y=x$$, since the probability does not change.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/likelihoodratio2.png" alt="Post-test probability as a function of likelihood ratio for various values of prior-test probability">
</p>

The Fagan nomogram is a graphical tool for estimating how much the result of a diagnostic test changes the probability that a patient has the disease in question. See also "Fagan TJ. Letter: Nomogram for Bayes theorem. N Engl J Med. 1975;293(5):257". To use Fagan nomogram, you need to have an estimate of the probability of the disease prior to testing. This estimate can be based on the prevalence of the disease modified accordingly by considering other risk factors. You also need to know the likelihood ratio for the diagnostic test (obviously).

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/fagan_nomogram.png" alt="Fagan nomogram">
</p>



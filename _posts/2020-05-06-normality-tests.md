---
layout: post
title:  "Normality tests"
date:   2020-05-06
categories: [math]
tags: [mathematics, Shapiro-Wilk, statistics]
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

### Introduction
Some statistical tests (e.g., t-test for independent samples) assume that the data in question are modeled by a normal distribution. Therefore, before using such a test, one must first confirm that the data do not deviate much from a normal distribution. There are many ways to do this, and ideally one should combine them, such as looking at the **mean/median** (should be approximately equal), **skewness** (should be approximately zero), **histograms** (should be approximately bell-shaped), **Q-Q plots** (should approximately form a line)  and, ultimately, running a **normality test**, such as Shapiro-Wilk.

### Disagreement between plots and Shapiro-Wilk test
So, what happens when the plots say that the data aren't normally distributed, but Shapiro-Wilk test disagree? Or vice versa?

For *small sample sizes*, the histograms rarely resemble the shape of a normal distribution, and that's fine. As soon as one increases the sample size, the shape of the distribution converges to that of the underlying distribution. On the other hand, the Shapiro-Wilk test correctly implies normality, as you can see in the p-values of the following plot.

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/shapiro1.png" alt="Shapiro Wilk test">
</p>

{% highlight R %}
{% raw %}
################################################################################
#                                 NORMAL DISTRIBUTION
################################################################################
plot_sample <- function(sample_size) {
    sample_dist <- rnorm(sample_size, mean = 0, sd = 1)
    sp <- shapiro.test(sample_dist)
    par(ps = 10)
    hist(sample_dist, xlab = "x",
        main = sprintf("Sample size = %d\nShapiro p-value = %.3f",
                       sample_size, sp$p.value),
        col = "steelblue", border = "white", prob = T)
    lines(density(sample_dist), col = "red")
}
par(mfrow = c(2, 2))
lapply(c(30, 50, 200, 5000), plot_sample)
{% endraw %}
{% endhighlight %}

Shapiro-Wilk test begins to behave in a problematic manner when the *sample size is large*. In the following plots, I've fixed the sample size equal to 5000 (this is the largest allowed value for R's `shapiro.test()` anyway). Notice how the test rejects normality even for *slightly skewed normal distributions*. On the other hand, histograms look pretty good!

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/shapiro2.png" alt="Shapiro Wilk test">
</p>

{% highlight R %}
{% raw %}
################################################################################
#                             SKEWED NORMAL DISTRIBUTION
################################################################################
library(fGarch)

plot_sample2 <- function(skewness_param) {
    N <- 5000
    sample_dist <- rsnorm(n = N, mean = 0, sd = 1, xi = skewness_param)
    sp <- shapiro.test(sample_dist)
    par(ps = 10)
    hist(sample_dist, xlab = "x",
        main = sprintf("Skewness = %.2f\nShapiro p-value = %.5f",
                       skewness_param, sp$p.value),
        col = "steelblue", border = "white", prob = T)
    lines(density(sample_dist), col = "red")
}
par(mfrow = c(2, 2))
lapply(c(1, 1.05, 1.1, 1.15), plot_sample2)
{% endraw %}
{% endhighlight %}

So, the rule of thumb I follow is this: if histograms and Shapiro-Wilk disagree, for small sample size, I go with Shapiro (of course, I check mean/median, etc.). For a large sample size, I go with the histograms.

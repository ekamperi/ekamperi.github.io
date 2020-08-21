---
layout: post
title:  "Benford's law and COVID-19 conspiracy theories"
date:   2020-08-21
categories: [mathematics]
tags: ['covid-19', 'mathematics', 'statistics']
---

How do we know if a list of numbers is made up or has come up in a "natural" way?

This question may sound indifferent at first, but it has several applications. In the 2016 film, "The Accountant",
Ben Affleck's character uses Benford's law, which we'll talk about today, to expose fraud in a robotics company.
In particular, he notes that the digit "3" appears more often than expected in some financial numerical data.

Benford's law also called the law of "irregular numbers" or "the first digit", is an observation about how often
different numbers appear as digits in real numbers. The law states that in many cases (but not all!) when we have
lists of numbers produced by a natural process (e.g., number of coronavirus infections 😉), the most significant
digit is likely to be small.

The number "1" appears as the most significant digit in about 30% of cases, while "9" appears in less than 5% of
cases. If the digits were evenly distributed, we would encounter each with a frequency of about 11.1% (f=1/9 * 100%).
The law also makes predictions for the distribution of second digits, third digits, combinations of digits, etc.

People who do not know Benford's law, when making up numbers tend to distribute their digits evenly. Thus, a simple
comparison of the first or second digit frequency distribution could easily show "abnormal" results.

In the figure below, we have plotted how often the various digits appear in the number of coronavirus cases of
Washington state. We got the data from the link [1], and we can not guarantee their authenticity. However, it seems
that the distribution of numbers in the 1st and 2nd most significant position of the number of cases approaches the
theoretical distribution of Benford's law (dotted red line), so we can not assume that someone "cooked" the data,
within the confidence interval the statistics imply.

{% highlight R %}
{% raw %}
library(benford.analysis)
us <- read_csv("C:\\Users\\stath\\Downloads\\us-states.csv")
covid19 <- us[us$state == "Washington",]
bfd.cp <- benford(covid19$cases, number.of.digits = 1)
plot(bfd.cp)
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/covid19_benford_law.png">
</p>

In [this article](https://www.nature.com/articles/d41586-020-01565-5?fbclid=IwAR1FG9iAmGUuJhmgCNTZMHMdJuH4nJ3D2SGCw26lg1CjEPoHzXh4qzrjr40)
the authors performed a thorough analysis and found that records of cumulative infections and deaths from the United States, Japan,
Indonesia and most European nations adhered well to the Benford's law, consistent with accurate reporting. The results can be [found here](go.nature.com/2kqtut2).

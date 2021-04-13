---
layout: post
title: "Decision Trees: Gini index vs entropy"
date:   2021-04-13
categories: [machine learning]
tags: ['decision trees', 'machine learning', mathematics, 'R language']
description: Gini index vs entropy in decision trees with imbalanced datasets 
---

### Introduction
Decision trees are tree-based methods that are used for both regression and classification. They work by segmenting the feature space into several simple regions. To predict a given observation, we assume either the mean or the most frequent class of the training points inside the region to which our observation falls. Decision trees are straightforward to interpret, and as a matter of fact, they can be even easier to interpret than linear or logistic regression. Perhaps because decision trees are more close to the way the human decision-making process works. On the downside, trees usually lack the level of predictive accuracy of other regression and classification methods. Also, they can be susceptible to changes in the training dataset, where a slight change in it may cause a dramatic change in the final tree. That’s why bagging, random forests, and boosting are used to construct more robust tree-based prediction models. But that’s for another day.

### Gini impurity and information entropy
Trees are constructed via recursive binary splitting, and two measures are usually used. One is the Gini index, and the other one is information entropy. Both of these measures are pretty similar numerically. They both take small values, if most observations fall into the same class in a node. Contrastly, they assume a maximum value when there is an equal number of observations across all classes in a node. Such a node is called impure, and the Gini index is also referred to as a measure of impurity.

Concretely, for a set of items with $$K$$ classes, and $$p_i$$ being the fraction of items labeled with class $$i\in {1,2,\ldots,K}$$, the Gini impurity is defined as:

$$
G = \sum_{k=1}^K p_k (1 - p_k) = 1 - \sum_{k=1}^N p_k^2
$$

And information entropy as:

$$
H = -\sum_{k=1}^K p_k \log p_k
$$

In the following plot, both metrics are plotted assuming a set of 2 classes appearing with probability $$p$$ and $$1-p$$, respectively. Notice how for small values of $$p$$ Gini takes lower values than entropy. This is a key observation that will prove useful in the context of **imbalanced datasets**.

<p align="center">
<img style="width: 70%; height: 70%" src="{{ site.url }}/images/decision_trees/gini_vs_entropy.png" alt="Gini vs entropy">
</p>

The Gini index is used by the CART (classification and regression tree) algorithm for classification trees, whereas information gain via entropy reduction is used by algorithms like [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm). In the following image we see a part of a decision tree for predicting whether a person receiving a loan will be able to pay it back. The left node is an example of a node with low impurity, since most of the observations fall into the same class. Contrast this with the node on the right where observations of different classes are mixed in.

<p align="center">
<img style="width: 70%; height: 70%" src="{{ site.url }}/images/decision_trees/pure_vs_impure_node.png" alt="Decision trees: pure vs impure nodes">
</p>

Image taken from "Provost, Foster; Fawcett, Tom. Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking".

Let's calculate the **Gini impurity of the left node**:

$$
\begin{align}
G\left(\text{Balance < 50K}\right)
&= 1-\sum_{k=1}^{2} p_k^2 = 1-p_1^2 - p_2^2\\
&=1-\left(\frac{12}{13}\right)^2 -\left(\frac{1}{13}\right)^2
\simeq 0.14
\end{align} 
$$

And the **Gini impurity of the right node**:

$$
\begin{align}
G\left(\text{Balance} \ge \text{50K}\right)
&= 1-\sum_{k=1}^{2} p_k^2 = 1-p_1^2 - p_2^2\\
&=1-\left(\frac{4}{17}\right)^2 -\left(\frac{13}{17}\right)^2
\simeq 0.36
\end{align} 
$$

We notice that the left node has a lower Gini impurity index, which we'd expect since $$G$$ measures impurity, and the left node is purer relative to the right one. Let's calculate now the **entropy of the left node**:

$$
\begin{align}
H\left(\text{Balance < 50K}\right)
&= -\sum_{k=1}^{2} p_k \log{p}_k = -p_1 \log{p}_1 -p_2 \log{p}_2\\
&=-\frac{12}{13}\log\left(\frac{12}{13}\right) -\frac{1}{13}\log\left(\frac{1}{13}\right)
\simeq 0.27 nats
\end{align}
$$

Depending on whether we are using $$log_2$$ or $$log_e$$ in the entropy formula we get the result in *bits* or *nats*, respectively. For instance, here it's $$H \simeq 0.39 bits$$. Let's calculate the **entropy of the right node** as well:

$$
\begin{align}
H\left(\text{Balance}\ge\text{50K}\right)
&= -\sum_{k=1}^{2} p_k \log{p}_k = -p_1 \log{p}_1 -p_2 \log{p}_2\\
&=-\frac{4}{17}\log\left(\frac{4}{17}\right) -\frac{13}{17}\log\left(\frac{13}{17}\right)
\simeq 0.55 nats
\end{align} 
$$

Again, if we'd use base 2 in the entropy's logarithm, we'd get $$H \simeq 0.79 bits$$. Units aside, we see that the left node has a lower entropy than the right one, which is to be expected, since the left one is in a more *ordered* state and entropy measures *disorder*. So, it's $$H_\text{left} \simeq 0.27 nats$$ and  $$H_\text{right} \simeq 0.55 nats$$. The various algorithms for constructing decision trees, pick the next feature to split in such a way that maximum reduction of impurity is achieved.

Let's calculate how much entropy is reduced by splitting on the "Balance" feature:

$$
\begin{align*}
H(Parent) &= -\frac{16}{30} \log\left(\frac{16}{30}\right) -\frac{14}{30}\log\left(\frac{16}{30}\right)\simeq 0.69nats\\
H(Balance) &= \frac{13}{30} \times 0.27 + \frac{17}{30} \times 0.55 \simeq 0.43nats
\end{align*}
$$

Therefore, the information gain by splitting on the "Balance" feature is:

$$
\text{IG} = H(Parent) - H(Balance) = 0.69 - 0.43 = 0.26nats
$$

### An example of an imbalanced dataset
{% highlight R %}
{% raw %}
# Load the necessary libraries and the dataset 
library(ROSE)
library(rpart)
library(rpart.plot)
data(hacide)

# Check imbalance on training set
table(hacide.train$cls)
#
#   0   1 
# 980  20 
{% endraw %}
{% endhighlight %}

First, we will fit a decision tree by using Gini as the split criterion.
{% highlight R %}
{% raw %}
# Use gini as the split criterion
tree.imb <- rpart(cls ~ ., data = hacide.train, parms = list(split = "gini"))
pred.tree.imb <- predict(tree.imb, newdata = hacide.test)
accuracy.meas(hacide.test$cls, pred.tree.imb[,2])
roc.curve(hacide.test$cls, pred.tree.imb[,2], plotit = T, main = "Gini index")
{% endraw %}
{% endhighlight %}

And this is the ROC curve which shows how horrible our classifier is.

<p align="center">
<img style="width: 70%; height: 70%" src="{{ site.url }}/images/decision_trees/gini_auc.png" alt="Gini vs entropy ROC curv">
</p>

Let's take a look at the decision tree itself. You may notice that the left node has 10 observations of the minority class and 989 of the dominant class. 

{% highlight R %}
{% raw %}
rpart.plot(tree.imb, main = "Gini Index", type = 5, extra = 3)
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 70%; height: 70%" src="{{ site.url }}/images/decision_trees/gini_tree.png" alt="Gini vs entropy ROC curv">
</p>

{% highlight R %}
{% raw %}
# Use information gain as the split criterion
tree.imb <- rpart(cls ~ ., data = hacide.train, parms = list(split = "information"))
pred.tree.imb <- predict(tree.imb, newdata = hacide.test)
accuracy.meas(hacide.test$cls, pred.tree.imb[,2])
roc.curve(hacide.test$cls, pred.tree.imb[,2], plotit = T)
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 70%; height: 70%" src="{{ site.url }}/images/decision_trees/entropy_auc.png" alt="Gini vs entropy ROC curv">
</p>

{% highlight R %}
{% raw %}
rpart.plot(tree.imb, main = "Information Gain", type = 5, extra = 3)
{% endraw %}
{% endhighlight %}


<p align="center">
<img style="width: 70%; height: 70%" src="{{ site.url }}/images/decision_trees/entropy_tree.png" alt="Gini vs entropy ROC curv">
</p>

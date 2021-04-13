---
layout: post
title: "Decision Trees: Gini index vs entropy"
date:   2021-04-13
categories: [machine learning]
tags: ['machine learning', Mathematics, R]
description: Gini index vs entropy in decision trees with imbalanced datasets 
---

Decision trees are tree-based methods that are used for both regression and classification. They work by segmenting the feature space into several simple regions. To predict a given observation, we assume either the mean or the most frequent class of the training points inside the region to which our observation falls. Decision trees are straightforward to interpret, and as a matter of fact, they can be even easier to interpret than linear or logistic regression. Perhaps because decision trees are more close to the way the human decision-making process works. On the downside, trees usually lack the level of predictive accuracy of other regression and classification methods. Also, they can be susceptible to changes in the training dataset, where a slight change in it may cause a dramatic change in the final tree. That’s why bagging, random forests, and boosting are used to construct more robust tree-based prediction models. But that’s for another day.

Trees are constructed via recursive binary splitting, and two measures are usually used. One is the Gini index, and the other one is entropy. Both of these measures are pretty similar numerically. They both take small values, near zero, if all observations fall into the same class in a node. Contrastly, they assume a maximum value when there is an equal number of observations across all classes in a node. Such a node is called impure, and the Gini index is also referred to as a measure of impurity.

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/decision_trees/pure_vs_impure_node.png" alt="Decision trees: pure vs impure nodes">
</p>

Image taken from "Provost, Foster; Fawcett, Tom. Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking".

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/decision_trees/gini_vs_entropy.png" alt="Gini vs entropy">
</p>

{% highlight R %}
{% raw %}
# Load the necessary libraries and the dataset 
library(ROSE)
library(rpart)
library(rpart.plot)
data(hacide)

# Check imbalance on training set
table(hacide.train$cls)
> table(hacide.train$cls)

  0   1 
980  20 
>
{% endraw %}
{% endhighlight %}


{% highlight R %}
{% raw %}
# Use gini as the split criterion
tree.imb <- rpart(cls ~ ., data = hacide.train, parms = list(split = "gini"))
pred.tree.imb <- predict(tree.imb, newdata = hacide.test)
accuracy.meas(hacide.test$cls, pred.tree.imb[,2])
roc.curve(hacide.test$cls, pred.tree.imb[,2], plotit = T, main = "Gini index")
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/decision_trees/gini_auc.png" alt="Gini vs entropy ROC curv">
</p>

{% highlight R %}
{% raw %}
rpart.plot(tree.imb, main = "Gini Index", type = 5, extra = 3)
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/decision_trees/gini_tree.png" alt="Gini vs entropy ROC curv">
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
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/decision_trees/entropy_auc.png" alt="Gini vs entropy ROC curv">
</p>

{% highlight R %}
{% raw %}
rpart.plot(tree.imb, main = "Information Gain", type = 5, extra = 3)
{% endraw %}
{% endhighlight %}


<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/decision_trees/entropy_tree.png" alt="Gini vs entropy ROC curv">
</p>

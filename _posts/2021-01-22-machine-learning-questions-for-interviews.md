---
layout: post
title: "A list of machine-learning questions for interviews"
date:   2021-01-22
categories: [machine learning]
tags: ['machine learning', Python]
description: A list of machine-learning questions for interviews along with a short answer
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

**Here I'll be adding questions regarding machine-learning and data-science in general. In the future I'll group them by subject.**

### What are collinearity and multicollinearity? How can we detect them?
Collinearity is the existence of two correlated predictor variables in a regression problem. Multicollinearity is when there are more than two variables correlated. Collinearity hurts regression, both coefficient estimation and the interpretation of the model. It can be detected with a scatterplot matrix, a correlation matrix, and via the calculation of Variance Inflation Factors (VIFs).

### What's a type I and type II error? Which one is worse?
Type I is a false positive (you tell a healthy man he has cancer). Type II is a false negative (you tell a man with cancer he is healthy) Regarding which one is worse, it depends on the context. A false negative for a spam filter isn't a big deal, but a false negative for a cancer diagnostic test it is.

### What is a ROC curve?
ROC stands for Receiver Operating Characteristic. ROC curve is a plot of True Positive rate (sensitivity) vs. False Positive rate (1-specificity) for different diagnostic test cut-off values. It demonstrates the inherent tradeoff of sensitivity vs. specificity (e.g., a covid test that is too sensitive won't miss any real covid cases but will have a high number of false-positive). The close the curve on the left/upper-hand border of the plot, the better. A random classifier is represented by a 45-degree diagonal line. The area under the curve (AUC) is a measure of the test's accuracy.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/machine-learning-q/roc_curve.png" alt="A ROC curve">
</p>

### What are the differences between K-nearest neighbor (KNN) and k-means clustering (K-Means)?
KNN is a supervised algorithm, K-Means is an unsupervised technique. The former is used for classification or regression, the latter for clustering. In KNN, "k" is the number of nearest neighbors used to perform the classification (or regression). In K-Means is the number of clusters the algorithm will come up with.

### What is a confusion matrix?
It is a matrix used for evaluating a classification algorithm. The diagonal elements consist of true negative and true positive, whereas the off-diagonal consist of false negative and false positive.

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/machine-learning-q/confusion_matrix.png" alt="A confusion matrix">
</p>

### What are some differences between classification and regression?
In classification, the model predicts discrete class labels, in regression some continuous quantity. A classification problem with two classes (e.g., yes/no, spam/not spam, legit/fraud) is called binary and multi-class when there are more than two classes. A regression problem with many input variables is called a multivariate regression problem. Predicting the value of a cryptocoin over a period of time is a regression problem.

### What is reinforcement learning?
It is when an agent interacts with its environment by generating actions and then discovers rewards or penalties. It is suited for problems that are reward-based.

### What is bias, variance, and the bias-variance tradeoff?
Bias error is the difference between the average prediction of a model and the correct value we want to predict. A model with high bias is a model that doesn't learn from the training data and is overly simplistic. The model will perform poorly to the training data and the test data.

Variance is the variability of a model's prediction for a given data point. A high variance means that the model is learning from the training data more than there exists to learn. The model will perform very well on the training data but poor on the test data.

If we make our model too simple, we will decrease its variance but also increase its bias. If we make it too complex, we will reduce its bias but also increase its variance. This is the bias-variance tradeoff.

The model's total error is the sum of the bias squared, the variance, and the irreducible error. The latter is a term that we can't reduce no matter how good a model we build.

### What is a lambda function, and when do we use one?
It's a function without a name, i.e., an anonymous function. We use lambda functions when we need them for a short period of time, and then we dispose of them. For instance, to sort a list of tuples in Python based on the value of the second element, we would write:

{% highlight python %}
{% raw %}
lst_of_values = [('bob', 18), ('alice', 25), ('george', 99), ('stathis', 10)]
print(lst_of_values)
sorted_lst = sorted(lst_of_values, key=lambda x: x[1])

[('bob', 18), ('alice', 25), ('george', 99), ('stathis', 10)]
[('stathis', 10), ('bob', 18), ('alice', 25), ('george', 99)]
print(sorted_lst){% endraw %}
{% endhighlight %}

### What is the difference between classification and clustering?
Classification is supervised learning, where we assign a label class to some input. E.g., the input is a human face, and the label is whether the facial expression (neutral, smiles, sad, etc.). Clustering is unsupervised learning where we put similar inputs together without actually providing the labels. In a sense, the model will figure out the labels by itself. E.g., we provide a list of furniture images, and the model will put all the tables in a cluster in the feature space, all the chairs in another cluster in the feature space, etc.

### What is regularization? What are the L1 and L2 regularization methods, and how do they differ?
Regularization is the process of controlling the model complexity (e.g., the number of model parameters or the values the parameters take). It is realized as an additional term that is added to the loss function. The number of the model's parameters is regularized by L1 and the weights by L2. L1 is also called LASSO and L2 ridge.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/machine-learning-q/lasso_vs_ridge.png" alt="LASSO vs ridge regularization">
</p>

[You can read more here](https://ekamperi.github.io/machine%20learning/2019/10/19/norms-in-machine-learning.html#regularization)

### What is an eigenvector? And what is an eigenvalue?
The eigenvector is a vector whose direction remains unchanged after a linear transformation is applied to it. The eigenvalue is the scalar by which an eigenvector is scaled when a linear transformation is applied to it. For a square matrix $$A$$, a vector $$\mathbf{x} \ne 0$$, and some number $$\lambda$$, it holds that $$\mathbf{A} \mathbf{x} = \lambda \mathbf{x}$$.

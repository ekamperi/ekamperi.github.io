---
layout: post
title: "Bayesian optimization for hyperparameter tuning"
date:   2021-05-08
categories: [machine learning]
tags: [algorithms, 'Bayes theorem', 'neural networks', optimization, programming]
description: An introduction to Bayesian-based optimization for tuning hyperparameters in machine learning models
---

### Introduction
Scene: We died and ended up in [Dante's inferno](https://en.wikipedia.org/wiki/Inferno_(Dante)) -- the optimization version.

We are asked to optimize a function **we don't have an analytic expression** for. It follows that **we don't have access to the first or second derivatives**, hence using [gradient descent](https://ekamperi.github.io/machine%20learning/2019/07/28/gradient-descent.html) or [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) is a no-go. Also, **we don't have any convexity guarantees** about $$f(x)$$. Therefore, methods from the convex optimization field are also not an option. The only thing we can do is to evaluate $$f(x)$$ at some $$x$$'s. As if the situation was not bad enough, **the function that we want to optimize is very costly**. So, we can't just go ahead and massively evaluate $$f(x)$$ in, say, 100 billion random points and keep the one $$x$$ that optimizes $$f(x)$$'s value.

<p align="center">
<img style="width: 35%; height: 35%" src="{{ site.url }}/images/bayesian_optimization/dante_inferno.png" alt="Dante inferno">
</p>

To put it another way, we want to optimize an expensive, black-box, derivative-free, possibly non-convex function. And for this kind of problem, **Bayesian Optimization (BO)** is a robust method.

The evaluation of the function might not even be computational at all. Let me give you a couple of examples, where $$f(x)$$ is not something you can calculate with a computer.
1. You are a researcher, and you investigate combinations of chemotherapeutic drugs for their ability to kill cancer cells. You have 20 candidate molecules, and you need to come up with an effective drug combination. Evaluating the objective function $$f(x)$$ in this context entails conducting actual experiments in the lab requiring personnel, consumables and waiting for hours or days for the experiment to complete. Therefore, considering all the 190 combinations is not a realistic approach.
2. You work as a consultant for an oil company, and you need to maximize a probability density function $$f(\text{LAT}, \text{LONG})$$ of finding oil if we drill on $$(\text{LAT}, \text{LONG})$$ coordinates. Drilling costs lots of money, therefore we need to make good educated guesses, and we need to do so with only a few trials.

In other cases, however, $$f(x)$$ might be computational. For instance, we may define it as the cross-validation error of a machine-learning model, whose hyperparameters we want to tune. So, to sum up, we want to optimize $$f(x)$$ and:

1.	We don’t have a formula for $$f(x)$$
2.	We don’t have access to its derivatives $$f'(x)$$ and $$f''(x)$$
3.	We don't have any convexity guarantees for $$f(x)$$
4.	$$f(x)$$ is expensive to evaluate

### The ingredients of Bayesian Optimization
#### Gaussian Processes
Since we don't have an expression for the objective function, the first step is to **use a surrogate model to approximate $$f(x)$$**. It is typical in this context to use Gaussian Processes (GPs), as we have already discussed in a [previous blog post](https://ekamperi.github.io/mathematics/2021/03/30/gaussian-process-regression.html). It's vital that you grasp the concept of GPs, and then BO will require almost no mental effort to sink. Once we have built a proxy model for $$f(x)$$, we want to decide which point $$x$$ to sample next. For this, we need an acquisition function, which kind of "reads" the GP and outputs the best guess $$x$$. So, in BO, there are two components: the *surrogate model*, which most often is a Gaussian Process modeling $$f(x)$$, and the *acquisition function* that yields the next $$x$$ to evaluate. Having said that, a BO algorithm would look like this:

1. Evaluate $$f(x)$$ at $$n$$ initial points
2.	While $$n \le N$$ repeat:
    * Update the GP posterior using all available data
    * Compute the acquisition function using the current GP
    * Let $$x_n$$ be the maximizer of the acquisition function
    * Evaluate $$y_n = f(x_n)$$
    * Increment $$n$$
3.	Return either the $$x$$ evaluated with the largest $$f(x)$$, or the point with the largest posterior mean.

#### Acquisition function
As we have already noted, the role of the acquisition function is to guide the next best point to sample $$f$$. Acquisition functions are constructed so that a high value corresponds to potentially high values of the objective function. Either because the prediction is high or because the uncertainty is high. So, acquisition functions favor regions that already correspond to optimal values or areas that haven't been explored yet. This is known as the so-called **exploration-exploitation trade-off**.

If you have played strategy games, like Age of Empires or Command & Conquer, you are already familiar with the concept. Initially, we are placed at some part of the map, and only the immediate area is visible to us. We may choose to sit there and mine any resources we already have access to or send a scouter to explore the invisible part of the map. By exploring the map, we risk meeting the enemy and getting killed, but also, we may find some high-value resources.

<p align="center">
<img style="width: 80%; height: 80%" src="{{ site.url }}/images/bayesian_optimization/age_of_empires.png" alt="Exploitation vs exploration trafeodd">
</p>

To find the next point to evaluate, we optimize the acquisition function. This an optimization problem itself, but luckily does not require the evaluation of the objective function. In some cases, we may even derive the exact equation and find a solution with, say, gradient-based optimization. There are three often cited acquisition functions: **expected improvement** (EI), **maximum probability of improvement** (MPI), and **upper confidence bound** (UCB). Although often cited last, I think it's best to talk about UCB because it contains explicit exploitation and exploration terms:

$$
a_{\text{UCB}}(x;\lambda) = \mu(x) + \lambda \sigma(x)
$$

With UCB, the exploitation *vs.* exploration trade-off is explicit and easy to tune via the parameter $$\lambda$$. Concretely, we construct a weighted sum of the expected performance captured by $$\mu(x)$$ of the Gaussian Process, and of the uncertainty $$\sigma(x)$$, captured by the standard deviation of the GP. Assuming a small $$\lambda$$, BO will favor solutions that are expected to be high-performing, i.e., have high $$\mu(x)$$. Conversely, high values of $$\lambda$$ will make BO favor the exploration of currently uncharted areas in the search space. 

### A concrete example
Let's import some of the stuff we will be using:

{% highlight python %}
{% raw %}
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
{% endraw %}
{% endhighlight %}

Then, we construct an artificial training dataset with many classes, where some of the features are informative, and some are not:

{% highlight python %}
{% raw %}
# Create a random n-class classification problem.

# n_features is the total number of features
# n_informative is the number of informative features 
# n_redundant features are generated as random linear combinations of the informative features

X_train, y_train = make_classification(n_samples=2500, n_features=20, n_informative=7, n_redundant=3)
{% endraw %}
{% endhighlight %}

We define our objective/cost/loss function. This is the $$f(\mathbf{x})$$ that we want talked about in the introduction, and $$\mathbf{x} = [C, \gamma]$$ is the domain of the function. Therefore, we want to find the best combination of $$C, \gamma$$ values that minimizes $$f(\mathbf{x})$$. The machine learning model that we will be using is a [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support-vector_machine), and the loss will be derived from the average 3-fold cross-validation score.

{% highlight python %}
{% raw %}
def objective(args):
    "'Define the loss function / objective of our model.

    We will be using an SVM parameterized by the regularization parameter C
    and the parameter gamma.
    
    The C parameter trades off correct classification of training examples
    against maximization of the decision function's margin. For larger values
    of C, a smaller margin will be accepted.

    The gamma parameter defines how far the influence of a single training
    example reaches, with larger values meaning 'close'. 
    '''
    C, gamma = args
    model = SVC(C=10 ** C, gamma=10 ** gamma, random_state=12345)
    loss = 1 - cross_val_score(estimator=model, X=X_train, y=y_train, scoring='roc_auc', cv=3).mean()
    return {'params': {'C': C, 'gamma': gamma}, 'loss': loss, 'status': STATUS_OK }
{% endraw %}
{% endhighlight %}

Now, we will use the `fmin` function from the `hyperopt` package.

{% highlight python %}
{% raw %}
# Minimize a function using the downhill simplex algorithm.
# This algorithm only uses function values, not derivatives or second derivatives.
trials = Trials()
best = fmin(objective,
    space=[hp.uniform('C', -4., 1.), hp.uniform('gamma', -4., 1.)],
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials)
{% endraw %}
{% endhighlight %}

Let's print the results:

{% highlight python %}
{% raw %}
print(best)
100%|██████████| 1000/1000 [13:01<00:00,  1.28trial/s, best loss: 0.046323449153816476]
{'C': 0.7280999882033379, 'gamma': -1.6752085795502363}
{% endraw %}
{% endhighlight %}


Let us now extract the  value of our objective function for every $$C, \gamma$$ pair:

{% highlight python %}
{% raw %}
# Extract the loss for every combination of C, gamma
results = trials.results
ar = np.zeros(shape=(1000,3))
for i, r in enumerate(results):
    C = r['params']['C']
    gamma = r['params']['gamma']
    loss = r['loss']
    ar[i] = C, gamma, loss
{% endraw %}
{% endhighlight %}

And then use it to plot the loss surface:

{% highlight python %}
{% raw %}
C, gamma, loss = ar[:, 0], ar[:, 1], ar[:, 2]

fig, ax = plt.subplots(nrows=1)
ax.tricontour(C, gamma, loss, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(C, gamma, loss, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(C, gamma, 'ko', ms=1)
ax.set(xlim=(-4, 1), ylim=(-4, 1))
plt.title('Loss as a function of $10^C$, $10^\gamma$')
plt.xlabel('C')
plt.ylabel('gamma')

plt.show()
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 65%; height: 65%" src="{{ site.url }}/images/bayesian_optimization/bayesian_optimization.png" alt="Bayesian optimization">
</p>

Since the parameter space is just 2-dimensional, the dataset relatively small, and the SVM training fast, we can brute-force compute the value of the objective function for all possible values of $$C$$ and $$\gamma$$. These will be our ground-truth data against which we will compare the results from the BO run.

{% highlight python %}
{% raw %}
def sample_loss(args):
    C, gamma = args
    model = SVC(C=10 ** C, gamma=10 ** gamma, random_state=12345)
    loss = 1 - cross_val_score(estimator=model, X=X_train, y=y_train, scoring='roc_auc', cv=3).mean()
    return loss

lambdas = np.linspace(1, -4, 25)
gammas = np.linspace(1, -4, 20)
param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])

real_loss = [sample_loss(params) for params in param_grid]
{% endraw %}
{% endhighlight %}

And here is the respective contour plot:

{% highlight python %}
{% raw %}
C, G = np.meshgrid(lambdas, gammas)
plt.figure()
cp = plt.contourf(C, G, np.array(real_loss).reshape(C.shape), cmap="RdBu_r")
plt.colorbar(cp)
plt.title('Loss as a function of $10^C$, $10^\gamma$')
plt.xlabel('$C$')
plt.ylabel('$\gamma$')
plt.show()
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 65%; height: 65%" src="{{ site.url }}/images/bayesian_optimization/ground_truth.png" alt="Bayesian optimization">
</p>

Let's place the two plots side-by-side and talk about the results. In the **left image**, we see the ground-truth values of the loss function that we acquired by computing the value $$\ell(C, \gamma)$$ for every possible pair of $$(C, \gamma)$$ via a grid-search. You see the blue shaded region corresponding to low values for the loss function (good!) and the red stripe at the top corresponding to high values for the loss function (bad!). In the **right image**, we see the black points corresponding to our tried values. Do you notice how there is a high density of points near the blue shaded area where $$\ell(C,\gamma)$$ is minimized? That's **exploitation**! The BO algorithm picked up some good solutions into that area and sampled aggressively around that region. On the contrary, it tried some values near the top red stripe region, and since those trials yielded bad results, it didn't bother sampling any further there.

Ground-truth values             |  Bayesian Optimization
:--------------------------------------------------:|:-------------------------:
![]({{ site.url }}/images/bayesian_optimization/ground_truth.png)  |  ![]({{ site.url }}/images/bayesian_optimization/bayesian_optimization.png)

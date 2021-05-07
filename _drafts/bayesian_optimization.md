---
layout: post
title:  "Longest substring with non-repeating characters"
categories: [machine learning]
tags: [algorithms, programming]
description: An introduction to Bayesian optimization
---

### Introduction
Imagine that we are trapped in Dante’s optimization inferno. I.e., we are asked to optimize a function we don’t have an analytical expression for. It follows that we don’t have access to the first or second derivatives; hence using gradient descent or Newton’s method is a no-go. Also, we do not have any convexity guarantees about f(x). Therefore, methods from the convex optimization field are also not an option. The only thing we can do is to evaluate f(x) at some x. As if the situation was not bad enough, the function that we are trying to optimize is very costly to evaluate. Ergo, we can’t just go ahead and massively evaluate f(x) in, say, 100 billion random points and keep the one x that optimizes f(x) value.
The evaluation of the function might not be even purely computational. For example, evaluating the function may entail the conduction of some experiment in the lab requiring personnel, supplies, consumables, and waiting for hours or days for the experiment to complete. Another example is the maximization of a function f(lat, long) that gives the probability of finding oil if we drill on lat, long coordinates. Drilling costs lots of money, so we need to make good educated guesses unless we have an infinite amount of resources to spare. In other cases, f(x) might be the validation error of a neural network whose hyperparameters we would like to tune.
So, to sum up, we want to optimize $$f(x)$$ and:

1.	We don’t have a formula for $$f(x)$$
2.	We don’t have access to its derivatives $$f'(x)$$ and $$f''(x)$$
3.	We don’t have any convexity guarantees for $$f(x)$$
4.	$$f(x)$$ is expensive to evaluate for some $$x$4

To put it another way, we want to optimize an expensive, black-box, derivative-free function. And for this kind of problem, Bayesian Optimization (BO) is a very robust method.
Since we don’t have an expression for the objective function, the first step is to use a surrogate model to approximate $$f(x)$$. It is typical in this context to use Gaussian Processes (GPs), as we have already discussed in a previous blog post. It’s vital that you grasp the concept of GPs, and then BO will require almost no mental effort to sink. Once we have built a proxy model for $$f(x)$$, we want to decide which point $$x$$ to sample next. For this, we need an acquisition function, which kind of “reads” the GP and outputs the best guess $$x$$. So, in BO, there are two components: the *surrogate model*, which most often is a Gaussian Process modeling f(x), and the *acquisition function* that yields the next $$x$$ to evaluate. Having said that, a BO algorithm would look like this:

1.	Evaluate $$f(x)$$ at $$n$$ initial points
2.	While $$n \le N$$ repeat:
a.	Update the GP posterior using all available data
b.	Compute the acquisition function using the current GP
c.	Let $$x_n$$ be the maximizer of the acquisition function
d.	Evaluate $$y_n = f(x_n)$$
e.	Increment $$n$$
3.	Return either the $$x$$ evaluated with the largest $$f(x)$$, or the point with the largest posterior mean.

### Acquisition function
As we have already noted, the role of the acquisition function is to guide the next best point to sample f to find the global optimum. Acquisition functions are constructed so that a high value corresponds to potentially high values of the objective function. Either because the prediction is high or the uncertainty is high. So, acquisition functions favor regions that already correspond to optimal values or areas that haven’t been explored yet. This is known as the so-called exploration-exploitation trade-off.
There are three often cited acquisition functions: expected improvement (EI), maximum probability of improvement (MPI), and upper confidence bound (UCB). Although often cited last, I think it’s best to talk about UCB because it contains explicit exploitation and exploration terms.


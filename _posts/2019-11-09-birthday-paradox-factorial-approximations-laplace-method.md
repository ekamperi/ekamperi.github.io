---
layout: post
title:  "The birthday paradox, factorial approximation and Laplace's method"
date:   2019-11-09
categories: [mathematics]
tags: ['mathematics']
---

### The birthday paradox
So, I was looking at the [birthday paradox](https://en.wikipedia.org/wiki/Birthday_problem) and got a little carried away. Here's how.

In probability theory, the *birthday paradox* or *birthday problem* refers to the probability that, in a set of $$N$$ randomly chosen people, some pair of them will have the same birthday. This probability reaches $$50\%$$ with $$23$$ people. On the other hand $$99.9\%$$ probability is reached with just $$70$$ people. These numbers might seem counter-intuitive (too small).

In order to calculate the probability of a birthday collision, it's easier to start by considering the probability of drawing $$23$$ people successively, so that each one has a birthday *not yet seen*. This is the probability of *no collision*, so the probability of a collision is its complementary, i.e. $$1$$ minus this.

$$
\begin{align*}
\text{Prob}(\text{collision})
&= 1 - \text{Prob}(\text{No collision})\\
&= \underbrace{1 - \underbrace{\underbrace{\left(\frac{365}{365}\right)}_{\substack{\text{1st person has}\\\text{b-day not yet seen}}} \cdot \overbrace{\left(\frac{365-1}{365}\right)}^{\substack{\text{2nd person has}\\\text{b-day not yet seen}}} \cdot \left(\frac{365-2}{365}\right) \cdots \underbrace{\left(\frac{365-k+1}{365}\right)}_{\substack{k\text{-th person has}\\\text{b-day not yet seen}}}}_{\text{Probability of no collision}}}_{\text{Probability of collision}}
\end{align*}
$$

Since it is:

$$
\begin{align*}
&\left(\frac{365}{365}\right) \cdot \left(\frac{365-1}{365}\right) \cdot \left(\frac{365-2}{365}\right) \cdots \left(\frac{365-k+1}{365}\right)\\
&=\frac{365 \cdot 364 \cdots (365-k+1)}{365^k}\\
&=\frac{365! \cdot (365-k) \cdot (365-k-1) \cdots 1}{365^k}\\
&=\frac{365!}{(365-k)!{365^k}\\
&= \frac{365!/(365-k)!}{365^k}
\end{align*}$$
\begin{align*}
&\left(\frac{365}{365}\right) \cdot \left(\frac{365-1}{365}\right) \cdot \left(\frac{365-2}{365}\right) \cdots \left(\frac{365-k+1}{365}\right)\\
&=\frac{365 \cdot 364 \cdots (365-k+1)}{365^k}\\
&=\frac{365 \cdot 364 \cdots (365-k+1) \cdot (365-k) \cdot(365-k-1) \cdots 1}{365^k \cdot (365-k) \cdot(365-k-1) \cdots 1}\\
&= \frac{365!}{365^k (365-k)!}\\
&=\frac{365!/(365-k)!}{365^k}
\end{align*}
$$

It follows that in a set of $$N$$ randomly chosen items, the probability of some collision after $$k$$ "draws" is:

$$
\text{Prob(collision)} = 1 - \frac{N!/(N-k)!}{N^k}
$$

### Factorial $$N!$$ approximation and the Stirling's formula



### Laplace's method for asymptotic integrals
Our goal is to approximate the value of the following integral as $$\lambda \to \infty$$:

$$
I(\lambda) = \int_a^b f(x) e^{-\lambda \varphi(x)} \mathrm{d}x
$$

Our assumptions include the convergence of $$I(\lambda)$$ for $$\lambda$$ sufficiently
large, that $$f(x)$$ and $$\varphi(x)$$ are smooth enough to be replaced by their local Taylor expansions of appropriate degree. Also $$\varphi'(x_0) = 0, \varphi''(x_0)>0, f(x_0)\ne0$$.

We linearize $$f(x)$$ and $$\varphi(x)$$ we expand it into a second order Taylor series around the point $$x=x_0$$:

$$
\begin{align*}
f(x) \simeq f(x_0), \qquad \varphi(x) &\simeq \varphi(x_0) + \underbrace{\varphi'(x_0) (x-x_0)}_{\text{zero, since } \varphi'(x_0)=0}  + \frac{1}{2}\varphi''(x_0) (x-x_0)^2 + \ldots\\
&= \varphi(x_0) + \frac{1}{2}\varphi''(x_0) (x-x_0)^2 + \ldots
\end{align*}
$$

Therefore:

$$
\begin{align*}
I(\lambda) &\simeq f(x_0) \int_a^b \exp\left[ -\lambda \left(\varphi(x_0) + \frac{1}{2}\varphi''(x_0)(x-x_0)^2\right)\right] \mathrm{d}x\\
&= f(x_0) e^{-\lambda \varphi(x_0)} \int_a^b e^{-\frac{\lambda}{2}\varphi''(x_0)(x-x_0)^2} \mathrm{d}x\\
&= f(x_0) e^{-\lambda \varphi(x_0)}\int_a^b e^{-\frac{\lambda}{2}\varphi''(x_0)s^2} \mathrm{d}s
\end{align*}
$$

Recall that for Gaussian integral it is:

$$
\int_{-\infty}^{+\infty} e^{-a x^2} \mathrm{d}x = \sqrt{\frac{\pi}{a}}
$$

Here it is $$\frac{\lambda}{2}\varphi''(x_0) = a$$, therefore:

$$
\begin{align*}
I(\lambda) &\simeq f(x_0) e^{-\lambda \varphi(x_0)} \sqrt{\frac{\pi}{\frac{\lambda}{2}\varphi''(x_0)}}\\
&= f(x_0) e^{-\lambda \varphi(x_0)} \sqrt{\frac{2\pi}{\lambda \varphi''(x_0)}} 
\end{align*}
$$

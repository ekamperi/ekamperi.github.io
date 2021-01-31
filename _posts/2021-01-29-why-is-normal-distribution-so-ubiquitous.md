In 1948 Claude Shannon laid out the foundations of information theory. The need this new theory was supposed to meet was the effective and reliable transmission of messages. Although the motive was applied, information theory is deeply mathematical in its nature. A central concept in it is *entropy*, which is used somewhat differently than in thermodynamics. Consider a random variable $$X$$, which assumes the discrete values $$X = {x_i \mid i=-K, \ldots, 0, \ldots, +K}$$. Of course, $$0 \le p_i \le 1$$ and $$\sum_{i=-K}^{K} p_i = 1$$ must also be met.

Suppose now the extreme case that the value $$X=x_i$$ has a probability of occurring $$p_i=1$$, and $$p_{j\ne i}=0$$. In this scenario, there's no 
"surprise" by knowing the value of $$X$$, and there's no message being transmitted. It is as if I told you that it's chilly today in Alaska or that sun raised at east. In this context, we define the information content that we gain by observing $$X$$ as the following function:

$$
I(x_i) = \log\left(\frac{1}{p_i}\right) = -\log p_i
$$

We define as *entropy* as the expected value of $$I(x_i)$$ over all $$2K+1$$ discrete values $$X$$ takes:

$$
H(X) = \mathop{\mathbb{E}} \left(I(x_i)\right) = \sum_{i=-K}^K p_i I(x_i) = \sum_{i=-K}^{K} p_i \log\frac{1}{p_i} = -\sum{i=-K}^{K} p_i \log p_i
$$

Similarly, we define *differential entropy* for continuous variables:

$$
h(X) = -\int_{-\infty}^{\infty} p_X(x) \log p_X(x) \mathrm{d}x
$$

Alright, but what normal distribution has anything to do with these? It turns out that normal distribution is the distribution that maximizes information entropy under the constraint of fixed mean $$\m$$ and standard deviation $$s^2$$ of a random variable $$XX$$.

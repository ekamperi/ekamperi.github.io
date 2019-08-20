---
layout: post
title:  "Energy considerations for training deep neural networks"
date:   2019-08-14
categories: [machine learning]
tags: ['environment', 'machine learning']
---

The latest advances, both in hardware and theory of training neural networks, have enabled researchers to train very deep models on voluminous data (e.g. the Wikipedia corpus). The two main categories include networks that perform image recognition/classification and those that perform [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) tasks. Training such networks and achieving a high accuracy requires unusually large computational resources. As a result, these models are costly to train, fine-tune and deploy, both *financially* (due to the cost of purchasing hardware and paying electricity bills or renting cloud compute time) and *environmentally*, due to the carbon dioxide emissions required to run modern hardware.

<p align="center">
 <img style="width: 75%; height: 75%" src="{{ site.url }}/images/compute_diagram.png">
</p>
Image taken [from here](https://openai.com/blog/ai-and-compute/).

We will give a couple of examples on the depth of contemporary neural networks (DNNs). The *BERT* is a new language representation model which stands for "Bidirectional Encoder Representations from Transformers" (Devlin et al, 2018). In its base form BERT has 110M parameters and its training on 16 [TPU](https://en.wikipedia.org/wiki/Tensor_processing_unit) chips takes 4 days (96 hours). Another DNN from Radform et al (2019) has 1542M parameters, 48 layers and it needs 1 week (168 hours) to train on 32 TPUv3 chips.

To put things in perspective, the [carbon footprint](https://en.wikipedia.org/wiki/Carbon_footprint) of training BERT on GPU is comparable to that of a trans-American flight (Strubell et al, 2019)! But that's nothing compared to the hyperparameter optimization of the "transformer big" model from So et al (2019), where an evaluation of *15K child models* was performed, requiring a total of 979M train steps! This means that they trained 15K candidate models to decide which one was the most promising to train it further.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/footprint_comparison.png">
</p>

As mind blowing as these *relative* data are, the absolute environmental impact of training DNNs is small at the moment. The computational (hence environmental) cost of deploying the model for *inference* (i.e., using it for the intended purpose in the real world *after* you have trained it) is bigger.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/nvidia_inference.png">
</p>
Image taken from a techical overview titled "NVIDIA AI Inference Platform".

Real-time applications may involve multiple inferences per instance. For example, a spoken question may as well go through an automatic speech recognition, speech to text, natural language processing, a recommender system, text to speech and then speech synthesis. Each of these steps is a distinct inference operation on its own. Inference throughtput and latency are both important.

Things to pursuit in the future:
* Research on hardware that requires less energy
* Efficient training algorithms
* Efficient techniques to perform hyperparameter optimization (e.g. Bayesian search vs. grid/random search)
* Neural network compression techniques (for an overview [this is an excellent introduction]({{ site.url }}/docs/dnn_compression.pdf)) for speeding up inference
* Energy profiling tools

#### References
1. Devlin J, Chang M-W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [Internet]. arXiv [cs.CL]. 2018. Available from: [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805)
2. Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. Language models are unsupervised multitask learners. OpenAI Blog [Internet]. 2019;1(8). Available from: [https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. So DR, Liang C, Le QV. The Evolved Transformer [Internet]. arXiv [cs.LG]. 2019. Available from: http://arxiv.org/abs/1901.11117
4. Strubell E, Ganesh A, McCallum A. Energy and Policy Considerations for Deep Learning in NLP [Internet]. arXiv [cs.CL]. 2019. Available from: [http://arxiv.org/abs/1906.02243](http://arxiv.org/abs/1906.02243)

---
layout: post
title:  "Energy considerations for training deep neural networks"
date:   2019-08-14
categories: [machine learning]
tags: ['environment', 'machine learning']
---

The advances during recent years both in hardware and theory of training neural networks have enabled researchers to train very deep networks on voluminous data. The two main categories include networks that perform image recognition/classification and those that perform natural language processing tasks. Training such networks and achieving a high accuracy requires unusually large computational resources. As a result, these models are costly to train and fine-tune, both *financially* (due to the cost of purchasing hardware and paying electricity bills or renting cloud compute time), and *environmentally*, due to the carbon dioxide emissions required to run modern hardware.

We will give a couple of examples on the depth of contemporary deep neural networks (DNN). The *BERT* is a new language representation model which stands for "Bidirectional Encoder Representations from Transformers" (Devlin et al, 2018). In its base form BERT has 110M parameters and its training on 16 TPU chips takes 4 days (96 hours). Another DNN from Radform et al (2019) has 1542M parameters, 48 layers and it needs 1 week (168 hours) to train on 32 TPUv3 chips.

To put things in perspective, the carbon footprint of training BERT on GPU is comparable to that of a trans-American flight (Strubell et al, 2019).

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/footprint_comparison.png">
</p>

Things to pursuit in the future:
* Research on hardware that requires less energy
* Efficient training algorithms
* Efficient techniques to perform hyperparameter optimization (e.g. Bayesian search vs. grid/random search)
* Neural network compression techniques (for an overview [this is an excellent introduction]({{ site.url }}/docs/dnn_compression.pdf))
* Energy profiling tools

#### References
1. Devlin J, Chang M-W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [Internet]. arXiv [cs.CL]. 2018. Available from: [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805)
2. Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. Language models are unsupervised multitask learners. OpenAI Blog [Internet]. 2019;1(8). Available from: [https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. Strubell E, Ganesh A, McCallum A. Energy and Policy Considerations for Deep Learning in NLP [Internet]. arXiv [cs.CL]. 2019. Available from: [http://arxiv.org/abs/1906.02243](http://arxiv.org/abs/1906.02243)

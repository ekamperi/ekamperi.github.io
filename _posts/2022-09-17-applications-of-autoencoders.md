---
layout: post
title: "Applications of autoencoders"
date:   2022-09-17
categories: [mathematics]
tags: ['machine learning', 'mathematics', 'statistics']
description: A high-level summary of autoencoders' applications
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
Hello, world! It's been nine months since my last post! I was so engaged working at Chronicles Health that I couldn't find time to reserve for blogging. However, the previous week was my last one there. Now I'll wear my medical hat again and work as a [radiation oncology consultant](https://en.wikipedia.org/wiki/Radiation_therapy), hopefully enjoying a more predictable work schedule. I will probably write a blog post about my venture working on a startup. But for now, all I wanted was to make a soft comeback by writing a short post on **the applications of autoencoders**, one of my favorite machine learning topics.

In the future, I expect to find time to expand on these topics via separate posts, with in-depth analysis and coding examples.

## Applications of autoencoders
### Dimensionality reduction
[We have already used autoencoders as a dimensionality reduction technique before](https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html), and judging from Google Analytics, this post has been quite a success! So, the idea here is to compress the input
by learning some efficient low-dimensional data representation. To the extent that we accomplish that, we can then replace the original input $$x$$ with the new $$x_\text{latent}$$,
just like we can replace $$x$$ with the first couple of principal components when doing PCA.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/autoencoder/autoencoder_schematic.png" alt="Schematic representation of an autoencoder">
</p>

As it turns out, though, there are quite a few more applications that we will present here briefly.

### Feature extraction
To the uninitiated, feature extraction is the process where we take some data and then transform it so that the new variables are more informative and less redundant than the original ones. Also, the new derived values (features), hopefully, can differentiate different classes of things in a classification task or predict some target value in a regression task. This application is tightly related to dimensionality reduction. Here's how we do it. We take raw (unlabelled) data and train an autoencoder with it to force the model learn efficient data representations (the so-called latent space). Once we have trained the autoencoder network, we then **ignore the decoder part of the model**. Instead, we use only the encoder to convert new raw input data into the latent space representation. This new representation can then be used for supervised learning tasks. So, instead of training a supervised model to learn how to map $$x$$ to $$y$$, we ask it to map $$x_\text{latent}$$ to $$y$$.

### Object search
Again, this application is connected to the previous one. Say we'd like to build a search engine for images or songs. We could save all the items in a database and then go through each one, comparing it with our target. But that would be very time-consuming. Instead, we could run the entire thing in the latent space. Concretely, we would first pass all the known images (or songs) from an autoencoder and save their latent space representation (which, by definition, is low dimensional and doesn't take much space!) in a database. Then, given an image (or song) to search for, we would convert it into a latent space representation, and *then* we would search the database for it. The rationale is that operating on low-dimensional latent space is much more economical than high-dimensional original space.

### Denoising
Autoencoders can be trained in such a way that they learn how to perform efficient denoising of the source. Contrary to conventional denoising techniques, they do not actively look for noise in the data. Instead, they extract the source from the noisy input by learning a representation of it. The representation is subsequently used to decompress the input into noise-free data. A concrete example is training an autoencoder to remove noise from images. The key to
accomplishing this is to take the training images, *add some noise* to them, and use them as the $$x$$. Then use the original images (without the noise) as the $$y$$. So, to put it a bit more formally, we are asking the network to learn the mapping $$f:(x+\text{noise}) \to x$$. The following figure is taken from Keras's documentation on autoencoders. The upper row consists of the original untainted images (the $$y$$), and the lower row contains images with some noise added by us (the $$x$$).

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/autoencoder/noisy_digits.png" alt="Noisy digits for training a denoising autoencoder">
</p>

### Anomaly detection
Since autoencoders are trained to reconstruct their input as well as they can, naturally, if they are given an *out of distribution* example, the reconstruction will not be as good as if this example was *from the training distribution*. So, by using some proper threshold for the reconstruction loss, one can build an anomaly detector: any outlier $$x$$ will be reconstructed as $$x'$$, where $$\left|x' - x\right| \gt \text{thresh}$$.

### Synthetic data generation
Variational autoencoders can be used to generate new synthetic data, mostly image, but also time-series. The way to do this is by first training an autoencoder with some data, and then *randomly sampling the latent dimension* of the autoencoder. These random samples are then handed over to the decoder part of the network, leading to new data generation. The following image shows the results of sampling an autoencoder trained on the MNIST dataset. These digits do not exist in the training dataset; they are *generated* by the network.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/autoencoder/latent_sample.png" alt="Image generation with variation autoencoders">
</p>


### Data imputation
This is similar to the previous application. The idea here is to take a dataset *without* any missing entries and randomly *delete* some of the rows for some of the columns, pretending they are missing. However, we know the ground truth values and train the autoencoder to output those. Once trained, we can present a *really missing* entry to the network, and assuming that it has been trained robustly, it should perform efficient imputation. Again, to be a bit more concrete, given a dataset with $$x$$ values *without* any missing data, we artificially remove some values and then train an autoencoder to learn the mapping $$x_{\text{missing}} \to x$$.

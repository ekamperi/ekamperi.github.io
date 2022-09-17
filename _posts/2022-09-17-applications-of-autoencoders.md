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
Hello, world! It's been nine months since my last post! I was so engaged working at Chronicles Health that I couldn't find time to reserve for blogging. However, the previous week was my last one there. Now I'll wear my medical hat again and work as a [radiation oncology consultant](https://en.wikipedia.org/wiki/Radiation_therapy), hopefully enjoying a more predictable work schedule. I will probably write a blog post about my venture working on a startup. But for now, I wanted to make a soft comeback by writing a short blog post on **the applications of autoencoders**, one of my favorite machine learning topics.

In the future, I expect to find time to expand on these topics via separate posts, with in-depth analysis and coding examples.

## Applications of autoencoders
### Dimensionality reduction
[We have already used autoencoders as a dimensionality reduction technique before](https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html), and judging from Google Analytics, this post has been quite a success! So, the idea here is to compress the input,
by learning some efficient low-dimensional representation. If we do that, we can replace the original input $x$ with the new $x_\text{latent}$.

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/autoencoder/autoencoder_schematic.png" alt="Schematic representation of an autoencoder">
</p>

As it turns out, though, there are quite a few more applications that we will present here briefly.

### Feature extraction
This application is tightly related to the previous one. Here's how we do it. We take raw (unlabelled) data and train an autoencoder with it to force it to learn efficient data representations. So, we train an autoencoder network and then **ignore the decoder part of the model**. Instead, we use only the encoder to convert new raw input data into the latent space representation. This new representation can then be used for supervised learning tasks.

### Object search
Again, this application is related to the previous one. Say we'd like to build a search engine for images or songs. Instead of comparing the given image (or song), we could run the entire thing in the latent space. Concretely, we would first pass all the known images (or songs) from the autoencoder and save their latent space representation (which is low dimensional and doesn't take much space!) in a database. Then, given an image (or song) to search for, we would convert it into a latent space representation, and *then* we would search the database for it. The rationale is that operating on the low-dimensional latent space is much faster than the high-dimensional original space.

### Denoising
Autoencoders can be trained in such a way that they learn how to perform efficient denoising of the source. Contrary to conventional denoising techniques, they do not actively look for noise in the data. Instead, they extract the source from the noisy input by learning a representation of it. The representation is subsequently used to decompress the input into noise-free data. A concrete example is training an autoencoder to remove noise from images. The key to
accomplishing this is to take the training images, *add some noise* to them, and use them as the $x$. Then use the original images (without the noise) as the $y$. The following figure is taken from Keras's documentation on autoencoders. The upper row consists of the original untainted images (the $y$), and the lower row contains
images with some noise added by us (the $x$).

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/autoencoder/noisy_digits.png" alt="Noisy digits for training a denoising autoencoder">
</p>

### Anomaly detection
Since autoencoders are trained to reconstruct their input as well as they can, naturally, if they are given an *out of distribution* example, the reconstruction will not be as good as if this example was *from the training distribution*. So, by using some proper threshold for the reconstruction loss, one can build an anomaly detector: any outlier $x$ will be reconstructed as $x'$, where $\left|x' - x\right| \gt \text{thresh}$.

### Synthetic data generation
Variational Autoencoders can be used to generate both new synthetic data. The way to do this is by *randomly sampling the latent dimension* of the autoencoder. These random samples are then handed over to the decoder part of the network, leading to new data generation.

## Data imputation
This is similar to the previous application. The idea here is to take a dataset *without* any missing entries and randomly *delete* some of the rows for some of the columns, pretending they are missing. However, we know the ground truth values and train the autoencoder to output those. Once trained, we can present a *really missing* entry to the network, and assuming that it has been trained robustly, it should perform efficient imputation.

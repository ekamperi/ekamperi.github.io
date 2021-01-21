---
layout: post
title: "The encoder-decoder model as a dimensionality reduction technique"
date:   2021-01-21
categories: [machine learning]
tags: ['machine learning', Python, Tensorflow]
description: Introduction to the encoder-decoder model, also known as autoencoder, for dimensionality reduction
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
In today's post, we will discuss the encoder-decoder model, or simply [autoencoder (AE)](https://en.wikipedia.org/wiki/Autoencoder).  This will serve as a basis for implementing the more robust [variational autoencoder (VAE)](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)) in the following weeks. For starters, we will describe the model briefly and implement a dead simple encoder-decoder model in Tensorflow with Keras, in an absolutely indifferent to you dataset (my master thesis data). As a reward for enduring my esoteric narrative, we will then proceed to a more exciting dataset, the Fashion-MNIST, where we will show how the encoder-decoder model can be used for dimensionality reduction. To spice things up, we will construct a Keras callback to visualize the encoder's feature representation before each epoch. You will then see how the network builds up its hidden model progressively, epoch by epoch. Finally, we will compare AE to other standard methods, such as [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). Without further ado, let's get started!

An encoder-decoder network is an unsupervised artificial neural model that consists of an encoder component and a decoder one (duh!). Its goal is to learn a representation (read: encoding) for a set of data. The encoder takes the input and transforms it into a compressed encoding, handed over to the decoder. The decoder strives to reconstruct the original representation as close as possible. In a sense, we push the AE to memorize the training data by devising some mnemonic rule. As you see in the following figure, typically, the network has a bottleneck-like shape. It starts wide, then its connections are squeezed toward the middle, and then they fan out again. This architecture forces the AE to compress the training data's informational content and embed it into a low-dimensional space. By the way, you may encounter the term "latent space" for this intermediate data's representation.

<p align="center">
 <img style="width: 80%; height: 80%" src="{{ site.url }}/images/autoencoder/autoencoder_schematic.png" alt="Schematic representation of an autoencoder">
</p>

## Reconstructing outputs
### Preprocessing
First, we import the modules and functions we will be using:
{% highlight python %}
{% raw %}
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
{% endraw %}
{% endhighlight %}

Next, we load our dataset. Here, I'm using a dataset from [my master thesis](https://ekamperi.github.io/mrm_thesis/abstract.html) about comparing complexity indices in [volumetric modulated arc therapy](https://en.wikipedia.org/wiki/External_beam_radiotherapy#Volumetric_Modulated_Arc_Therapy) plans in prostate cancer. The details don't really matter; any high-dimensional data would do.

{% highlight python %}
{% raw %}
# Load database
pdf = pd.read_excel(r'/home/stathis/Jupyter_Notebooks/datasets/vmat_prostate_complexity.xlsx')

# Select only VMAT plans with radical intent (i.e., skip patients treated postoperatively
# in a salvage or adjuvant setting)
pdf = pdf[pdf['setting'] == 'radical']

# Calculate the combinatorial complexity index LTMCSV = LT * MCSV
pdf['ltmcsv'] = pdf['lt'] * pdf['mcsv']

# Select only the columns corresponding to the desired complexity metrics
metric_names = ['coa', 'em', 'esf', 'lt', 'ltmcsv', 'mcs', 'mcsv', 'mfa', 'pi', 'sas']
npdf = pdf[metric_names]

# Convert pandas dataframe to numpy array
x_train = npdf.to_numpy()

# Scale data to have zero mean and unit variance
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
{% endraw %}
{% endhighlight %}

### Building the autoencoder
Next, we build our autoencoder's architecture. We will squeeze our 10-dimensional space into a 2-dimensional latent or encoding space. Our choices are very rudimentary; the goal is to demonstrate how an encoder works, not build the optimal one.

{% highlight python %}
{% raw %}
# This is the dimension of the original space
intent_dim = 10

# This is the dimension of the latent space (encoding space)
latent_dim = 2

encoder = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(latent_dim, activation='relu')
])

decoder = Sequential([
    Dense(64, activation='relu', input_shape=(latent_dim,)),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(input_dim, activation=None)
])
{% endraw %}
{% endhighlight %}

Here comes the surgical part of the work. We stitch up the encoder and the decoder models into a single model, the autoencoder. The autoencoder's input is the input of the encoder, and the output of the autoencoder is the output of the decoder. The output of the decoder is the result of calling the decoder on the output of the encoder. We also set the loss to MSE.

{% highlight python %}
{% raw %}
autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
autoencoder.compile(loss='mse', optimizer='adam')
{% endraw %}
{% endhighlight %}

At this point, our autoencoder has not been trained yet. Let's feed it with some examples from the dataset and see how well it performs in reconstructing the input.

{% highlight python %}
{% raw %}
def plot_orig_vs_recon(title='', n_samples=3):
    fig = plt.figure(figsize=(10,6))
    plt.suptitle(title)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        idx = random.sample(range(x_train.shape[0]), 1)
        plt.plot(autoencoder.predict(x_train[idx]).squeeze(), label='reconstructed' if i == 0 else '')
        plt.plot(x_train[idx].squeeze(), label='original' if i == 0 else '')
        fig.axes[i].set_xticklabels(metric_names)
        plt.xticks(np.arange(0, 10, 1))
        plt.grid(True)
        if i == 0: plt.legend();

plot_orig_vs_recon('Before training the encoder-decoder')
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 90%; height: 90%" src="{{ site.url }}/images/autoencoder/orig_vs_recon_untrained.png" alt="Original vs. reconstructed values of an autoencoder">
</p>

### Training the autoencoder
Great! The autoencoder does not work at all! We then train the model and check the loss *vs.* epoch to make sure that it converged.

{% highlight python %}
{% raw %}
model_history = autoencoder.fit(x_train, x_train, epochs=5000, batch_size=32, verbose=0)

plt.plot(model_history.history["loss"])
plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/autoencoder/loss_vs_epoch.png" alt="Loss vs. echo of an autoencoder training">
</p>

Woot. The optimizer converged, and we can check again how well the autoencoder can reconstruct an input.

<p align="center">
 <img style="width: 90%; height: 90%" src="{{ site.url }}/images/autoencoder/orig_vs_recon_trained.png" alt="Original vs. reconstructed values of an autoencoder">
</p>

That's pretty damn good. The reconstructed values are very close to the original ones.

## A more interesting dataset
We now move forward to the Fashion MNIST dataset. This consists of a training set of 60.000 examples and a test set of 10.000 samples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Fashion MNIST has been proposed as a replacement for the original MNIST dataset with the handwritten digits for benchmarking machine learning algorithms.

We set up the autoencoder as before. Please keep in mind that whatever has to do with image classification uses convolutional neural networks of some sort. However, here we keep it simple and go with dense layers.

To visualize how the autoencoder builds up the latent space, we will create a custom callback by subclassing the `tf.keras.callbacks.Callback`. We will then override the method `on_epoch_begin(self, epoch, logs=None)`, which is called at the beginning of an epoch during training. There, we will hook up our code to extract the latent space representation. To obtain the output of an intermediate layer (in our case, we want to extract the encoder's output), we will retrieve it via `layer.output`. 

{% highlight python %}
{% raw %}
class TestEncoder(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        super(TestEncoder, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = self.current_epoch + 1
        encoder_model = Model(inputs=self.model.input,
                              outputs=self.model.get_layer('encoder_output').output)
        encoder_output = encoder_model(self.x_test)
        plt.subplot(4, 3, self.current_epoch)
        plt.scatter(encoder_output[:, 0],
                    encoder_output[:, 1],  s=20, alpha=0.8,
                    cmap='Set1', c=self.y_test[0:x_test.shape[0]])
        plt.xlim(-9, 9)
        plt.ylim(-9, 9)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
{% endraw %}
{% endhighlight %}

Off to train the model!

{% highlight python %}
{% raw %}
plt.figure(figsize=(15,15))
model_history = autoencoder.fit(x_train, x_train, epochs=12, batch_size=32, verbose=0,
                                callbacks=[TestEncoder(x_test[0:500], y_test[0:500])])
{% endraw %}
{% endhighlight %}

Here is the latent space evolution as the autoencoder is trained, starting with an untrained state at the top left and ending in a fully trained state at the bottom right.

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/autoencoder/latent_space1.png" alt="Evolution of latent space representation during the training of an autoencoder">
</p>

We also check the loss *vs.* epoch to make sure the optimizer converged. You may even find a correspondence between the classes' separation and how fast the loss drops during the training.

{% highlight python %}
{% raw %}
plt.plot(model_history.history["loss"])
plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)
{% endraw %}
{% endhighlight %}

<p align="center">
 <img style="width: 50%; height: 50%" src="{{ site.url }}/images/autoencoder/loss_vs_epoch_mnist.png" alt="Loss vs. epoch during the training of an autoencoder">
</p>

And here is another run:

<p align="center">
 <img style="width: 100%; height: 100%" src="{{ site.url }}/images/autoencoder/latent_space2.png" alt="Evolution of latent space representation during the training of an autoencoder">
</p>

## Autoencoder vs. Principal component analysis

As we've seen, both autoencoder and PCA may be used as dimensionality reduction techniques. However, there are some difference between the two:

1. By definition, PCA is a linear transformation, whereas AEs are capable of modeling complex non-linear functions. There is, however, kernel PCA that can model non-linear data.
2. In PCA, features are by definition linearly uncorrelated. Recall that they are projections onto an orthogonal basis. On the contrary, autoencoded features might be correlated. The two optimization objectives are simply different (an orthogonal basis that maximizes variance when data are projected onto it vs. maximum accuracy reconstruction).
3. PCA is computationally less demanding than autoencoders.
4. Autoencoders having many trainable parameters are vulnerable to overfitting, similar to other neural networks.

Regarding the question of which one to use, I'm afraid I'll sound cliche. It depends on the problem you are trying to solve. If your data share non-linear correlations, AE will compress them into a low-dimensional latent space since it is endowed with the capability to model non-linear functions. If your data are mostly linearly correlated, PCA will do fine. By the way, there's also a kernel version of PCA. Using a [kernel trick](https://en.wikipedia.org/wiki/Kernel_method), similar to the one with [Support Vector Machines](https://en.wikipedia.org/wiki/Support-vector_machine), the originally linear operations of PCA are performed in a reproducing kernel space. But that's the subject of a future post.

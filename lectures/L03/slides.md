---
title: MSAI 495
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 70%; position: absolute;">

  #  Generative AI
  ## L.03 | Autoencoders

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 30%">

  <iframe src="https://lottie.host/embed/e7eb235d-f490-4ce1-877a-99114b96ff60/OFTqzm1m09.json" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Generative AI.
  ## Please check in by entering the provided code.

  </div>
  </div>

  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Enter Code" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with Autoencoders / Variational Autoencoders?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Agenda

<div class = "col-wrapper" style="font-size: 0.8em;">
<div class="c1" style = "width: 40%">

### Introduction
- Why Autoencoders?
- Applications

### Autoencoders
- Definition
- Architecture
- Loss Functions
- Limitations of Autoencoders

</div>
<div class="c2" style = "width: 60%; margin: 0px; padding: 0px;">

### Variational Autoencoders
- Definition & Architecture
- Key Features of VAEs
- Applications
- Advantages Over Autoencoders

</div>
</div>



<!--s-->

## Autoencoders

Autoencoders are a type of neural network that learns to encode the input data into a lower-dimensional representation and then decode it back to the original data. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The autoencoder was introduced in 1986 by D. E. Rumelhart, G. E. Hinton, and R. J. Williams.

</div>
<div class="c2" style = "width: 30%;">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/ae-paper.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Rumelhart (1986)</p>
</div>

</div>
</div>

<!--s-->

## Autoencoders

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Autoencoders | Applications

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; padding: 10px;">

### Dimensionality Reduction 
Autoencoders can be used to reduce the dimensionality of data, similar to PCA, but with the added benefit of being able to learn non-linear transformations.

### Denoising
Autoencoders can be trained to remove noise from data, making them useful for tasks such as image denoising.

</div>
<div class="c2" style = "width: 50%; padding: 10px;">

### Anomaly Detection
Autoencoders can be used to detect anomalies in data by training on normal data and then identifying data points that have a high reconstruction error.

### **Generative Modeling**
Autoencoders can be used to generate new data by sampling from the latent space and decoding it back to the original data space.

</div>
</div>

<!--s-->

## Autoencoder Architecture | Nutshell

The autoencoder architecture consists of three main components: the encoder, the latent space ($z$), and the decoder. The **encoder** maps the input data to the **latent space**, which is a lower-dimensional representation of the data. The **decoder** maps the latent space back to the original data space. 

Sampling from $z$ and decoding it back to the original data space is how we generate new data.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## L.04 | Q.01

How do we generate new images from the latent space?

<div class = 'col-wrapper'>
<div class='c1 col-centered' style = 'width: 50%; align-items: left;'>

A. Decode($z$) <br><br>
B. Encode($z$) <br><br>
C. $z$

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.01' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

## Autoencoder Architecture | Encoder

An encoder is any model that takes the input data and maps it to any latent space. The encoder can be any type of neural network, but for this example and since we're working with images, we will use a simple convolutional neural network (CNN).

We want a latent space that is small enough to capture the important features of the data, but large enough to allow for some variability. For demonstration purposes, let's choose a latent space of 2 dimensions.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
IMAGE_SIZE = 32
CHANNELS = 1
EMBEDDING_DIM = 2

encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)
```

</div>
<div class="c2" style = "width: 50%">

<div style="text-align: center;">
  <img src="https://storage.googleapis.com/slide_assets/vae_lecture/encoder_summary.png" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Foster (2023)</p>
</div>


</div>
</div>

<!--s-->

## Autoencoder Architecture | Decoder

The decoder is a model that takes the latent space and maps it back to the original data space. The decoder can be any type of model, but again for this example, we will use a simple convolutional model (CNN). 

We want the decoder to be able to reconstruct the original data from the latent space. So we choose a decoder that is symmetric to the encoder. But how can we expand the latent space back to the original data space? 

<!--s-->

## Transpose Convolution

<span class="code-span">Transpose convolution</span> is a type of convolution that is used to upsample data. It is the reverse of a regular convolution, which downsamples data. Most deep learning frameworks have a built-in transpose convolution layer (e.g., <span class="code-span">Conv2DTranspose</span> in Keras, <span class="code-span">nn.ConvTranspose2d</span> in PyTorch).

<div style="text-align: center;">
<img src="https://d2l.ai/_images/trans_conv.svg" width="50%" style="border-radius: 10px;">
<p style="font-size: 0.6em; color: grey;">d2l 2025</p>
</div>

<!--s-->

## Autoencoder Architecture | Decoder

Using transpose convolution, we can build the decoder to **upsample** the latent space back to the original data space.

```python
CHANNELS = 1
EMBEDDING_DIM = 2

decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(CHANNELS,(3, 3),strides=1,activation="sigmoid",padding="same", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)
```

<!--s-->

## Autoencoder Architecture | Joining Encoder and Decoder

Now that we have the encoder and decoder, we can join them together to form the autoencoder.

```python
autoencoder = models.Model(encoder_input, decoder(encoder_output))
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Encoder

```python
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)
```

</div>
<div class="c2" style = "width: 50%">

### Decoder

```python
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(CHANNELS,(3, 3),strides=1,activation="sigmoid",padding="same", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)
```

</div>
</div>
```python
autoencoder = models.Model(encoder_input, decoder(encoder_output))
```
<!--s-->

## Autoencoder Training | Loss

Autoencoders can be trained using any reconstuction loss, but two common losses are **root mean squared error (RMSE)** and **binary cross-entropy (BCE)**.

<!--s-->

## Autoencoder Training | Loss | RMSE

RMSE is commonly used in evaluating the performance of image reconstruction models. It measures the average squared difference between the predicted and true pixel values.

<div style="text-align: center;">

$\text{RMSE}(x, \widehat{x}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \widehat{x}_i)^2} $

</div>

Where:
- $x_i$ is the true pixel value
- $\widehat{x}_i$ is the predicted pixel value
- $n$ is the number of pixels in the image


> RMSE will be symmetrically distributed around the average pixel values. An overestimation is penalized equally to an underestimation.

<!--s-->

## Autoencoder Training | Loss | BCE

BCE is commonly used in binary classification tasks, but it can also be used for image reconstruction tasks. It measures the difference between the predicted and true pixel values using a probabilistic approach.

$$ \text{BCE}(x, \widehat{x}) = -\frac{1}{n} \sum_{i=1}^{n} [x_i \log(\widehat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)] $$

Where:
- $x_i$ is the true pixel value
- $\widehat{x}_i$ is the predicted pixel value
- $n$ is the number of pixels in the image

BCE will be **asymmetrically distributed** around the average pixel values. Extremes are penalized more heavily than errors towards the center (0.5). 

For example, if the true pixel is high (0.7) then generating a pixel of 0.8 is penalized more than generating a pixel of 0.6. Similarly, if the true pixel is low (0.3) then generating a pixel of 0.2 is penalized more than generating a pixel of 0.4.


<!--s-->

## Autoencoder Training | Comparing Loss Functions

Choosing between RMSE and BCE depends on the application and the desired characteristics of the generated images.

Binary-cross entropy will produce slightly blurry images (as it pushes predictions towards 0.5), but RMSE can lead to pixelized images. Use the loss function that produces the best results for your application.

<!--s-->

## L.04 | Q.02

What loss function would you use for an autoencoder that generates oil paintings?

<div class = 'col-wrapper'>

<div class='c1 col-centered' style = 'width: 50%; align-items: left;'>

A. RMSE <br><br>
B. BCE <br><br>

</div>

<div class='c2' style = 'width: 50%;'>

<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.02' width = '100%' height = '100%'></iframe>

</div>
</div>

<!--s-->

## Autoencoder Training

Training your autoencoder is simple. We will use the Adam optimizer and binary cross-entropy loss.

```python
EPOCHS = 50
BATCH_SIZE = 128

autoencoder = models.Model(encoder_input, decoder(encoder_output))
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
history = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, x_test))
```

<!--s-->

## Autoencoders | Demonstration

The following slides include a simple example autoencoder that can generate images of fashion using the [Fashion-MNIST dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist). 

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/fashion_example.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Fasion MNIST</p>
</div>

<!--s-->

## Fashion-MNIST

Fashion-MNIST is a dataset of grayscale images of 10 different types of clothing items, including t-shirts, trousers, and shoes. Each image is 28x28 pixels in size. Given the popularity of Fashion-MNIST, tensorflow has a built-in function to load the dataset.

```python
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
```

<!--s-->

## Fashion-MNIST Preprocessing

Preprocessing the Fashion-MNIST dataset is simple. We will normalize the pixel values to be between 0 and 1 and pad the images to be 32x32 pixels.

> Why 32x32? 32 is a power of 2, which is a common size for images in deep learning and enables efficient computation on GPUs.

```python
def preprocess(imgs: np.ndarray) -> np.ndarray:
   """
   Normalize and reshape the images.

   Args:
      imgs (numpy.ndarray): Images to preprocess.
   Returns:
      imgs (numpy.ndarray): Preprocessed images.
   """
   imgs = imgs.astype("float32") / 255.0
   imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
   imgs = np.expand_dims(imgs, -1)
   return imgs
```

<!--s-->

## Reconstructing Images

Once the autoencoder is trained, we can use it to reconstruct images. We will use the encoder to encode the images and then use the decoder to decode the latent space back to the original data space. This is a simple test of our latent space ($z$).

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/fashion_autoencoder_example.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

<!--s-->

## Visualizing Latent Space

Once the autoencoder is trained, we can visualize the latent space by plotting the encoded images in a 2D scatter plot. This will allow us to see how the autoencoder has learned to represent the data in the latent space.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder_space.png' style='border-radius: 10px; width: 50%; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster (2023)</p>
</div>

<!--s-->

## Autoencoder Latent Space with Labels

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Fashion MNIST Labels

| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser    |
| 2     | Pullover   |
| 3     | Dress      |
| 4     | Coat       |
| 5     | Sandal     |
| 6     | Shirt      |
| 7     | Sneaker    |
| 8     | Bag        |
| 9     | Ankle boot |


</div>
<div class="c2" style = "width: 50%">

### Visualizing Latent Space with Labels

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder_colored_samples.png' style='border-radius: 10px; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster (2023)</p>
</div>

</div>
</div>

<!--s-->

## Generating Images through Sampling

Once the autoencoder is trained, we can generate new images by sampling from the latent space and using the decoder to decode the latent space back to the original data space.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

<!--s-->

## Latent Space Characteristics | Issues

Our simple autoencoder works great! But, we can see some issues with the latent space.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

- Clothing items do not cover similar area of the latent space.

- Distribution is not symmetric around (0, 0) or bounded. 

- Gaps in the latent space.

- No enforced continuity in the latent space.

This results in **inconsistent generation** of images and largely incomprehensible latent space.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder_sample_overlay.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

</div>
</div>

<!--s-->

## L.04 | Q.03

Look at this latent space and the range of the axes. If we enforce a normal distribution, what do you expect the range of the axes to approximately be?

<div class = 'col-wrapper'>
<div class='c1 col-centered' style = 'width: 50%; align-items: left;'>



A. (-1, 1) <br><br>
B. (-2, 5) <br><br>
C. (-3, 3) <br><br>
D. (-10, 15)

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder_sample_overlay.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>


</div>

<div class='c2' style = 'width: 50%;'>

<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.03' width = '100%' height = '100%'></iframe>

</div>
</div>


<!--s-->

<div class="header-slide">

# Variational Autoencoders

</div>

<!--s-->

## Variational Autoencoders | Overview

Variational autoencoders (VAEs) are a type of autoencoder that learns a probabilistic representation of the data in the latent space. VAEs are based on the idea of variational inference, which is a method for approximating complex probability distributions.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/ae_vs_vae.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

<!--s-->

## Variational Autoencoders | Encoder

In an autoencoder, each image is mapped to a single point in the latent space. In a VAE, each image is mapped to a distribution in the latent space. This allows for more variability in the generated images and helps to enforce continuity in the latent space.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder.png' style='border-radius: 10px; width: 80%;'>

   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

<!--s-->

## Normal Distribution

The normal distribution is a continuous probability distribution that is symmetric around the mean. It is defined by two parameters: the mean ($\mu$) and the standard deviation ($\sigma$).

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ \mathscr{N}(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} $$

We can sample a point $z$ from any normal distribution using the following equation:

$$ z = \mu + \sigma \cdot \epsilon $$

where $\epsilon$ is a random variable sampled from a standard normal distribution, defined by $\epsilon \sim \mathscr{N}(0, 1)$.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://www.scribbr.de/wp-content/uploads/2023/01/Standard-normal-distribution.webp'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

</div>
</div>

<!--s-->

## Multivariate Normal Distribution

The multivariate normal distribution is a generalization of the normal distribution to multiple dimensions. It is defined by a mean vector $\mu$ and a covariance matrix $\Sigma$.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ \mathscr{N}(x; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)} $$

Where:
- $k$ is the number of dimensions
- $x$ is a vector of random variables
- $\mu$ is the mean vector
- $\Sigma$ is the covariance matrix

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://upload.wikimedia.org/wikipedia/commons/8/8e/MultivariateNormal.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

</div>
</div>


<!--s-->

## Multivariate Normal Distribution

A multivariate normal distribution is fully defined by its mean vector and covariance matrix. A multivariate standard normal distribution is a multivariate normal distribution with mean vector 0 and covariance matrix $I$ 

$$ \mathscr{N}(0, I) $$.

<!--s-->

## Variational Autoencoders | Encoder

Our goal with the encoder is to learn the mean and variance of the multivariate normal distribution that best represents the data. However, variance values are always positive. So we predict the log of the variance which has range ($-\infty, \infty$).

$$ z_{mean}, z_{logvar} = encoder(x) $$

and since we want to sample from the distribution, we need to calculate the standard deviation of the distribution:

$$ z = z_{mean} + e^{z_{logvar} / 2} \cdot \epsilon $$

where: 

- $z_{sigma} = e^{z_{logvar} / 2}$
- $\epsilon \sim \mathscr{N}(0, I)$

<!--s-->

## Variational Autoencoders | Encoder

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>


<!--s-->

## Variational Autoencoders | Sampling Layer

The sampling layer is a custom layer that takes the mean and log variance of the distribution and samples from it using the reparameterization trick. Recall this equation: 

$$ z = z_{mean} + e^{z_{logvar} / 2} \cdot \epsilon $$

Here it is represented in Keras:

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

```



<!--s-->

## Reparameterization Trick

Sampling directly from a probability distribution is not differentiable, which prevents the use of backpropagation for training the encoder. The reparameterization trick addresses this issue by allowing for differentiable sampling from a probability distribution. This is achieved by reparameterizing a random variable as a function of a Gaussian distribution, effectively bypassing the non-differentiable sampling step.

By encapsulating all the randomness within $\epsilon$, the partial derivative of the layer output with respect to its input becomes deterministic and does not depend on $\epsilon$.

$$ z = \mu + \sigma \cdot \epsilon $$

Or since we're actually working with the log of the variance:

$$ z = \mu + e^{\frac{1}{2} \cdot \log(\sigma^2)} \cdot \epsilon $$

<!--s-->

## Variational Autoencoders | Encoder

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

```python
EMBEDDING_DIM = 2
IMAGE_SIZE = 32

encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
z_mean = layers.Dense(EMBEDDING_DIM, name="z_mean")(x)
z_log_var = layers.Dense(EMBEDDING_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
```

<!--s-->

## Variational Autoencoders | Decoder

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

```
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D( 1, (3, 3), strides=1, activation="sigmoid", padding="same", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)
```

<!--s-->

## Variational Autoencoders | Loss

In addition to the reconstruction loss, we also need to add a regularization term to the loss function that encourages the encoder to learn a multivariate normal distribution. 

This is done using the Kullback-Leibler divergence (KL divergence) between the learned distribution and the standard normal distribution. In our VAE, we want to measure how much our normal distribution with parameters $(\mu, \sigma)$ diverges from the standard normal distribution with parameters $(0, I)$.

$$ D_{KL}[\mathscr{N}(\mu, \sigma) || \mathscr{N}(0, I)] = - \frac{1}{2} \sum_{i=1}^{k} (1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2) $$

Where: 
- **log(σ²ᵢ)**: The natural logarithm of the variance of the i-th dimension of $\mathscr{N}(\mu, \sigma)$.
- **μ²ᵢ**: The square of the mean of the i-th dimension of $\mathscr{N}(\mu, \sigma)$.
- **σ²ᵢ**: The variance of the i-th dimension of $\mathscr{N}(\mu, \sigma)$.

The sum is taken over all dimensions $k$ in the latent space $z$. Loss is minimized when $z_{mean}$ is 0, $z_{logvar}$ is 0. 

<!--s-->

## Variational Autoencoders | Loss

Here is what KL divergence looks like in TensorFlow / Keras.

```python
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
```

<!--s-->

## Variational Autoencoders | Benefits of KL Divergence

The KL divergence term encourages the encoder to learn a multivariate normal distribution, which helps to enforce continuity in the latent space. The KL divergence term tries to force all encoded distributions to be similar to the standard normal distribution. This helps to ensure that the latent space is well-structured, symmetric, and continuous (so less large gaps).

In the original VAE paper, the loss function was simply: 

$$ \text{Loss} = \text{Reconstruction Loss} + \text{KL Divergence} $$

But a variance on this $\beta$-VAE has become popular. The $\beta$-VAE loss function is:
$$ \text{Loss} = \text{Reconstruction Loss} + \beta \cdot \text{KL Divergence} $$

where $\beta$ is a hyperparameter that controls the trade-off between the reconstruction loss and the KL divergence. A larger value of $\beta$ will encourage the encoder to learn a more structured latent space, but may result in poorer reconstruction quality.

<!--s-->

## Variational Autoencoder

Here is an outline for a VAE in TensorFlow / Keras:

```python

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [ self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker, ]

    def call(self, inputs):
        """Call the model on a particular input."""
        z_mean, z_log_var, z = encoder(inputs)
        reconstruction = decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        """Step run during training."""

        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean( BETA * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3)))
            kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Step run during validation."""
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(BETA * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3)))
        kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
        total_loss = reconstruction_loss + kl_loss

        return { "loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss, }
```
<p style='font-size: 0.6em; color: grey; text-align: center;'>Foster (2023)</p>

<!--s-->

## Gradient Tape

TensorFlow's <span class="code-span">tf.GradientTape</span> is a context manager that records operations for automatic differentiation. It allows you to compute gradients of a computation with respect to some inputs, which is essential for training custom layers.

<div style='text-align: center;'>
   <img src='https://debuggercafe.com/wp-content/uploads/2021/07/tensorflow-gradienttape.jpg' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey;'>debuggercafe 2021</p>
</div>

<!--s-->

## Variational Autoencoders | Training

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster (2023)</p>
</div>

```python
vae = VAE(encoder, decoder)
optimizer = optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer)
history = vae.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, x_test))
```

<!--s-->

## Analysis of VAE

There are significant differences between the VAE and a simple autoencoder.

<div class = "col-wrapper">
<div class="c1" style = "width: 40%">

### KL Divergence

KL divergence term ensures that latent space never strays too far from the standard normal distribution.

### Continuity

The space is much more continuous, so we can sample from the latent space and get a more consistent generation of images.


</div>
<div class="c2 col-centered" style = "width: 60%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae_samples.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

</div>
</div>


<!--s-->

## Variational Autoencoder Latent Space with Labels

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Fashion MNIST Labels

| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser    |
| 2     | Pullover   |
| 3     | Dress      |
| 4     | Coat       |
| 5     | Sandal     |
| 6     | Shirt      |
| 7     | Sneaker    |
| 8     | Bag        |
| 9     | Ankle boot |


</div>
<div class="c2" style = "width: 50%">

### Visualizing Latent Space with Labels

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae_colored_samples.png' style='border-radius: 10px; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster (2023)</p>
</div>

</div>
</div>

<!--s-->

## Autoencoder Latent Space with Labels

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Fashion MNIST Labels

| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser    |
| 2     | Pullover   |
| 3     | Dress      |
| 4     | Coat       |
| 5     | Sandal     |
| 6     | Shirt      |
| 7     | Sneaker    |
| 8     | Bag        |
| 9     | Ankle boot |


</div>
<div class="c2" style = "width: 50%">

### Visualizing Latent Space with Labels

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/autoencoder_colored_samples.png' style='border-radius: 10px; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster (2023)</p>
</div>

</div>
</div>

<!--s-->

## CelebA

The Fashion-MNIST dataset with 2 latent dimensions is a simple example. Let's now move our attention to a more complex dataset: CelebA. CelebA is a large-scale face dataset with over 200,000 celebrity images with various labels (e.g. glasses, pointy nose, etc). 

The following slides will demonstrate the results of a VAE trained on the CelebA dataset.

<div style='text-align: center;'>
   <img src='https://mmlab.ie.cuhk.edu.hk/projects/CelebA/overview.png' style='border-radius: 10px; width: 50%; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>CelebA</p>
</div>

<!--s-->

## Analysis of CelebA VAE

Training the VAE and sampling from the latent space using the same process as before, we can test how well we preserve facial features.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/encoded-decoded-vae.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Analysis of CelebA VAE

Here we have a histogram for each dimension of the latent space. The latent space is much more structured, thanks to the KL divergence term.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/distributions.png' style='border-radius: 10px; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster 2023</p>
</div>


<!--s-->

## Generating New Faces

We can sample from the latent space to generate new faces. Here are some randomly generated faces.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/vae_face_generated.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Latent Space Arithmetic

One of the cool things about VAEs is that you can perform arithmetic in the latent space.

Consider the power of this for a moment. You can find some latent vector for increasing "smile", then adding this vector to a latent vector for a "neutral" face will result in a "smiling" face. That's exactly what we'll do next. 

<!--s-->

## Latent Space Arithmetic | Finding the Smile Vector

To find the "smile" vector, we can sample a large number of faces from the dataset and calculate the average difference between the latent vectors of smiling faces and neutral faces.

$$ \text{smile\_vector} = \text{mean}(\text{latent\_vectors\_smile}) - \text{mean}(\text{latent\_vectors\_neutral}) $$

<!--s-->

## Latent Space Arithmetic | Formal Approach

Let's formalize our morphing process. We can add a feature to a latent vector by adding a scaled version of the feature vector to the latent vector.

$$ z_{new} = z + \alpha \cdot \text{feature_vector} $$

where:
 
- $z$ is the latent vector starting point (e.g., neutral face)
- $\alpha$ is a scalar that controls the intensity of the feature
- $\text{feature_vector}$ is the vector that represents the feature (e.g., smile, glasses, etc.)

<!--s-->

## Latent Space Arithmetic | Examples

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/latent_space_arithmetic.png' style='border-radius: 10px;margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster 2023</p>
</div>

<!--s-->

## Latent Space Arithmetic | Morphing

Since in a VAE the latent space is continuous, you can morph between two faces by linearly interpolating between their latent vectors.

$$ z_{new} = (1 - \alpha) \cdot z_1 + \alpha \cdot z_2 $$

where: 
  - $z_1$ is the latent vector of the first face
  - $z_2$ is the latent vector of the second face
  - $\alpha$ is a number between 0 and 1 that controls the interpolation
  - $z_{new}$ is the interpolated latent vector

So in effect, we take two images, encode them into the latent space, then decode points along the straight line between them at regular intervals.

<!--s-->

## Latent Space Arithmetic | Morphing Example

$$ z_{new} = (1 - \alpha) \cdot z_1 + \alpha \cdot z_2 $$

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/vae_lecture/face_morphing.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

<div class="header-slide">

# Flowers Demo

</div>

<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

### Autoencoders
Autoencoders are a type of neural network that learns to encode and decode data.

### Variational Autoencoders
Variational autoencoders (VAEs) are a type of autoencoder that learns a probabilistic representation of the data in the latent space.

Demonstrations for both Fashion-MNIST (simple) and CelebA (complex) datasets were shown.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with Autoencoders / Variational Autoencoders?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Project Time

</div>

<!--s-->
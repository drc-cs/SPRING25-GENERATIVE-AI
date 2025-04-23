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
  ## L.04 | Generative Adversarial Networks

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
  ## On a scale of 1-5, how confident are you with the following topics: 

- Generative Adversatial Networks
- Wasserstein GANs
- Conditional GANs

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Generative Adversarial Networks

</div>

<!--s-->

## Introduction to GANs

Generative Adversarial Networks (GANs) were introduced in 2014 by Ian Goodfellow and his colleagues. GANs work by having two neural networks, a generator and a discriminator, compete against each other. 

The generator creates fake data, while the discriminator tries to distinguish between real and fake data. Over time, the generator learns to create more realistic data.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_paper.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

<!--s-->

## GAN Architecture

GANs consist of two main components: the **generator** and the **discriminator**. The generator creates fake data, while the discriminator tries to distinguish between real and fake data. The two networks are trained simultaneously in a game-like setting, where the generator tries to fool the discriminator, and the discriminator tries to correctly classify the data.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_overview.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Source: Ian Goodfellow</p>
</div>

<!--s-->

## Example Dataset | LEGO Bricks

The [LEGO Bricks dataset](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images) is a collection of images of LEGO bricks. The dataset contains over 40,000 images of 50 different LEGO bricks taken at different angles. 

Our goal with the GAN is to generate new images of LEGO bricks that look like they were taken from the same dataset.


<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/scene.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Hazelzet (2020)</p>
</div>

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/3001R.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Hazelzet (2020)</p>
</div>

</div>
</div>

<!--s-->

## Lego Bricks Preprocessing

We preprocess the images by resizing them to 64x64 pixels and normalizing the pixel values to be between **-1 and 1**.

```python

import tensorflow as tf

# Load the dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "/app/data/lego-brick-images/dataset/",
    labels=None,
    color_mode="grayscale",
    image_size=(64, 64),
    batch_size=512,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

# Normalize the pixel values
train_data = train_data.map(lambda x: (x - 127.5) / 127.5)
```

<!--s-->

## Why between -1 and 1?

The reason we normalize the pixel values to be between -1 and 1 is because we will be using a **tanh** activation function in the generator. The tanh function outputs values between -1 and 1, so we need to normalize the pixel values to be in the same range. 

tanh tends to provide stronger gradients than sigmoid, especially for values close to 0.

<!--s-->

## GAN Architecture

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_overview.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Source: Ian Goodfellow</p>
</div>

<!--s-->

## Discriminator

The discriminator is a convolutional neural network (CNN) that takes an image as input and outputs a single value between 0 and 1. The output represents the probability that the input image is real. The discriminator is trained to correctly classify real and fake images.

```python
from tensorflow.keras import layers, models

discriminator_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(discriminator_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D( 1, kernel_size=4, strides=1, padding="valid", use_bias=False, activation="sigmoid")(x)
discriminator_output = layers.Flatten()(x)
discriminator = models.Model(discriminator_input, discriminator_output)
```

The shape of the tensor will decrease by a factor of 2 in each dimension after each Conv2D layer (64, 32, 16, 8, 4) and the number of filters will increase by a factor of 2 (64, 128, 256, 512).

<!--s-->

## Generator

The generator is a convolutional neural network (CNN) that takes a random noise vector as input and outputs an image. Similar to the VAE, the generator will use a vector drawn from a multivariate normal distribution as input. 

```python
generator_input = layers.Input(shape=(Z_DIM,))
x = layers.Reshape((1, 1, Z_DIM))(generator_input)
x = layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding="valid", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose( CHANNELS, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh")(x)
generator = models.Model(generator_input, generator_output)
```

The shape of the tensor will increase by a factor of 2 in each dimension after each Conv2DTranspose layer (4, 8, 16, 32, 64) and the number of filters will decrease by a factor of 2 (512, 256, 128, 64, 1).

<!--s-->

## UpSampling2D + Conv2D vs Conv2DTranspose

The main difference between <span class='code-span'>UpSampling2D</span> + <span class='code-span'>Conv2D</span> and <span class='code-span'>Conv2DTranspose</span> is that <span class='code-span'>UpSampling2D</span> + <span class='code-span'>Conv2D</span> first upsamples the input tensor and then applies a convolution, while <span class='code-span'>Conv2DTranspose</span> applies a transposed convolution directly to the input tensor.

<span class='code-span'>Conv2DTranspose</span> can lead to artifacts in the form of small checkerboard patterns in the generated images.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/checkered.png' style='border-radius: 10px;margin: 0px;'>
   <p style='font-size: 0.6em; color: grey;margin: 0px;'>Odena 2016</p>
</div>

<!--s-->

## GAN Training

The GAN is trained using the following steps:

1. Generate fake images using the generator.
2. Train the discriminator on real and fake images.
3. Train the generator to fool the discriminator.

It is **critical** to train the discriminator and generator in an alternating fashion. We want the generated images to be predicted close to 1 because the generator is strong, not because the discriminator is weak.

<!--s-->

## GAN Training

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_training.png' style='border-radius: 10px; width: 70%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## GAN Training

<div style="height: 100%">

```python
IMAGE_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 300
LOAD_MODEL = False
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002
NOISE_PARAM = 0.1

class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.d_real_acc_metric = metrics.BinaryAccuracy(name="d_real_acc")
        self.d_fake_acc_metric = metrics.BinaryAccuracy(name="d_fake_acc")
        self.d_acc_metric = metrics.BinaryAccuracy(name="d_acc")
        self.g_loss_metric = metrics.Mean(name="g_loss")
        self.g_acc_metric = metrics.BinaryAccuracy(name="g_acc")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_real_acc_metric,
            self.d_fake_acc_metric,
            self.d_acc_metric,
            self.g_loss_metric,
            self.g_acc_metric,
        ]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the discriminator on fake images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(generated_images, training=True)

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + NOISE_PARAM * tf.random.uniform(tf.shape(real_predictions))
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - NOISE_PARAM * tf.random.uniform(tf.shape(fake_predictions))

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(real_labels, fake_predictions)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.d_real_acc_metric.update_state(real_labels, real_predictions)
        self.d_fake_acc_metric.update_state(fake_labels, fake_predictions)
        self.d_acc_metric.update_state([real_labels, fake_labels], [real_predictions, fake_predictions])
        self.g_loss_metric.update_state(g_loss)
        self.g_acc_metric.update_state(real_labels, fake_predictions)
        return {m.name: m.result() for m in self.metrics}

dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=Z_DIM)
dcgan.compile(
    d_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
    g_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
)

dcgan.fit(train, epochs=EPOCHS)

```
<p style='font-size: 0.6em; color: grey; text-align: center; margin: 0;'>Foster 2023</p>

</div>

<!--s-->

## Noisy Labels


The discriminator is trained on noisy labels to prevent it from becoming too confident in its predictions. This is done by adding noise to the labels during training. 

```
...
real_labels = tf.ones_like(real_predictions)
real_noisy_labels = real_labels + NOISE_PARAM * tf.random.uniform(tf.shape(real_predictions))
fake_labels = tf.zeros_like(fake_predictions)
fake_noisy_labels = fake_labels - NOISE_PARAM * tf.random.uniform(tf.shape(fake_predictions))
...
```

It's useful to remember that the discriminator has an easier job than the generator, so this is one way that we can compensate for it.

<!--s-->

## GAN Loss

The goal is for the generator and discriminator to reach an equilibrium where the generator produces realistic images and the discriminator is unable to distinguish between real and fake images.

This is a difficult task, because the discriminator will always have an advantage.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_training_loss.png' style='border-radius: 10px; width: 60%; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey;margin: 0px;'>Foster 2024</p>
</div>

<!--s-->

## GAN Epoch Performance

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_performance_per_epoch.png' style='border-radius: 10px;margin: 0px;'>
   <p style='font-size: 0.6em; color: grey;margin: 0px;'>Foster 2024</p>
</div>

<!--s-->

## GAN Epoch Performance

It's important to ensure that we are not simply overfitting the training data. A good generator should be able to generate images that are not in the training set. One way to assure this is to compare the generated images to their nearest neighbors in the training set.


This can be easily done via L1 distance.

$$ \text{L1}(x, y) = \sum_{i=1}^{n} |x_i - y_i| $$

```python
def compare_images(img1, img2):
    return np.mean(np.abs(img1 - img2))
```

<!--s-->

## GAN Epoch Performance

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/gan_compare_closest_examples.png' style='border-radius: 10px;margin: 0px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

<div class="header-slide">

# GAN Tips and Tricks

<div style="text-align: left; margin-left: 20%; margin-right: 20%;">

As noted previously, GANs are **not** easy to train. Let's talk about some common issues and how to address them.

</div>

</div>

<!--s-->

## Issue: Discriminator Overpowers Generator

The discriminator may be too good at distinguishing between real and fake images. This can cause the generator to get stuck and not improve. In the worst scenario, the discriminator does a perfect job and the generator never learns.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/overpowered_discriminator.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Issue: Discriminator Overpowers Generator

If you find this happening, try the following:

- Increase regularization on the discriminator by adding more dropout or weight decay.

- Decrease the learning rate of the discriminator.

- Reduce parameters in discriminator.

- Add more noise to labels.

<!--s-->

## Issue: Generator Overpowers Discriminator

If you have a poor discriminator, it's possible that the generator will trick the discriminator with a small sample of specific images. This is known as *mode collapse*.

If we were to train the generator over several batches without updating the discriminator, the generator may find a single observsation (mode) that always fools the discriminator. This would cause the gradients of the loss function to collapse to near 0 and it would not recover since evaluating the generator is dependent on the discriminator.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/overpowered_generator.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Issue: Generator Overpowers Discriminator

If you find this happening, try the following:

- Decrease regularization on the discriminator.

- Increase parameters in discriminator.

- Reduce the learning rate of both networks and increase the batch size.

<!--s-->

## Issue: Hyperparameters

GANs have a lot of hyperparameters! By understanding the architecture and the training process, you can better understand how to tune the hyperparameters. 

Alternatively, you can use some of the approaches we covered in L.02.

<!--s-->

## Issue: Uninformative Loss

Since the discriminator and the generator are trained in an adversarial fashion, a natural thought would be that the smaller loss function for the generator == better image quality. However, the generator is only graded against the discriminator performance (which is improving). 

We cannot compare the loss function evaluated at different points in time. 

<!--s-->

<div class="header-slide">

# Wasserstein GAN with Gradient Penalty
# (WGAN-GP)

</div>

<!--s-->

## WGAN-GP

The Wasserstein GAN with Gradient Penalty (WGAN-GP) is a variant of the GAN that uses the Wasserstein distance as the loss function. The Wasserstein distance is a measure of the distance between two probability distributions. The WGAN-GP uses a gradient penalty to enforce the **Lipschitz** constraint on the discriminator.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin: 0px; padding: 0px;">

The WGAN model was introduced in 2017 by Martin Arjovsky and his colleagues. 

This new loss metric is **more meaningful** than the original GAN loss function. It has been shown that Wasserstein loss correlates with a generator's convergence and sample quality.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/wgan_paper.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

</div>
</div>

<!--s-->

## Binary Cross Entropy Loss

<div style = "font-size: 0.9em;">

The binary cross entropy loss function is used to measure the difference between the predicted and actual values.

$$ \text{BCE}(y, \widehat{y}) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\widehat{y}_i) + (1 - y_i) \log(1 - \widehat{y}_i)] $$

To train the GAN discriminator $D$, we calculate the loss when comparing predictions for real images $p_i = D(x_i)$ to the response $y_i = 1$ and the predictions for fake images $p_i = D(G(z_i))$ to the response $y_i = 0$. So, the GAN discriminator loss minimization can be written in terms of expectations as: 

$$ \mathcal{min}_D = -(\mathbb{E}\_{x \sim p\_{data}}[\log D(x)] + \mathbb{E}\_{z \sim p\_z}[\log(1 - D(G(z)))]) $$

To train the GAN generator $G$, we calculate the loss when comparing predictions for fake images $p_i = D(G(z_i))$ to the response $y_i = 1$. So, the GAN generator loss minimization can be written in terms of expectations as:

$$ \mathcal{min}_G = - \mathbb{E}\_{z \sim p\_z}[\log D(G(z))] $$

</div>

<!--s-->

## Wasserstein Loss

We can remove the sigmoid activation function from the discriminator and the output can be any number in range ($-\infty, \infty$). 

The removal of the sigmoid activation in the WGAN critic is crucial because the critic's role is to learn a scoring function and not a probability distribution. The unbounded output allows the critic to provide a more nuanced and informative signal for estimating the Wasserstein-1 distance, which is essential for the stable training of the generator. Constraining the output to (0, 1) with a sigmoid would limit the critic's ability to capture the true differences between the real and generated data distributions.

> This means the discriminator in a WGAN is usually referred to as the *critic* that outputs a *score* rather than a probability.

<!--s-->

## Wasserstein Loss | Critic

The WGAN critic loss can be written in terms of expectations as:

$$ \mathcal{min}_D = - (\mathbb{E}\_{x \sim p\_{data}}[D(x)] - \mathbb{E}\_{z \sim p_z}[D(G(z))])  $$

So the critic tries to **maximize the difference** between the score of real and fake images.

<!--s-->

## Wasserstein Loss | Generator

The WGAN generator loss can be written in terms of expectations as:

$$ \mathcal{min}_G = - \mathbb{E}\_{z \sim p\_z}[D(G(z))] $$

So the generator tries to produce images that are scored as **high as possible** by the critic.

<!--s-->

## The Lipschitz Constraint

The Lipschitz constraint is a condition that ensures that the function does not change too rapidly. 

Recall that our critic can output any number in range ($-\infty, \infty$). This means that the critic can output very large values for some inputs and very small values for others. This can cause the generator to get stuck and not improve.

This requires the critic to be a 1-Lipschitz continuous function. 

$$ \frac{|D(x) - D(y)|}{||x - y||} \leq 1 $$

Where:

- $||x - y||$ is the distance between the two inputs.
- $|D(x) - D(y)|$ is the difference between the outputs of the critic.

<!--s-->

## The Lipschitz Constraint

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Intuitively, this means that we require a limit on the rate at which predictions can change between two images.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://upload.wikimedia.org/wikipedia/commons/5/58/Lipschitz_Visualisierung.gif' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Wikipedia 2023</p>
</div>

</div>
</div>


<!--s-->

## The Lipschitz Constraint

In the original WGAN paper, the authors proposed to enforce the Lipschitz constraint by clipping the weights of the critic to a small range. However, this approach can cause the critic to become too weak and not learn properly -- you're essentially removing the ability of the critic to learn.

A quote from the original WGAN authors (Page 7): 

> Weight clipping is a clearly terrible way to enforce a Lipschitz contraint

<!--s-->

## The Lipschitz Constraint | Gradient Penalty

The gradient penalty is a technique that enforces the Lipschitz constraint by adding a penalty term to the loss function. This penalty term will penalize the model if the gradient norm deviates too much from 1. The gradient penalty loss measures the squared difference between the norm of the gradient of the predictions with respect to the input images and 1. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
def gradient_penalty(self, batch_size, real_images, fake_images):
   alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
   diff = fake_images - real_images
   interpolated = real_images + alpha * diff

   with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated)
      pred = self.critic(interpolated, training=True)

   grads = gp_tape.gradient(pred, [interpolated])[0]
   norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])) #L2 Norm.
   gp = tf.reduce_mean((norm - 1.0) ** 2)
   return gp
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/wasserstein_interpolation.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Wikipedia 2023</p>
</div>

</div>
</div>

<!--s-->

## The Lipschitz Constraint | Gradient Penalty

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/wasserstein_training.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## WGAN-GP

With our new loss function, we don't need to worry about balancing the training of the critic and generator anymore. We can train the critic as many times as we want (actually, to convergence) and the generator will still learn.

A typical ratio is 3-5:1, meaning we update the critic 3-5 times for every generator update.

<div style = "font-size: 0.95em;">

```python

class WGANGP(models.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(WGANGP, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(WGANGP, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric = metrics.Mean(name="c_gp")
        self.c_loss_metric = metrics.Mean(name="c_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                            fake_images = self.generator(random_latent_vectors, training=True)
                            fake_predictions = self.critic(fake_images, training=True)
                            real_predictions = self.critic(real_images, training=True)

                            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                            c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
                            c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

```
<p style='font-size: 0.6em; color: grey; text-align: center; margin: 0;'>Foster 2023</p>
</div>
<!--s-->

## WGAN-GP | Note on Batch Normalization

Batch normalization should not be used in the critic. Batch normalization creates correlation between images in the same batch, which can make a gradient penalty loss less effective.

<!--s-->

## WGAN-GP | CelebA Example

While VAEs tend to produce soft images, GANs are able to produce sharper images.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/wgan_results.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## WGAN-GP | CelebA Example Loss

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/wasserstein_loss.png' style='border-radius: 10px; width: 70%; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster 2023</p>
</div>

<!--s-->

<div class="header-slide">

# Conditional GANs

</div>

<!--s-->

## Conditional GANs

Conditional GANs (cGANs) are a variant of GANs that allow us to generate images conditioned on some input (usually a label). This allows us to generate images that are more specific and controlled.

For example, we can generate images of a specific class (e.g. dogs, cats, etc.) or we can generate images with specific attributes (e.g. hair color, eye color, etc.).

[Conditional GANs](https://arxiv.org/abs/1411.1784) were introduced in 2014 by Mehdi Mirza and Simon Osindero.

<!--s-->

## Conditional GANs | Architecture

The architecture of a cGAN is similar to a regular GAN, but we add the condition to both the generator and the discriminator. The condition can be any type of data, but it is usually a label or a vector. For the generator, we can simply append the latent space vector to the condition vector. For the critic, we add the label information as extra channels to the RGB image.

Your critic has an easy job here -- it can simply look at the label and see if it matches the image. This means that the generator has to work harder to fool the critic, and in doing so it learns to generate images that match the condition.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/conditional_gan.png' style='border-radius: 10px; width: 50%; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Conditional GANs | Train Step

```python
# Critic adjustments.
critic_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
label_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES))
x = tf.keras.layers.Concatenate(axis=-1)([critic_input, label_input])
```

```python
# Generator adjustments.
generator_input = tf.keras.layers.Input(shape=(Z_DIM,))
label_input = tf.keras.layers.Input(shape=(CLASSES,))
x = tf.keras.layers.Concatenate(axis=-1)([generator_input, label_input])
```

```python
# Train step adjustments.
class ConditionalWGAN(models.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(ConditionalWGAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(ConditionalWGAN, self).compile(run_eagerly=True)
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric = metrics.Mean(name="c_gp")
        self.c_loss_metric = metrics.Mean(name="c_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, one_hot_labels = data

        image_one_hot_labels = one_hot_labels[:, None, None, :]
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=IMAGE_SIZE, axis=1)
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=IMAGE_SIZE, axis=2)

        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                            fake_images = self.generator([random_latent_vectors, one_hot_labels], training=True)

                            fake_predictions = self.critic([fake_images, image_one_hot_labels], training=True)
                            real_predictions = self.critic([real_images, image_one_hot_labels], training=True)

                            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                            c_gp = self.gradient_penalty(batch_size, real_images, fake_images, image_one_hot_labels)
                            c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
            fake_predictions = self.critic([fake_images, image_one_hot_labels], training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

```
<p style='font-size: 0.6em; color: grey; text-align: center;'>Foster 2023</p>

<!--s-->

## Conditional GANs | Blonde Label Vector

As you can see here, even when keeping the latent space vector fixed, we can show the effect of the conditional label (blonde).

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/blond_label.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>

</div>

<!--s-->

<div class="header-slide">

# Flowers Demo Pt. 2

</div>

<!--s-->

<div class="header-slide">

# Project 1 | Image Generation

</div>

<!--s-->

## Project 1 | Image Generation Grading Criteria

Project 1 is due on 05.07.2025 and is worth 100 points.

| Criteria | Points | Description |
| -------- | ------ | ----------- |
| Generation of Image | 40 | Your model should be capable of generating images. |
| Code Quality | 20 | Your code should be well-organized and easy to read. Please upload to GitHub and share the link. Notebooks are fine but **must** be tidy. |
| Code Explanation | 25 | You should know your code inside and out. Please do not copy and paste from other sources (including GPT). Xinran and I will conduct an oral exam for your code. |
| Extra Criteria | 15 | Extra criteria is defined in [L.02](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L02/#/78). |

<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

- GANs are a powerful tool for generating images.

- GANs are trained in an adversarial fashion, with the generator trying to fool the discriminator and the discriminator trying to distinguish between real and fake images.
- Wasserstein GANs with Gradient Penalty (WGAN-GP) are a variant of GANs that use the Wasserstein distance as the loss function and a gradient penalty to enforce the Lipschitz constraint on the discriminator.
- Conditional GANs (cGANs) are a variant of GANs that allow us to generate images conditioned on some input (usually a label).
- GANs are not easy to train and require careful tuning of hyperparameters.
- Project 1 is due on 05.07.2025.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following topics: 

- Generative Adversatial Networks
- Wasserstein GANs
- Conditional GANs

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->
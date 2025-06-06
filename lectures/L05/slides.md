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
  ## L.05 | Denoising Diffusion Models

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
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. Diffusion Models
  2. U-Net Architecture

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">


<div style='text-align: center;'>
   <img src='https://miro.medium.com/v2/resize:fit:512/0*CvJeaqjx8FwsSdBZ.gif/' style='border-radius: 10px; height: 100%'>
   <p style='font-size: 0.6em; color: grey;'>DZ, Medium</p>
</div>

</div>

<!--s-->

<div class="header-slide">


<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_text.gif' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey;'>Mayank</p>
</div>

</div>

<!--s-->

## Denoising Diffusion Models

Diffusion models are a state-of-the-art class of generative models that have shown impressive results in generating high-quality images, audio, and other data types. They work by modeling the process of gradually adding noise to data and then learning to reverse this process to generate new samples.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/diffusion_paper.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ho et al.</p>
</div>

<!--s-->

## Denoising Diffusion Models | Main Idea

The main idea behind denoising diffusion models is to model the data distribution by iteratively denoising the data.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/forward_backward_diffusion.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Flowers Dataset

As mentioned in previous lectures, I have the goal of generating flowers. VAEs (L.03) produced blurry images, and GANs (L.04) were difficult to nail the parameters. 

Today we're going to revisit the flower generation task using diffusion models. As a reminder, the [Oxford 102 Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) has over 8000 color images of 102 different flowers.


<div style='text-align: center;'>
    <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/flower_dataset.png' style='border-radius: 10px;'>
    <p style='font-size: 0.6em; color: grey;'>Oxford 102 Flower Dataset</p>
</div>

<!--s-->

## Flowers Dataset | Preprocessing

We can preprocess the images in the [Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) by resizing them and normalizing the pixel values to be between 0 and 1.

Using <span class="code-span">utils.image_dataset_from_directory</span> from TensorFlow, we can load the dataset directly from the directory structure. Lazily loading the images allows us to handle large datasets without running out of memory. The images are loaded in batches, and we can apply preprocessing functions to each batch as they are loaded.

```python

import tensorflow as tf
import tensorflow_datasets as tfds

train = tfds.load('oxford_flowers102', split='train', shuffle_files=True, as_supervised=False)

def preprocess(example):
    img = example['image']
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method='bilinear')
    img = tf.cast(img, tf.float32) / 255.0
    return img

train = train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train = train.shuffle(1000, seed=42)
train = train.batch(BATCH_SIZE, drop_remainder=True)
train = train.prefetch(tf.data.AUTOTUNE)
train = train.repeat(DATASET_REPETITIONS)

```
<p style='font-size: 0.6em; color: grey; margin: 0; text-align: center;'>Foster 2023</p>

<!--s-->

<div class="header-slide">

# Forward Diffusion
*Adding Iterative Noise.*

</div>

<!--s-->

## Denoising Diffusion Models | Main Idea

The main idea behind denoising diffusion models is to model the data distribution by iteratively denoising the data.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/forward_backward_diffusion.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Forward Diffusion Process

The forward diffusion process is a Markov chain that gradually adds noise to the data. The process is defined by a series of Gaussian distributions, where each step adds a small amount of noise to the data. The noise schedule is a sequence of values that control the amount of noise added at each time step.

The forward diffusion process can be defined as:

$$ x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon $$

Where:
- $ x_t $ is the noisy image at time step $ t $
- $ x_{t-1} $ is the image at the previous time step
- $ \beta_t $ is the noise schedule at time step $ t $
- $ \epsilon $ is Gaussian noise sampled from $\mathscr{N}(0, I)$

<!--s-->

## Forward Diffusion Process

We scale the input data by $\sqrt{1 - \beta_t}$ and add noise scaled by $\sqrt{\beta_t}$. This ensures that the variance of the output image $x_t$ remains constant over time.

With large enough $T$, the $x_t$ will be approximately Gaussian.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/forward_diffusion.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0'>Foster 2023</p>
</div>

<!--s-->

## Forward Diffusion Process | Why the Square Roots?

### Rule 1: 

$$ Var(aX) = a^2 Var(X) $$

If we assume that $x_{t-1}$ has zero mean and unit variance, then $\sqrt{1 - \beta_t} \cdot x_{t-1}$ will have variance 1 - $\beta_t$ and $\sqrt{\beta_t} \cdot \epsilon_{t-1}$ will have variance $\beta_t$.

### Rule 2: 

$$ Var(X + Y) = Var(X) + Var(Y) $$

By adding these together, we get a new distribution $x_t$ with zero mean and variance $1 - \beta_t + \beta_t = 1$. So, if $x_0$ is zero mean and unit variance, then $x_t$ will also be zero mean and unit variance for all $t$.

<!--s-->

## Forward Diffusion Process | Noise Schedule

Put another way: 

$$ q(x_t | x_{t-1}) = \mathscr{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot I) $$

You can think of this as slowly shifting the original image distribution towards a normal noise distribution.


<!--s-->

## Reparameterization Trick

This reparameterization trick makes it possible to jump straight from an image $x_0$ to a noisy image $x_t$, without having to go through all the intermediate steps ($t$ applications of $q$).

Lets say: 

$$ \alpha_t = 1 - \beta_t $$
$$ \bar{\alpha_t} = \prod_{i=1}^{t} \alpha_{i} $$

We can re-write the following, using the fact that we can add two Gaussians to obtain a new Gaussian:

$$ x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{(1 - \alpha_t)} \cdot \epsilon_{t-1} $$
$$ x_t = \sqrt{(\alpha_t \cdot \alpha_{t-1})} \cdot x_{t-2} + \sqrt{(1 - \alpha_t) \cdot \alpha_{t-1}} \epsilon $$
$$ x_t = \ldots $$

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{(1 - \bar{\alpha}_t)} \cdot \epsilon$$

<!--s-->

## Reparameterization Trick

$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$

With this reparameterization trick, we have the ability to sample from the distribution $q(x_t | x_0)$ directly, without having to go through all the intermediate steps. Additionally, we can define the diffusion schedule using the $\bar{\alpha}_t$ values instead of the $\beta_t$ values, with the interpretation that $\bar{\alpha}_t$ is the variance due to the signal (the original image, $x_0$) and $1 - \bar{\alpha}_t$ is the variance due to the noise ($\epsilon$).

Finally, we can rewrite the forward diffusion process as:

$$ q(x_t | x_0) = \mathscr{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I) $$

<!--s-->

## Reparameterization Trick | Continued

$$ q(x_t | x_0) = \mathscr{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I) $$

Interpretation of each term: 

- $\mathscr{N}(x_t; \mu, \sigma^2)$ is a Gaussian distribution with mean $\mu$ and variance $\sigma^2$.

- $x_0$ is the original image, which is the data we want to model.

- $q(x_t | x_0)$ is the distribution of the noisy image, which is a Gaussian distribution with mean $\sqrt{\bar{\alpha}_t} \cdot x_0$ and variance $(1 - \bar{\alpha}_t) \cdot I$.

- $\sqrt{\bar{\alpha}_t} \cdot x_0$ is the mean of the distribution, which is a scaled version of the original image. The scaling factor $\sqrt{\bar{\alpha}_t}$ controls how much of the original image is preserved in the noisy image.

- $(1 - \bar{\alpha}_t) \cdot I$ is the covariance of the distribution, which is a scaled version of the identity matrix. The scaling factor $(1 - \bar{\alpha}_t)$ controls how much noise is added to the original image.

<!--s-->

## Diffusion Schedules

We're free to choose a different $\beta_t$ schedule to be used at each time step. How the noise is added to the image is determined by the $\beta_t$ schedule, and this is called the **diffusion schedule**. Three types of diffusion schedules are commonly used: Linear, Cosine, and Offset Cosine.

<!--s-->

## Diffusion Schedules | Linear

The linear diffusion schedule is the simplest and most commonly used. It is defined by a linear function of time.

<div class = "col-wrapper">
<div class="c1" style = "width: 30%; font-size: 0.8em;">

$$ \beta_t = \beta_0 + t \cdot \frac{\beta_T - \beta_0}{T} $$
$$ \beta_0 = 0.0001, \beta_T = 0.02, T = 1000 $$

</div>
<div class="c2" style = "width: 70%">

```python

T = 1000
diffusion_times = tf.convert_to_tensor([x / T for x in range(T)])

def linear_diffusion_schedule(diffusion_times):
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = tf.math.cumprod(alphas)
    signal_rates = tf.sqrt(alpha_bars)
    noise_rates = tf.sqrt(1 - alpha_bars)
    return noise_rates, signal_rates

```
<p style='font-size: 0.6em; color: grey; margin: 0; text-align: center;'>Foster 2023</p>

</div>
</div>

<!--s-->

## Diffusion Schedules | Cosine

The cosine diffusion schedule is a more complex schedule that uses trigonometric functions to control the noise level.

<div class = "col-wrapper">
<div class="c1" style = "width: 40%; font-size: 0.8em;">

$$ \bar\alpha_t = \cos^2\left(\frac{t}{T} \cdot \frac{\pi}{2}\right) $$

Recall that the x_t equation was: 

$$ x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$

Using the trig identity $\cos^2(x) + \sin^2(x) = 1$, we can rewrite this as:

$$ x_t = \cos\left(\frac{t}{T} \cdot \frac{\pi}{2}\right) x_0 + \sin\left(\frac{t}{T} \cdot \frac{\pi}{2}\right) \epsilon $$

</div>
<div class="c2" style = "width: 60%">

```python
def cosine_diffusion_schedule(diffusion_times):
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates
```

</div>
</div>

<!--s-->

## Diffusion Schedules | Offset Cosine

The author of the paper "Denoising Diffusion Probabilistic Models" proposed an offset cosine schedule, which is a variant of the cosine schedule that adds a small offset to the noise level.

```python
def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates
```

<!--s-->

## Diffusion Schedules | Comparison

We can compute the $\bar\alpha_t$ values for each $t$ to show how much the signal ($\bar\alpha_t$) and noise ($1 - \bar\alpha_t$) change over time.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/schedule_comparisons.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Diffusion Schedules | Comparison

We can also visualize the noise level at each time step for each schedule.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/linear_vs_cosine.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0; text-align: center;'>Foster 2023</p>
</div>

<!--s-->

## Reverse Diffusion Process

The reverse diffusion process is the process of denoising the data. We want to build a neural network $p_0(x_{t-1}|x_t)$ that can undo the noising process by approximating the reverse distribution $q(x_{t-1}|x_t)$.

Then, the goal is to sample random noise from $\mathscr{N}(0, I)$ and use the reverse diffusion process to generate new data.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/forward_backward_diffusion.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

<!--s-->

## Reverse Diffusion Process

By predicting the noise $\epsilon$ at each time step, we can subtract it from the noisy image to get the denoised image. So we train a network to predict $\epsilon$ that has been added to a given image $x_0$ at timestep $t$.

So we sample an image $x_0$ and transform it by $t$ noising steps to get the image $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$. We give this new image and the noising rate $\bar{\alpha}_t$ to the network, and train it to predict the noise $\epsilon$. 

The gradient step is against the squared error between the predicted noise and the actual noise.

<!--s-->

## Reverse Diffusion Process | Training

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/algo.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Ho 2020</p>
</div>

<!--s-->

## Reverse Diffusion Process | EMA

We actually want to train an exponential moving average (EMA) version of the model weights to get a more stable prediction. This is because the model is trained on a noisy version of the data, and the EMA helps to smooth out the noise. We'll see an example of this in the final codeblock.

<!--s-->

## DiffusionModel (Keras)

```python

class DiffusionModel(models.Model):
    def __init__(self):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = unet
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

        diffusion_times = tf.random.uniform(shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.random.uniform(shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)
        return {m.name: m.result() for m in self.metrics}

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
        return pred_images

```

<!--s-->

<div class="header-slide">

# U-Net Architecture

</div>

<!--s-->

## Model Selection

We can use any model we want for the network. The original authors for DDM used a U-Net architecture, which is a type of convolutional neural network that is commonly used for image segmentation tasks. 
<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The U-Net architecture is designed to capture both local and global features of the input image, making it well-suited for denoising tasks. It also outputs the same size as the input, which is important for our task.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center; height: 70%;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/unet.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Foster 2023</p>
</div>

</div>
</div>

<!--s-->

## U-Net | Sinusoidal Positional Embedding

The sinusoidal embedding was introduced in a paper called "Attention is All You Need" by Vaswani et al. in 2017. An adaptation of that idea was utilized in "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Mildenhall et al. in 2020.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/embedding.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023, Vaswani 2017</p>
</div>

<!--s-->

## U-Net | Sinusoidal Positional Embedding

The core idea is that we want to convert a scalar value (the noise variance) into a distinct vector representation that can be used as input to the model. Vaswani et al used it for the position of a word in a sentence, and Mildenhall et al extended the idea to continuous values. We can use this idea to encode the noise variance into a vector representation that can be used as input to the model.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The noise variance is a continuous value, and we want to provide the model with a way to understand how the noise level changes over time. By encoding the noise variance into a vector representation, we can provide the model with a way to understand the relationship between the noise level and the image at each time step.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/embedding.png' style='border-radius: 10px; width: 100%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023, Vaswani 2017</p>
</div>

</div>
</div>

<!--s-->

## U-Net | Sinusoidal Positional Embedding

Here's an implementation of the sinusoidal positional embedding function.


```python
def sinusoidal_embedding(x):
    frequencies = tf.exp(tf.linspace( tf.math.log(1.0), tf.math.log(1000.0), NOISE_EMBEDDING_SIZE // 2))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings
```


<!--s-->

## U-Net | Residual Blocks

Both the DownBlocks and UpBlocks use residual blocks, which are a type of neural network layer that allows for the training of very deep networks by adding a skip connection between the input and output of the layer. This allows for the training of very deep networks without the vanishing gradient problem.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

> Vanishing gradient problem. The gradient is backpropagated through the network, it can become very small, making it difficult to update the weights of the earlier layers. This can make it difficult to train very deep networks.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/residual_block.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

</div>
</div>


<!--s-->

## U-Net | Residual Blocks

You can think of residual blocks as a way to add a shortcut connection between the input and output of a layer. This allows the network to learn an identity function, which is a function that simply returns the input. In some residual blocks, we include an extra Conv2D layer with kernel size 1 to match the number of channels in the input and output.

```python
def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply
```

<!--s-->

## U-Net | DownBlocks

Every DownBlock increases the number of channels via block_depth ResidualBlocks, and then downsamples the image by a factor of 2 using an AveragePooling layer. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/down_up_block.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>


</div>
</div>

<!--s-->

## U-Net | UpBlocks

An UpBlock first applies an UpSampling layer to increase the size of the image by a factor of 2, and then concatenates the upsampled image with the corresponding skip connection from the DownBlock.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/down_up_block.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>


</div>
</div>

<!--s-->

## U-Net | Model

```python
noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = layers.Conv2D(32, kernel_size=1)(noisy_images)

noise_variances = layers.Input(shape=(1, 1, 1))
noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
noise_embedding = layers.UpSampling2D(size=IMAGE_SIZE, interpolation="nearest")(noise_embedding)

x = layers.Concatenate()([x, noise_embedding])

skips = []
x = DownBlock(32, block_depth=2)([x, skips])
x = DownBlock(64, block_depth=2)([x, skips])
x = DownBlock(96, block_depth=2)([x, skips])
x = ResidualBlock(128)(x)
x = ResidualBlock(128)(x)
x = UpBlock(96, block_depth=2)([x, skips])
x = UpBlock(64, block_depth=2)([x, skips])
x = UpBlock(32, block_depth=2)([x, skips])
x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)
unet = models.Model([noisy_images, noise_variances], x, name="unet")
```
<p style='font-size: 0.6em; color: grey; margin: 0; text-align: center;'>Foster 2023</p>

<!--s-->

## Training the DDM

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
ddm = DiffusionModel()
ddm.normalizer.adapt(train)
ddm.compile(optimizer=optimizers.experimental.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY), loss=losses.mean_absolute_error)
ddm.fit(train, epochs=EPOCHS)

```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/training_epochs.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

</div>
</div>

<!--s-->

## Training Performance

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/performance_per_epoch.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Generating Images

Okay, so we have a model capable of denoising images by predicting the added noise. So our goal is to use this model to **gradually** undo noise from a random noise image. To accomplish this, we can jump from $x_t$ to $x_{t-1}$ in two steps. First, we use our models noise prediction to calculate an estimate for the original image $x_0$. Then we re-apply the predicted noise to this image (but only over $t - 1$ timesteps to produce $x_{t-1}$).

If we repeat this process over a number of steps (which we can determine), we can estimate $x_0$ and generate a new image.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/sampling.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Generating Images

The following equation (Song et al. 2020) is used to generate images from the model:

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/song_equation.png' style='border-radius: 10px; margin: 0'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Song 2020</p>
</div>

The first term inside the brackets is the estimated image $x_0$, calculated using the noise predicted by our network $\epsilon_\theta(x_t, t)$. We then scale this by the $t-1$ signal rate $\sqrt{\bar{\alpha}\_{t-1}}$ and add the noise predicted by our network scaled by the $t-1$ noise rate $\sqrt{(1 - \bar{\alpha}\_{t-1} - \sigma\_{t}^2)}$. Additional noise is added to the image ($\sigma_t \epsilon_t$) with $\sigma_t$ determining how random we want our generation process to be.

<div style = "font-size: 0.8em;">

> The special case $\sigma_t = 0$ for all t is called the Denoising Diffusion Implicit Model (DDIM), which is a deterministic sampling method.
</div>

<!--s-->

## Generating Images (DDIM)


```python[31:40; ]
class DiffusionModel(models.Model):
    def __init__(self):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = unet
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

        diffusion_times = tf.random.uniform(shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.random.uniform(shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)
        return {m.name: m.result() for m in self.metrics}

```

<!--s-->

## Generating Images | Diffusion Steps

Intuitively, the more diffusion steps we take, the more accurate our estimate of $x_0$ will be. However, this comes at the cost of increased computation time. In practice, we can get good results with as few as 10 diffusion steps.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/diffusion_steps.png' style='border-radius: 10px; width: 70%'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Foster 2023</p>
</div>

<!--s-->

## Generating Images | Interpolating

Just like with VAEs, we can interpolate between points in the latent space to generate new images. With spherical interpolation, we can ensure that the variance remains constant while blending the two Gaussian noise maps together. This is important because it allows us to smoothly transition between the two noise maps without introducing artifacts or discontinuities in the generated images.

The initial noise map at each step is given by: 

$$ a \cdot \sin(\frac{\pi}{2} \cdot t) + b \cdot \cos(\frac{\pi}{2} \cdot t) $$

Where t ranges smoothly from 0 to 1 and $a$ and $b$ are the two noise maps we want to interpolate between.

<!--s-->

## Generating Images | Interpolating

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_lecture/diffusion_morphing.png' style='border-radius: 10px; margin: 0;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Foster 2023</p>
</div>

<!--s-->

<div class="header-slide">

# Demo

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Midterm Feedback

  How are things going in MSAI 495? Please provide any feedback or suggestions for improvement, and I'll do my best to accommodate for future lectures. 

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Midterm Feedback" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. Diffusion Models
  2. U-Net Architecture

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->
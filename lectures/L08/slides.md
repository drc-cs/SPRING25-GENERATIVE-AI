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
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  #  Generative AI
  ## L.08 | Conditional & Multimodal Models

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
  ## On a scale of 1-5, how confident are you with conditional generation & multi-modal models?

  - Conditional VAEs / GANs / Diffusion Models
  - Multimodal Models (CLIP, DALL-E, PaliGemma)

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Conditional Image Generation
### Easily Extend VAE / GAN / Diffusion Models

</div>

<!--s-->

<div class="header-slide">

# Conditional VAE

</div>

<!--s-->

## VAE Review

Recall that a Variational Autoencoder (VAE) is a generative model that learns to encode data into a latent space and then decode it back to the original data space. The VAE consists of two main components: the encoder and the decoder.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_1.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

## VAE Review

The VAE learns to encode the data into a lower-dimensional latent space while ensuring that the latent variables follow a specific distribution (usually Gaussian).

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_2.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

## VAE Review

The goal of a VAE is to take any sampled point from this distribution and decode it back to the original data space.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_3.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

## VAE Review

This works very well in practice, but there is one big issue: we cannot determine ahead of time what the generated image will look like. 

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_5.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

## Conditional VAE

Enter Conditional VAE! We can condition the encoder and decoder on the input data, which allows us to generate images that are *conditioned*.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_6.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

## Conditional VAE

But now, something interesting happens to our latent space. Sampling the same point in the latent space will yield different images depending on the *condition*. How does this work?

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_7.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

## Conditional VAE

Now our latent space contains information that is not class-spcific. In this case, it may contain other information, like stroke width or the angle the number is written. 

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cvae/cvae_8.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Dykeman 2016</p>
</div>

<!--s-->

<div class="header-slide">

# Conditional GAN

</div>

<!--s-->

## Conditional GANs (From L.04)

Conditional GANs are a variant of GANs that allow us to generate images conditioned on some input (usually a label). This allows us to generate images that are more specific and controlled.

For example, we can generate images of a specific class (e.g. dogs, cats, etc.) or we can generate images with specific attributes (e.g. hair color, eye color, etc.).

[Conditional GANs](https://arxiv.org/abs/1411.1784) were introduced in 2014 by Mehdi Mirza and Simon Osindero.

<!--s-->

## Conditional GANs | Architecture  (From L.04)

The architecture of a conditional GAN is similar to a regular GAN, but we add the condition to both the generator and the discriminator. The condition can be any type of data, but it is usually a label or a vector. For the generator, we can simply append the latent space vector to the condition vector. For the critic, we add the label information as extra channels to the RGB image.

Your critic has an easy job here -- it can simply look at the label and see if it matches the image. This means that the generator has to work harder to fool the critic, and in doing so it learns to generate images that match the condition.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/gan_lecture/conditional_gan.png' style='border-radius: 10px; width: 50%; margin: 0px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Conditional GANs | Train Step  (From L.04)

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

<div class="header-slide">

# Conditional Diffusion

</div>

<!--s-->

## Conditional Diffusion Models | Stable Diffusion

Stable diffusion was released in August 2022 by Stability AI and continues to be one of the most popular text-to-image diffusion models. It may be the most popular **open-source** text-to-image diffusion model. 

Stable Diffusion uses latent diffusion (introduced by Rombach et al in 2021) in the paper "High-Resolution Image Synthesis with Latent Diffusion Models". [[paper](https://arxiv.org/pdf/2112.10752)]

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/stability.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>stability.ai</p>
</div>

<!--s-->

## Conditional Diffusion Models | Stable Diffusion

**Latent Diffusion** is a diffusion model wrapped within an autoencoder, so that diffusion process operates on a latent space representation of the image (rather than the image itself). This allows for the denoising U-Net model to be much smaller and faster.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/stable_diffusion.png' style='border-radius: 10px;  width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Conditional Diffusion Models | Stable Diffusion

To enable text conditioning, the first version of Stable Diffusion used CLIP (OpenAI) text embeddings. Stable Diffusion 2 custom-trained their own CLIP called OpenCLIP. We'll cover CLIP in more detail shortly.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/stable_diffusion.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## Conditional Diffusion Models | Stable Diffusion

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; height: 100%">

```text
Prompt: generate a collage-style illustration inspired by the Procreate raster graphic editor, 
photographic illustration with the theme, 2D vector, art for textile sublimation, containing 
surrealistic cartoon cat wearing a baseball cap and jeans standing in front of a poster, 
inspired by Sadao Watanabe, Doraemon, Japanese cartoon style, Eichiro Oda, Iconic high 
detail character, Director: Nakahara Nantenbō, Kastuhiro Otomo, image detailed, by Miyamoto, 
Hidetaka Miyazaki, Katsuhiro illustration, 8k, masterpiece, Minimize noise and grain in photo 
quality without lose quality and increase brightness and lighting,Symmetry and Alignment, 
Avoid asymmetrical shapes and out-of-focus points. Focus and Sharpness: Make sure the image 
is focused and sharp and encourages the viewer to see it as a work of art printed on fabric.

Parameters: Steps: 20, Sampler: Euler a, CFG scale: 7.0, Seed: 1117431804, Size: 712x1024, 
Model : SD XL
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/stable_diffusion_cat.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>https://stablediffusion.fr/prompts</p>
</div>

</div>
</div>

<!--s-->

<div class="header-slide">

# CLIP

</div>

<!--s-->

## CLIP | History [[openai](https://openai.com/index/clip/)]

> In 2013, Richard Socher and co-authors at Stanford developed a proof of concept by training a model on CIFAR-10 to make predictions in a word vector embedding space and showed this model could predict two unseen classes. [[paper](https://papers.nips.cc/paper/2013/file/2d6cc4b2d139a53512fb8cbb3086ae2e-Paper.pdf)]

> The same year DeVISE scaled this approach and demonstrated that it was possible to fine-tune an ImageNet model so that it could generalize to correctly predicting objects outside the original 1000 training set. [[paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf)]

<!--s-->

## CLIP | Contrastive Language-Image Pretraining

Introduced in early 2021, openai's CLIP is capable of combining images and text into the same latent space.

CLIP consists of two main components: an **image encoder** and a **text encoder**. The image encoder processes images, while the text encoder processes text descriptions. Both encoders map their respective inputs into a shared latent space, allowing for direct comparison between images and text.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/CLIP.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>openai 2021</p>
</div>

<!--s-->

## CLIP

CLIP is trained on a large dataset of images and their corresponding text descriptions (~400 million samples). The model learns to associate images with their textual descriptions by maximizing the similarity between the image and text embeddings in a shared latent space.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# 1. extract feature representations of each modality.
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# 2. Linear layer to project joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# 3. Calculate scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# 4. Calculate loss along text and image axes (implicit softmax)
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/CLIP.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>openai 2021</p>
</div>

</div>
</div>


<!--s-->

## CLIP | Cosine Similarity to Probability

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cosine_to_probability.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Warfield 2023</p>
</div>

<!--s-->

## CLIP | Cross-Entropy Loss

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/cross_entropy_loss.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Warfield 2023</p>
</div>
</div>

<!--s-->

## CLIP | Zero-Shot Classification

By combining the flexibility of language with vision models, CLIP can perform zero-shot classification. This means that it can classify images into categories that it has never seen before, based on the text descriptions provided. In practice, this greatly enhances the model's ability to generalize and adapt to new tasks without requiring additional training.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/resnet_v_clip.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>Radford 2021</p>
</div>

<!--s-->

<div class="header-slide">

# DALL.E 2
### (Text-to-Image Diffusion Model)

</div>

<!--s-->

## DALL.E 2

CLIP isn't a generative model, but it can be used to guide generative models (like with Stable Diffusion). DALL.E 2 is a text-to-image diffusion model that uses CLIP to guide the generation process.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/DALLE2.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## DALL.E 2 | CLIP Utility

DALL.E 2 uses the text encoder from CLIP to generate text embeddings. These embeddings are then used to condition the diffusion model during the image generation process. Why do you think the authors chose to use CLIP for this task?

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/DALLE2.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## DALL.E 2 | The Prior

We need a model to convert the text embedding into an image embedding. This model is called the **prior**. The prior is a diffusion model that takes the text embedding and generates an image embedding. This image embedding is then passed to the decoder, which generates the final image.

DALL.E 2 authors tried two different approaches for this task: a diffusion model and an autoregressive model. The diffusion model was found to be more effective, so they used it as the prior.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/diffusion_prior.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2023</p>
</div>

<!--s-->

## DALL.E 2 | The Decoder

After the prior, DALL.E 2 has a decoder. With the image embedding prediction that we get from the prior, we can concatenate the image embedding with the text embedding to condition our diffusion-based decoder.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/dalle_decoder.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster 2023</p>
</div>

<!--s-->

## DALL.E 2 | Upscaling

DALL.E 2 uses a diffusion model to upscale the image, allowing it to generate high-resolution images.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/dalle_upscaling.png' style='border-radius: 10px; width: 30%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Foster 2023</p>
</div>

<!--s-->

## DALL.E 2 | Generated Images

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/dalle2_examples.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>Ramesh 2022</p>
</div>

<!--s-->

<div class="header-slide">

# SigLIP + Gemma == PaliGemma (VLM)

</div>

<!--s-->

## SigLIP

Unlike CLIP, SigLIP calculates the loss for each image-text pair independently using a sigmoid function. It treats the prediction of whether an image and text pair match as a binary classification problem.

SigLIP's loss function does not require a global view of all pairwise similarities within a batch. The loss for each pair is independent of other pairs in the mini-batch.

The pairwise sigmoid loss allows for more efficient scaling to larger batch sizes as it avoids the quadratic memory complexity associated with CLIP's softmax normalization. It can also perform well with smaller batch sizes.

<!--s-->

## SigLIP

[[paper](https://arxiv.org/pdf/2303.15343)]

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### CLIP

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/siglip_clip.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Taha 2024</p>
</div>

</div>
<div class="c2" style = "width: 50%">

### SigLIP

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/siglip.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Taha 2024</p>
</div>


</div>
</div>

<!--s-->

## Gemma

Gemma is a family of lightweight, **open-source** models developed by Google, offering a range of capabilities and model sizes for various generative AI tasks. These models are designed to be deployed on devices, making them accessible for developers and researchers. 

They are similar to Gemini, but much smaller (Gemma 3 ranges from 1B - 27B) and more efficient. They are also similar to your favorite *[insert ~7B parameter LLM here]*, making them a good example for a simple VLM extension.

<!--s-->

## PaliGemma

By combining the image encoder from SigLIP with the text encoder from Gemma, we can create a new vision-language model called PaliGemma. The original model ([PaLI-3](https://arxiv.org/pdf/2310.09199)) was introduced in 2023 by Google. [PaliGemma](https://arxiv.org/abs/2407.07726) was introduced in 2024 by Google.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/PaliGemma0.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Google</p>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with conditional generation & multi-modal models?

  - Conditional VAEs / GANs / Diffusion Models
  - Multimodal Models (CLIP, DALL-E, PaliGemma)

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Zero-Shot Detection with CLIP
### [[colab](https://colab.research.google.com/drive/1dQ2-Q_OHfWzQUiCNto9ugOGeSlPmwzpr?usp=sharing)]

</div>

<!--s-->
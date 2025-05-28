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
  ## L.09 | World Models and Exam Review

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

## Announcements | General

- Exam will be on **June 4th**.

- Presentations will be on **June 4th**.
   - If selected, you will present your project in the last class for (substantial) extra credit.

- We will schedule time to do your oral project review (~5 minutes each) on June 5th or 6th.

<!--s-->

## Announcements | Lecture Polls

Thank you for doing the lecture polls! Here is how we have fared so far this quarter. If you had any other feedback to provide, please make sure you fill out the CTEC.

| Lecture | Avg. Starting Comprehension ( / 5) |Avg. Delta [25th-75th Percentile] | Wilcoxon P-Value |
|------------|-----------------|-------|-------|
| L.01 | 2.69 | <span style="color:#2eba87; font-weight: bold">+ 1.38 [1.0-2.0]</span> | $1.9e-03$ |
| L.02| 1.40 | <span style="color:#2eba87; font-weight: bold">+ 1.62 [1.0-2.0]</span> | $6.2e-02$ |
| L.03| 2.94 | <span style="color:#2eba87; font-weight: bold">+ 1.12 [1.0-2.0]</span> | $1.0e-03$ |
| L.04 | 2.50 | <span style="color:#2eba87; font-weight: bold">+ 1.26 [1.0-2.0]</span> | $6.7e-04$ |
| L.05 | 2.33 | <span style="color:#2eba87; font-weight: bold">+ 1.57 [1.0-2.0]</span> | $7.2e-04$ |
| L.06 | 2.89 | <span style="color:#2eba87; font-weight: bold">+ 1.00 [0.5-1.0]</span> | $5.8e-04$ |
| L.07 | 2.72 | <span style="color:#2eba87; font-weight: bold">+ 0.97 [1.0-1.0]</span> | $1.8e-04$ |
| L.08 | 2.60 | <span style="color:#2eba87; font-weight: bold">+ 1.20 [1.0-2.0]</span> | $1.7e-03$ |

<!--s-->

## Announcements | Midterm Feedback Check-in

Feedback helps our program get better! If you have any additional feedback for Gen AI, please schedule some time to discuss with me or fill out the CTEC form next week.

| Feedback | Action |
| ------------|-------|
| Less theory, more examples | We spent less time in math world and more time on applications (RAG, MCP, Stable Diffusion), as well as more interactive demos (Forecasting Competition, CLIP). |
| PDFs of Lecture Slides | Slides continue to be available to download. |
| Textbook & Code Snippets | Foster 2024 is a great textbook, I hope a few of you have been able to use it. Every code snippet is available in their [book / repo](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/)

<!--s-->

<div class="header-slide">

# Projects

</div>

<!--s-->

## Project 2 Rubric

Project 2 is due on 06.04.2025 and is worth 100 points.

| Criteria | Points | Description |
| -------- | ------ | ----------- |
| Generation of Text | 40 | Your model should be capable of generating text. |
| Code Quality | 20 | Your code should be well-organized and easy to read. Please upload to GitHub and share the link. Notebooks are fine but **must** be tidy. |
| Code Explanation | 25 | You should know your code inside and out. Please do not copy and paste from other sources (including GPT). Xinran and I will conduct an oral exam for your code. |
| Extra Criteria | 15 | Extra criteria is defined in the [README](https://github.com/drc-cs/SPRING25-GENERATIVE-AI?tab=readme-ov-file#extra-project-criteria). |

<!--s-->


## Project Votes

We're doing a ranked-choice vote for the projects. Please vote for your top 3 projects in each category (image and text). 

The top 6 projects (3 from each category) will be selected for the final presentations.

<!--s--> 

## Image Project Vote

Please vote in the format <span class="code-span">8,3,5</span> if you are voting for projects 8,3,5 in that ranked order. 

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

```text
1. InPaintrait - Rediscovering Lost Details in Portraits (In-painting)
2. AI-Powered Image Restoration: A Unified Pipeline for Noise Removal, Super-Resolution, and Colorization
3. FurnitureGen: AI-Powered Interior Design Generator
4. Image Inpainting for Removal of Certain Predefined Object Classes
5. Pokémon Diffusing：Catching ’Em All with a Diffusion Model！
6. Impressionist Paintings Generator
7. Diffusion Models for Generating Fashion Images
8. "Paint me like one of your French girls" -- Image Generation from 27 Art Styles
9. Pixel Pastries: GAN based dessert dream machine
10. Latent Space Exploration of Icons with a Variational Autoencoder
11. Variational Autoencoder for Fashion Image Generation
12. Clinical Impressions: Medical Imaging as Artistic Expression
13. Fruit Generator
14. Minecraft Texture Generator: Diffusion Model for 16×16 pixel art generation
15. VAE from Scratch for Fashion Image Generation
16. Acute stroke detection using images of facial assymetry
17. Generating Hand-drawn Sketches from Text Prompts using a Multimodal Variational Autoencoder
18. Cubism Art Generator
19. Low data approaches for GANimal generation and latent space interpretation
20. T-shirt Graphic Design Generator
21. Aesthetic Moodboard Generator
```

</div>
<div class="c2" style = "width: 40%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Image Vote" width="100%" height="100%" style="border-radius: 10px; padding: 0; margin: 0"></iframe>
</div>
</div>


<!--s-->

## Text Project Vote

Please vote in the format <span class="code-span">8,3,5</span> if you are voting for projects 8,3,5 in that ranked order. 

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

```text
1. Asimbot
2. Mirror Me: A Self-Affirmation Text Generator
3. Food Fact Generator
4. Graph-Augmented Retrieval-Augmented Generation (RAG) for Biomedical Question Answering
5. DataVizAI: Fine-Tuning Vision-Language Models for Chart/Plot Understanding
6. Haiku Text Generator
7. StoryTeller AI: Personalised Story Generator with Voice Narration
8. K-Drama Browser - A tool for getting quick summaries of K-Drama's
9. GPT-2 Small from Scratch
10. CaptionCrafter：A Multimodal AI Co-Creation Tool for Lifestyle Content Creators
11. OneLine News Headline Generator
12. Text Generation Project Proposal: CreativeWritingGPT
13. Dad-Joke Generator
14. Empathetic Echoes: Emotion Aware Psychiatric Chatbot
15. StoryGen - Generates short stories with intro, middle and ending from text input
16. Transformer-Based News Headline Generation
17. Automatic Generation of Detailed Operative Reports from Brief Descriptions
18. Generating Movie Scripts by Genre
19. News article TLDR/summary generator
20. Conversation Primer Bot
21. Explainable GPT2 for Email Subject Line Generation
```

</div>
<div class="c2" style = "width: 30%">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Text Vote" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide"> 

# World Models

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **world models**?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Agenda

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### World Models
- Reinforcement Learning
- VAE + MDN-RNN + CMA-ES
- Dream State Learning

</div>
<div class="c2" style = "width: 50%">

### Exam Review
- Format
- Review
- Study Tips

</div>
</div>

<!--s-->

## Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.

<div class = "col-wrapper" style = "font-size: 0.8em;">
<div class="c1" style = "width: 50%; margin-right: 10px;">

### Environment

The world in which the agent operates. It defines the set of rules that govern the game state update process and reward allocation, given the agent’s previous action and current game state.

### Agent

The entity that interacts with the environment.

### State

The data that represents a particular situation that the agent may encounter (also just called a state).

</div>
<div class="c2" style = "width: 50%">

### Action

The set of all possible actions that the agent can take in the environment.

### Reward

The feedback signal that the agent receives from the environment after taking an action. It indicates how good or bad the action was in terms of achieving the goal.

### Episode

A sequence of states, actions, and rewards at timestep t that ends when the agent reaches a terminal state or a predefined condition is met.

</div>
</div>

<!--s-->

## Reinforcement Learning

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/reinforcement_learning.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>IBM 2024</p>
</div>

<!--s-->

## Car Racing Challenge

The [Car Racing Challenge](https://gymnasium.farama.org/environments/box2d/car_racing/) is a reinforcement learning task where an agent learns to drive a car on a track. It is available in OpenAI Gym.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/carracing_overview_video.gif' style='border-radius: 10px; margin-bottom: 0px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Ha 2018</p>
</div>

<!--s-->

## Car Racing Challenge 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 10px;">

### Action Space (Environment)

<div style="margin-left: 1em;">

**0**: Steering (-1 is full left, 1 is full right)<br><br>
**1**: Gas (0 is no throttle, 1 is full throttle)<br><br>
**2**: Brake (0 is no brake, 1 is full brake)

</div>

### Observation Space (Environment)

Top-down view (96 x 96 pixels) of the car and race track.

</div>
<div class="c2" style = "width: 50%">

### Reward

The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

### Episode Termination

The episode finishes when all the tiles are visited. The car can also go outside the playfield - that is, far off the track, in which case it will receive -100 reward and die.

</div>
</div>

<!--s-->

## Car Racing Challenge 

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
   <div style="text-align: center; width: 70%;">
      <img src="https://storage.googleapis.com/slide_assets/car_racing.png" style="border-radius: 10px; margin-bottom: 0px;">
      <p style="font-size: 0.6em; color: grey; margin: 0px;">Foster 2024</p>
   </div>
</div>

<!--s-->

<div class="header-slide">

# World Model

</div>

<!--s-->

## World Model Overview

The world model was introduced by Ha and Schmidhuber in 2018. It is a framework that combines reinforcement learning with generative models to learn a compact representation of the environment.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Variational Autoencoder (VAE)

A VAE is used to learn a compact representation of the environment's state.

### Recurrent neural network with Mixture Density Network (RNN-MDN)

An RNN-MDN is used to model the temporal dynamics of the environment.

### Controller (CMA-ES)

The controller is responsible for taking actions based on the learned world model.
</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

</div>
</div>

<!--s-->

## Variational Autoencoder (VAE)

As we have learned, a VAE is a generative model that learns to encode input data into a latent space and then decode it back to the original data. In the context of world models, the VAE is used to learn a compact representation of the environment's state.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

**Intuition**: You don't actively analyze every single pixel in your view when you're driving a car. Instead, you have a mental model of the environment that allows you to make decisions based on a simplified representation of the world.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

</div>
</div>

<!--s-->

## Recurrent Neural Network with Mixture Density Network (RNN-MDN)

As you drive, you continuously observe the environment and update your mental model. The RNN-MDN is used to model the temporal dynamics of the environment, allowing the agent to predict future states based on past observations.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

**Intuition**: You don't just look at the current state of the road; you also consider how the road has changed over time and how it might change in the future. This helps you make better decisions about when to turn, accelerate, or brake.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

</div>
</div>

<!--s-->

## Controller

The controller is responsible for taking actions based on the learned world model. The controller is a densely connected neural network with inputs as the latent state from the VAE, and the hidden state from the RNN-MDN.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

**Intuition**: You need to take your understanding of the environment **and** the temporal changes to make a decision. The output is a set of actions that the agent can take in the environment.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>


</div>
</div>

<!--s-->

## Dialogue Scenario

**VAE** (looking at latest 64 × 64 × 3 observation): This looks like a straight road, with a slight left bend approaching, with the car facing in the direction of the road (z). 

**RNN**: Based on that description (z) and the fact that the controller chose to accelerate hard at the last timestep (action), I will update my hidden state (h) so that the next observation is predicted to still be a straight road, but with slightly more left turn in view. 

**Controller**: Based on the description from the VAE (z) and the current hidden state from the RNN (h), my neural network outputs [0.34, 0.8, 0] as the next action.

<p style='font-size: 0.6em; color: grey;'>Quoted from Foster 2024</p>

<!--s-->

<div class="header-slide">

# Training the World Model

</div>

<!--s-->

## Training the World Model Step-by-Step

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Step 1

Collect 10,000 rollouts from a random policy.

### Step 2

Train VAE (V) to encode each frame into a latent vector $z \in \mathbb{R}^{32}$.

</div>
<div class="c2" style = "width: 50%">

### Step 3

Train MDN-RNN (M) to model $P(z_{t+1} | a_t, z_t, h_t)$ and $P(r_{t+1} | a_t, z_t, h_t)$, where $h_t$ is the hidden state of the RNN at timestep $t$.

### Step 4

Evolve Controller (C) to maximize the expected cumulative reward of a rollout.

</div>
</div>

<!--s-->

## Architecture Overview

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

<!--s-->

## Step 1 | Collecting Random Rollout Data

The first step is to collect rollout data from the environment, using an agent taking random actions. Practically, we can capture multiple episodes in parallel by using multiple Python processes, each running a separate instance of the environment.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/random_rollout_data.png' style='border-radius: 10px; width: 40%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## Step 2 | Training the VAE | Architecture

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_vae_architecture.png' style='border-radius: 10px; width: 40%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## Step 2 | Training the VAE | Visual Validation

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/carracing_vae_compare.gif' style='border-radius: 10px; margin-bottom: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Ha 2018</p>
</div>

<!--s-->

## Step 2 | Training the VAE | Latent Space Interpolation

The decode model accepts a $z$ vector as input and reconstructs the original image. Below, you can see the interpolation of two of the dimensions of $z$ to show how each dimension appears to encode a particular aspect of the track.

Ha et al has a fun [interactive demo](https://worldmodels.github.io/) as well.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_latent_space_interpolation.png' style='border-radius: 10px; width: 40%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Foster 2024</p>
</div>

<!--s-->

## Architecture Overview

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

<!--s-->

## Step 3 | Training the RNN-MDN

We can use our trained VAE model to pre-process each frame at time $t$ into $z_t$ to train our RNN-MDN model. Using this pre-processed data, along with the recorded random actions, our MDN-RNN can now be trained to model $P(z_{t+1} | a_t, z_t, h_t)$ as a mixture of Gaussians.

*Sidebar -- they actually also use the previous reward. $P(z_{t+1}, r_{t+1} | a_t, z_t, h_t, r_{t-1})$ 

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/mdn_rnn_ha.png' style='border-radius: 10px; width: 60%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

<!--s-->

## Step 3 | Training the RNN-MDN


<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The MDN-RNN consists of an LSTM layer (the RNN), followed by a densely connected layer (the MDN) that transforms the hidden state of the LSTM into the parameters of a mixture distribution.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/RNNMDN.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

</div>
</div>

<!--s-->

## Step 3 | Training the RNN-MDN | Shapes

<div style = "margin-left: 1em;">

**36**: The input to the LSTM layer is a vector of length 36. A concatenation of the encoded z vector (length 32) from the VAE, the current action (length 3), and the previous reward (length 1).

**256**: The output from the LSTM layer is a vector of length 256. One value for each LSTM cell in the layer.

**481**: The output from the MDN layer is a vector of length 481 (next slide).

</div>

<!--s-->

## Step 3 | Training the RNN-MDN | 481 RNN-MDN Output

The aim of a mixture density network is to model the fact that our next z could be drawn from one of several possible distributions with a certain probability. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

In the car racing example, five mixture distributions are chosen to represent the possible scenarios the car could be in. For each of the 5 mixtures, we need a mu and a logvar (to define the distribution) and a log-probability of this mixture being chosen (logpi), for each of the 32 dimensions of z. This makes 5 × 3 × 32 = 480 parameters. 

The one extra parameter is for the reward prediction.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/RNN_output.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

</div>
</div>

<!--s-->

## Step 3 | RNN-MDN | Getting $z_{t+1}$

Using the output of the MDN layer, we can sample to create $z_{t+1}$.

1. Split the 481-dimensional output vector into the 3 variables (logpi, mu, logvar) and the reward value. 

2. Exponentiate and scale logpi so that it can be interpreted as 32 probability distributions over the 5 mixture indices. 

3. For each of the 32 dimensions of z, sample from the distributions created from logpi (i.e., choose which of the 5 distributions should be used for each dimension of z). 

4. Fetch the corresponding values of mu and logvar for this distribution. 

5. Sample a value for each dimension of z from the normal distribution parameterized by the chosen parameters of mu and logvar for this dimension.

<!--s-->

## L.09 | Q.01

Why do you think we choose to sample $z_{t+1}$ from a mixture (MDN-RNN) instead of directly predicting $z_{t+1}$?

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.09 | Q.01" width = "100%" height = "100%" style = "border-radius: 10px;"></iframe>

<!--s-->

## Step 3 | Training the RNN-MDN | Loss Function

The loss function for the MDN-RNN is the sum of the $z$ vector reconstruction loss and the reward loss. The $z$ vector reconstruction loss is the negative log-likelihood of the distribution predicted by the MDN-RNN, given the true value of $z$.

The reward loss is the mean squared error between the predicted reward and the true reward.

<!--s-->

## Architecture Overview

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

<!--s-->

## Step 4 | Training the Controller | Architecture

The controller is a densely connected neural network with no hidden layers. It connects the input vector (size 288) directly to the action vector. The input vector is a concatenation of the current z vector (length 32) and the current hidden state of the LSTM (length 256), giving a vector of length 288. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Since we are connecting each input unit directly to the 3 output action units, the total number of weights to tune is 288 × 3 = 864, plus 3 bias weights, giving 867 in total.

How should we train the weights of this neural network? 

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

</div>
</div>

<!--s-->

## Step 4 | Evolutionary Strategy

1. Create a population of candidate solutions (i.e., neural networks) with random weights.

2. Loop over the population until convergence:
   - Evaluate each candidate solution by running it in the environment and calculating the cumulative reward.
   - Select the top-performing candidates based on their cumulative rewards.
   - Generate new candidates by mutating the weights of the selected candidates and adding some random noise.
   - Replace the worst-performing candidates with the new candidates.

<!--s-->

## Step 4 | CMA-ES

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is an evolutionary strategy that adapts the covariance matrix of the search distribution to improve the exploration of the search space.

At each generation, CMA-ES updates the mean of the distribution to maximize the likelihood of sampling the high-scoring agents from the previous timestep. At the same time, it updates the covariance matrix of the distribution to maximize the likelihood of sampling the high-scoring agents, but using the previous mean.

<!--s-->

## Step 4 | CMA-ES | Example Update Step

Here we are trying to find the minimum point of a highly nonlinear function in two dimensions—the value of the function in the red/black areas of the image is greater than the value of the function in the white/yellow parts of the image.

<div class = "col-wrapper" style = "margin: 0; height: 45%;">
<div class="c1" style = "width: 50%; margin: 0">

1. Randomly generated 2D normal and sampled population (blue). 

2. Isolate best 25% of population (purple).

</div>
<div class="c2" style = "width: 50%; margin: 0">

3. Set the mean of the new distribution to the mean of the best 25% (purple). Set the covariance matrix to the covariance of the best 25% (purple) but use the existing mean in the covariance calculation.

4. Then sample new population from the new normal distribution (grey)

</div>
</div>

<div style='text-align: center; margin: 0; padding: 0;'>
   <img src='https://storage.googleapis.com/slide_assets/CMAES1.png' style='border-radius: 10px; width: 70%; margin: 0;'>
   <p style='font-size: 0.4em; color: grey; margin: 0;'>Foster 2024</p>
</div>

> The larger the difference between the current mean and the mean of the best 25%, the more exploration we will do in the next generation.

<!--s-->

## Step 4 | CMA-ES | Example Generations

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/CMAES2.png' style='border-radius: 10px; width: 70%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Foster 2024</p>
</div>

<!--s-->

## CMA-ES | Car Racing

For our car racing eample, we don't have a well-defined loss function to optimize, so we use the cumulative reward as the fitness function. The CMA-ES algorithm will then optimize the weights of the controller to maximize the cumulative reward.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/CMAES_Orch.png' style='border-radius: 10px; width: 50%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## Architecture Overview

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_overview_ha.png' style='border-radius: 10px; width: 60%;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

<!--s-->

## Car Racing | Results

After many generations, the controller has learned to drive the car around the track, maximizing the cumulative reward. Ha et al. is still near the top of the leaderboard 7 years later, with a score of 906 +/- 21. 

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/ha_performance.png' style='border-radius: 10px; margin-bottom: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Ha 2018</p>
</div>

<!--s-->

## Car Racing | Ablation

Ha et al tested the benefit of the MDN-RNN contribution to the world model by comparing the performance of the controller with and without the $h$ vector.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### $z$ only

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/carracing_z_only.gif' style='border-radius: 10px; margin-bottom: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Ha 2018</p>
</div>

</div>
<div class="c2" style = "width: 50%">

### $z$ and $h$

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/carracing_z_and_h.gif' style='border-radius: 10px; margin-bottom: 0px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Ha 2018</p>
</div>

</div>
</div>


<!--s-->

<div class="header-slide">

# In-Dream Training

</div>

<!--s-->

## In-Dream Training

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_without_dream.png' style='border-radius: 10px; width: 50%; margin: 0;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## In-Dream Training

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_without_dream.png' style='border-radius: 10px; width: 90%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/world_with_dream.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

</div>
</div>

<!--s-->

## In-Dream Training | Temperature

Training entirely within the MDN-RNN can lead to overfitting -- e.g. when the agent finds a strategy that doesn't generalize well to the real environment.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The authors, Ha and Schmidhuber, propose to use a temperature parameter that amplifies variance when sampling z through the MDN-RNN. This allows the agent to explore more diverse actions and states, leading to better generalization. 

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/mdn_rnn_ha.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha 2018</p>
</div>

</div>
</div>

<!--s-->

## In-Dream Training | Performance

Tuning temperature allows the agent to explore more diverse actions and states, leading to better generalization. They used an example based on the computer game Doom, and beat the current leader at the time.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

What's remarkable is that they showed excellent performance having *only trained the world model on random steps in the real environment*. 

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/dream_performance.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ha and Schmidhuber 2018</p>
</div>

</div>
</div>
<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with **world models**?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Exam Review

</div>

<!--s-->

## Exam Review

<div class = "col-wrapper" style = "font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Format

~ 10 multiple choice and 10 short answer questions. This amounts to ~2 questions per lecture for 20 points total. Partial credit will be given for short answer questions, so please write your answers clearly and concisely.

### Timing

You will have ~ 1.5 hours to complete the in-class exam.

</div>
<div class="c2" style = "width: 50%">

### Resources

This is a closed-note exam. If I want you to interpret a diagram or equation, I will provide it. Focus on learning the concepts and how they relate to each other, rather than memorizing equations or diagrams.

### Content

Focus on the concepts in the following slides. Anything is fair game, but I will focus on these. 

</div>
</div>


<!--s-->

## Exam Review

### L01 (Intro) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L01/#/)

- Know generative vs discriminative models in terms of p(x) and p(y|x).
- How are MLE and generative models related?

### L02 (MLOps) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L02/#/)

- Understand, compare, and contrast grid search, random search, and hyperband optimization.
- Understand and differentiate between data and model parallelism.

### L03 (Autoencoders) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L03/#/)

- Understand AEs and VAEs
    - Encoder vs Decoder vs latent space (z)
    - Choosing the best loss function (RMSE vs BCE)
    - What issue(s) do VAEs solve compared to AEs?
    - What do we add to the VAE loss term compared to AEs?
    - What is the reparameterization trick? Why do we need it?
    - How would you create a face morph between two images using a VAE?

<!--s-->

## Exam Review

### L04 (Generative Adversarial Networks) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L04/#/)

- Understand GANs
    - Why is the loss term for a vanilla GAN uninterpretable?
    - What does Wasserstain Loss in GANs do, and why does that require a gradient penalty? Use terms like score vs probability and lipschitz.

### L05 (Diffusion Models) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L05/#/)

- Understand the goals of the forward and reverse diffusion processes.
- Understand the reparameterization trick in diffusion models.
- Compare linear and cosine noise schedules.
- Describe the generation process of a diffusion model.

### L06 (Autoregressive Text) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L06/#/)

- Explain k, q, and v in the context of attention mechanisms.
- Explain different parts of self-attention described in lecture (will provide a diagram).
- Describe a RAG pipeline.

<!--s-->

## Exam Review

### L07 (Forecasting) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L07/#/)

- Describe probabilistic forecasting and how it differs from point forecasting.
- Describe how to build a forecasting model with XGBoost.
- Explain N vs D vs Vanilla Linear models. 

### L08 (Multimodal / Conditional) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L08/#/)

- Explain how zero-shot detection with CLIP works.
- What is meant by "latent diffusion model" in the context of Stable Diffusion?

### L09 (World Models) [🔗](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L09/#/)

- Understand the components of a world model (VAE, RNN-MDN, CMA-ES).
- Explain how the VAE is trained and how it is used in the world model.

<!--s-->
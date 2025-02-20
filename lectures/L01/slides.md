---
title: MSAI XXX
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
  ## L.01 | Introduction

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/e7eb235d-f490-4ce1-877a-99114b96ff60/OFTqzm1m09.json" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Generative AI.
  ## Please check in by creating an account and entering the code on the board.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.web.app?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

# Syllabus

<div style="overflow-y: scroll; height: 80%; font-size: 0.8em;">

| Week | Date       | Topic                                | Module       | Topics Covered  |
|------|------------|--------------------------------------|--------------|--------------------------------------------------------------------------------|
| 1    | ------ | Introduction, Environment, and Containerization | Setup | Course structure, generative vs. discriminative modeling, rise of generative AI, containers |
| 2    | ------ | Machine Learning Operations | Setup | Distributed data preprocessing, containerized training, hyperparameter tuning and distributed training strategies |
| 3    | ------ | Foundational Knowledge | Image | Multilayer perceptrons, data preparation, model training, evaluation |
| 4    | ------ | Autoencoders | Image | Autoencoders, variational autoencoders, latent space analysis |
| 5    | ------ | Generative Adversarial Networks | Image | Deep convolutional GANs, bricks dataset, discriminator, generator, WGAN-GP, Wasserstein Loss, gradient penalty, conditional GAN |
| 6    | ------ | Diffusion Models | Image | Denoising diffusion models, forward/reverse process, U-Net denoising model, stable diffusion |
| 7    | ------ | Autoregressive Models | Text | text data handling, RNN, LSTM, GRU, extensions to LSTM, Bidirectional LSTM, Stacked LSTM, attention mechanisms, transformers, text generation metrics |
| 8    | ------ | Autoregressive Models | Text | Encoder vs. Decoder, BERT, GPT-2, T5 architectures |
| 9    | ------ | Multimodal Model Strategies | Text + Image | Image + Text models approaches, CLIP, DALL-E |
| 10   | ------ | Demo Day & Final Exam | Text + Image | Presentation and demonstration of **any** generative AI projects |

</div>

<!--s-->

## Attendance

Attendance at lectures is **mandatory** and in your best interest.

Your Attendance & Comprehension score is worth 40% of your final grade. Lectures will have graded quizzes throughout, and the top 12 quiz scores will be used to calculate your attendance grade.

<!--s-->

## Grading

There is a high emphasis on the practical application of the concepts covered in this course. 
<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There is a high emphasis on the practical application of the concepts covered in this course.

| Component | Weight |
| --- | --- |
| Attendance & Comprehension Quizzes | 40% |
| Project Part I (Exhibit) | 20% |
| Project Part II (Chatbot) | 20% |
| Exam | 20% |

</div>
<div class="c2" style = "width: 50%">

| Grade | Percentage |
| --- | --- |
| A | 94-100 |
| A- | 90-93 |
| B+ | 87-89 |
| B | 83-86 |
| B- | 80-82 |
| C+ | 77-79 |
| C | 73-76 |
| C- | 70-72 |

</div>
</div>

<!--s-->

## LLMs (The Talk)

<iframe src="https://lottie.host/embed/e7eb235d-f490-4ce1-877a-99114b96ff60/OFTqzm1m09.json" height = "100%" width = "100%"></iframe>

<!--s-->

## Exams

There are two exams in this class. They will cover the theoretical and practical concepts covered in the lectures and homeworks. If you follow along with the lectures and homeworks, you will be well-prepared for the exams.

<!--s-->

## Academic Integrity [&#x1F517;](https://www.northwestern.edu/provost/policies-procedures/academic-integrity/index.html)

### Homeworks

- Do not exchange code fragments on any assignments.
- Do not copy solutions from any source.
- You cannot upload / sell your assignments to code sharing websites.

<!--s-->

## Accommodations

Any student requesting accommodations related to a disability or other condition is required to register with AccessibleNU and provide professors with an accommodation notification from AccessibleNU, preferably within the first two weeks of class. 

All information will remain confidential.

<!--s-->

## Mental Health

If you are feeling distressed or overwhelmed, please reach out for help. Students can access confidential resources through the Counseling and Psychological Services (CAPS), Religious and Spiritual Life (RSL) and the Center for Awareness, Response and Education (CARE).

<!--s-->

## Stuck on Something?

### **Office Hours**

- Time: TBA
- Location: TBA

### **Canvas Discussion**

- Every homework & project will have a discussion thread on Canvas.
- Please post your questions there so that everyone can benefit from the answers!

<!--s-->

## Stuck on Something?

### **Email**

We are here to help you! Please try contacting us through office hours or the dedicated Canvas discussion threads.

Meixi Lu (GA): meixilu2025@u.northwestern.edu
Joshua D'Arcy (Professor): joshua.darcy@northwestern.edu


<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Spring Quarter Plan
  ## After looking at the syllabus, is there anything you want me to cover that I'm not?

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.web.app?label=Coverage" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

# Core Concepts of Generative AI

<!--s-->

## Core Concepts | Generative vs Discriminative Models

### Generative Models

- Estimates distribution $p(x)$.

### Discriminative Models

- Estimate the conditional probability distribution $p(y|x)$.

<!--s-->

## Core Concepts | Hello, World! 

Let's create a very simple generative model. Here we will generate a set of points $X$ from a distribution $p_{data}$.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Our task is to choose a different point $x = (x_1, x_2)$ from the distribution $p_data$ Where would you place the point $x$?

</div>
<div class="c2" style = "width: 50%">
</div>
</div>

<!--s-->

## Core Concepts | Intuition

The point $x$ should be placed in the region where the density of $p_data$ is high. We did this by creating a mental model which we can call $p_model$ that approximates $p_data$. The goal of generative modeling is to learn the distribution $p_model$ that best approximates $p_data$.

<!--s-->

## Core Concepts | Generative Modeling Framework

- We have a dataset $X = \{x_1, x_2, ..., x_n\}$ drawn from an unknown distribution $p_data$.
- We want to build a model $p_model$ that approximates $p_data$.
- We can then sample from $p_model$ to generate new data points.

<!--s-->

## Core Concepts | Desired Properties of $p_model$

    - Accuracy (if $p_model$ is close to $p_data$, then it should be able to generate data points that look like they were drawn from $p_data$)
    - Generation (it shuold be possible to generate new data points from $p_model$)
    - Representation (it should be possible to understand how different high-level features of the data are represented in $p_model$)

<!--s-->

## Core Concepts | Generative Modeling

Revisit the original exercise with the map overlay. It becomes clear that our $p_model$ is an oversimplification of $p_data$. This is the essence of generative modeling: approximating a complex distribution with a simpler one.

<!--s-->

## Core Concepts | Representation Learning
 
  - Representation learning is the process of learning a representation of the data that makes it easier to extract useful information when building models.
  - Cover example of the bicuit tin dataset.
  - Latent space $z$ is a lower-dimensional representation of the data $x$. The goal is to learn a mapping from $x$ to $z$ that captures the underlying structure of the data with $f(x) = z$. This is the essence of representation learning.

<!--s-->

<div class="header-slide">

# Probability Theory

</div>

<!--s-->

## Probability Theory | Sample Space

The set of all possible outcomes of an experiment.

<!--s-->

## Probability Theory | Probability Distribution

A function $p$ that represents the relative likelihood of each outcome in the sample space. The integral of $p$ over the sample space is 1.

<!--s-->

## Probability Theory | Parametric Modeling

A technique that we can use to structure our approach to finishing $p_model$. A parametric model is a family of density functions $p_{\theta}(x)$, where $\theta$ is a set of parameters that we can tune to make $p_{\theta}(x)$ more closely approximate $p_data$.

<!--s-->

## Probability Theory | Likelihood

The likelihood of a set of parameters *likelihood* $L(\theta | X)$ is the probability of observing the data $X$ given the parameters $\theta$. It is defined as $L(\theta | X) = p_{\theta}(X)$. The goal of training a generative model is to find the parameters $\theta$ that maximize the likelihood of the data. That is, the likelihood of $\theta$ given a point $x$ is $p_{\theta}(x)$. If we have a whole dataset $X = \{x_1, x_2, ..., x_n\}$, then the likelihood of $\theta$ given the dataset is $p_{\theta}(X) = \prod_{i=1}^{n} p_{\theta}(x_i)$. The log-likelihood is often used instead of the likelihood because it is easier to work with. The log-likelihood is defined as $\log p_{\theta}(X) = \sum_{i=1}^{n} \log p_{\theta}(x_i)$. 

Subtle note: The likelihood is a function of the parameters and not the data. It should not be intrepreted as the probability that a given parameter set is correct.

<!--s-->

## Probability Theory | Maximum Likelihood Estimation

    - Maximum likelihood estimation is the technique that allows us to estimate theta hat $\hat{\theta}$ -- the set of parameters $\theta$ that is most liklely to explain som eobserved data X. Formally in terms of likelihood, we can write this as:

    $$ \hat{\theta} = \arg \max_{\theta} p_{\theta}(X) $$

    $\hat{\theta}$ is also called the maximum likelihood estimate.

    In the world map example, the MLE is the smallest rectangle that still contains all of the points in the training set.

    Since neural networks typically minimize a loss function, we can also think of MLE as minimizing the negative log-likelihood:

    $$ \hat{\theta} = \arg \min_{\theta} -\log p_{\theta}(X) $$

    Generative modeling can be considered to be a form of MLE, where the parameters $\theta$ are learned to maximize the likelihood of the data.

<!--s-->

## Tools and Platforms

Generative AI is a rapidly evolving field, and often you're most limited by your ability to **scale**. For that reason, let's cover some of the tools and platforms that we'll be using in this course.

### Docker

Docker is a platform that allows you to develop, ship, and run applications in containers. Containers are lightweight, portable, and efficient, making them ideal for running computationally intensive tasks like training deep learning models.

### QUEST

Quest is Northwestern's high-performance computing cluster. It is a shared resource that is available to all Northwestern researchers. It is a great resource for running computationally intensive tasks, such as training deep learning models. We will run **slurm** jobs on Quest to train our models.

<!--s-->

<div class="header-slide"> 

# Docker

</div>

<!--s-->

## Docker | Agenda

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

1. Why Containerization?
2. What is Docker and Why is it Useful?
3. Benefits of Using Docker
4. Docker Theory
5. Using Docker
6. Docker Demo

</div>
<div class="c2" style = "width: 30%">

<img src="https://raw.githubusercontent.com/docker-library/docs/c350af05d3fac7b5c3f6327ac82fe4d990d8729c/docker/logo.png" width="100%">

</div>

</div>

> <span style="font-style: normal;"> Vocabulary will be placed in boxes. </span>

<!--s-->

## Why Containerization?

A **container** is a lightweight, portable, and efficient way to package applications and their dependencies. Containers isolate applications from the host system and other containers, making them easier to deploy and manage.

| **Container** | **Virtual Machine** |
|---------------|---------------------|
| Lightweight | Heavyweight |
| Faster startup | Slower startup |
| Less resource usage | More resource usage |
| Shared kernel | Separate kernel |

> <span style="font-style: normal;"> The **kernel** is the core of an operating system that manages system resources. </span>

> <span style="font-style: normal;"> **Virtual machines** are software emulations of physical computers that run an operating system and applications. </span>

<!--s-->

## What is Docker and Why is it Useful?

**Docker** is an open-source platform designed to simplify the process of creating, deploying, and running applications. You can think of Docker as a self-contained package that includes everything an application needs to run: the code, runtime, system tools, libraries, and settings. 

This package is called a **image**. When you run an image, it creates a **container**, an isolated environment that runs the application.

> <span style="font-style: normal;">An **image** is a snapshot of an application and its dependencies. </span>

> <span style="font-style: normal;">A **container** is a running instance of an image. </span>

<!--s-->

## Benefits of Using Docker

1. Consistency Across Environments

2. Efficiency

3. Scalability

4. Isolation and Security

<!--s-->

## Benefits of Using Docker | Consistency

<div class = "col-wrapper" style = "height: 70%;">
<div class="c1" style = "width: 50%;">

Docker ensures that software behaves the same on every machine. Developers can be confident that applications that work on their computers will work in production.

Docker containers are typically based on a Linux distribution, which provides a consistent environment for applications.

</div>
<div class="c2" style = "width: 50%">

<img src = "https://miro.medium.com/v2/resize:fit:1400/0*Qqqd7UsfFDPL7WXh.jpeg" width="100%">

</div>
</div>

> <span style="font-style: normal;"> **Linux** distributions: Variants of the Linux operating system. Linux is by far the most popular OS in the world for web servers, cloud computing, and supercomputers. </span>
 
<!--s-->

## Benefits of Using Docker | Efficiency


Containers share the host's operating system kernel, which makes them more lightweight and efficient than traditional virtual machines.

This results in faster application delivery, reduced resource consumption, and lower overhead.


<img src="https://www.docker.com/wp-content/uploads/2022/12/admins-ask-about-docker-2.png" width="100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Docker 2022</p>

<!--s-->

## Benefits of Using Docker | Scalability

Docker makes it easy to scale applications horizontally by adding more containers. This supports modern cloud-native development practices.

Containers can be easily replicated and distributed across multiple hosts, providing flexibility and scalability.

<img src = "https://miro.medium.com/v2/resize:fit:1400/1*MzGIkBAGQwUyN2-Rs9opfA.jpeg" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Thakur 2024</p>

<!--s-->

## Benefits of Using Docker | Isolation and Security

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Containers encapsulate applications and their dependencies completely, providing isolation that improves security.

</div>
<div class="c2" style = "width: 50%">

<img src="https://joeeey.com/static/c161c59028d1e817d0cdce747b9e79e7/d8bb9/covers.png" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Miller 2023</p>

</div>
</div>

<!--s-->


## Docker Architecture


<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

<!--s-->

## Docker Architecture | Key Docker Components

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

- Docker Daemon

- Docker Client

- Docker Images

- Docker Containers

- Docker Registries

- Namespaces and Control Groups

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

</div>
</div>

<!--s-->

## Docker Architecture | Daemon

The **Docker Daemon** (`dockerd`) is the heart of Docker, responsible for running containers on a host. It listens for API requests and manages Docker objects (images, containers, networks, etc.).

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" height="50%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

</div>

<div class="c2" style = "width: 50%">

> <span style="font-style: normal;"> A **daemon** is a background process that runs continuously, waiting for requests to process. </span>

</div>

</div>


<!--s-->

## Docker Architecture | Client

The **Docker Client** is a command-line tool (CLI) used by the user to interact with the Docker daemon. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Common CLI commands include <span class="code-span">docker pull</span>, <span class="code-span">docker build</span>, and <span class="code-span">docker run</span>. The client sends commands to the daemon, which executes them on the host.

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

</div>
</div>

<!--s-->

## Docker Architecture | Images

**Docker Images** are immutable, read-only templates used to create containers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

An image might include an OS, application code, and dependencies required to run an application. Images are built from a series of layers. Each layer represents a modification to the previous layer, allowing for efficient storage and distribution of images.

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

</div>
</div>

<!--s-->

## Docker Architecture | Containers

**Docker Containers** are running instances of Docker images.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

They can be started, stopped, moved, or deleted using Docker commands. Containers are isolated from each other and the host system, but they share the host OS's kernel. This makes them lightweight and efficient.

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

</div>
</div>

<!--s-->

## Docker Architecture | Registries

**Docker Registries** store Docker images. A popular public registry is Docker Hub, but private registries can also be used. Docker images can be pushed to and pulled from registries, allowing for easy distribution and sharing of images.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Other examples of registries include:

- **Amazon Elastic Container Registry (ECR)**
- **Google Container Registry (GCR)**
- **Azure Container Registry (ACR)**

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1400/0*G82uZfX0ozIih3-_" width="100%" style = "border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NordicAPIs</p>

</div>
</div>

<!--s-->

## Docker Architecture | Core Concepts

### Namespaces and Control Groups
Docker uses Linux namespaces to provide isolation for containers and control groups (cgroups) to limit resource usage.

### Union File System
Layers are used to create Docker images. Each layer is a modification over the previous one, which allows efficient storage and reduced bandwidth usage when distributing an image. A Union File System (UFS) combines these layers into a single view (union) of the file system.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

> <span style="font-style: normal;"> **Namespaces**: Isolate containers from each other and the host system. </span>

> <span style="font-style: normal;"> **Control Groups (cgroups)**: Limit resource usage for containers. </span>

</div>
<div class="c2" style = "width: 50%">

> <span style="font-style: normal;"> **Union File System**: Efficiently store and distribute Docker images. </span>

</div>
</div>

<!--s-->

## Using Docker

1. Installing Docker
2. Building from a Dockerfile
3. Running Containers

<!--s-->

## Using Docker | Installing Docker

To start using Docker, you need to install the Docker Engine on your machine. It can be downloaded from the Docker website and is available for various operating systems, including Windows, MacOS, and Linux.

Download Docker Desktop: [Docker Desktop](https://www.docker.com/products/docker-desktop)

<!--s-->

## Using Docker | Building a Dockerfile

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

A **Dockerfile** is a text document that contains all the commands needed to assemble a Docker **image**. It starts with a <span class="code-span">FROM</span> instruction that specifies the base image. 


Usually, it also includes commands like <span class="code-span">WORKDIR</span>, <span class="code-span">COPY</span>, <span class="code-span">RUN</span>, and <span class="code-span">CMD</span> to set up the environment and run the application.

</div>
<div class="c2" style = "width: 50%">

```dockerfile

# Use an official Python runtime as a parent image.
FROM python:3.8-slim

# Set the working directory.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Install any needed packages specified in requirements.txt.
# RUN is used to execute commands during the build process.
RUN pip install --no-cache-dir -r requirements.txt

# Start the application. CMD specifies the command to run when the container starts.
CMD ["python", "app.py"]

```

</div>
</div>

<!--s-->

## Dockerfile | Cheatsheet

Here are some common Dockerfile commands.

| Command | Description |
|---------|-------------|
| <span class="code-span">FROM</span> | Specifies the base image to use. |
| <span class="code-span">WORKDIR</span> | Sets the working directory for subsequent commands. |
| <span class="code-span">COPY</span> | Copies files from the host to the container. |
| <span class="code-span">RUN</span> | Executes commands during the build process. |
| <span class="code-span">CMD</span> | Specifies the command to run when the container starts. |
| <span class="code-span">EXPOSE</span> | Exposes a port to the host machine. |
| <span class="code-span">ENV</span> | Sets environment variables. |
| <span class="code-span">ENTRYPOINT</span> | Configures the container to run as an executable. |
<!--s-->

## Docker | CLI Cheatsheet

Here are some common Docker CLI commands.

| Command | Description |
|---------|-------------|
| <span class="code-span">docker --version</span> | Checks the installed version of Docker. |
| <span class="code-span">docker pull [image_name]</span> | Pulls an image from a registry. |
| <span class="code-span">docker build -t [image_name] .</span> | Builds an image from a Dockerfile. |
| <span class="code-span">docker run [image_name]</span> | Runs a container from an image, common flags include <span class="code-span">-d</span> for detached mode, <span class="code-span">-p</span> for port mapping, and <span class="code-span">-v</span> for volume mounting. |
| <span class="code-span">docker ps</span> | Lists running containers. |
| <span class="code-span">docker images</span> | Lists images on the host. |

<!--s-->

## Docker Demo | Course Environment

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

To the right is a dockerfile that will install the course environment and run <span class="code-span">code-server</span> in a container. 

This will allow you to run the course environment locally without installing any dependencies to your operating system (Windows, MacOS, Linux).

</div>
<div class="c2" style = "width: 70%; height: 100%;">

```dockerfile
# Use the official Miniconda3 image as a parent image.
FROM continuumio/miniconda3

# Clone the class repository.
RUN apt-get update && apt-get install -y git curl
RUN git clone https://github.com/drc-cs/SPRING25-GENERATIVE-AI.git

# Set the working directory.
WORKDIR /SPRING25-GENERATIVE-AI

# Create a new Conda environment from the environment.yml file.
RUN conda env create -f environment.yml

# Install vscode server.
RUN curl -fsSL https://code-server.dev/install.sh | bash

# Add code-server to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Expose the port that the server is running on.
EXPOSE 8080

# Run the code-server command when the container starts.
CMD ["code-server", "--auth", "none", "--bind-addr", "0.0.0.0:8080", "."]

```

</div>
</div>

<!--s-->

## Docker Demo | Running the Course Environment

Since we run the <span class="code-span">code-server</span> on port 8080, we need to map the container's port to the host machine. We can do this using the <span class="code-span">-p</span> flag. Here we mapped the container's port 8080 to the host machine's port 8080. Since code-server is already running within the docker container, we can access it by visiting <span class="code-span">localhost:8080</span> in your browser.

As long as the container is running, you can access the course environment by visiting <span class="code-span">localhost:8080</span> in your browser at any time. It will even work offline!

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```bash
# Build the Docker image.
docker build -t genai .

# Run the Docker container on port 8080 (local).
docker run -p 8080:8080 genai
```

</div>
<div class="c2" style = "width: 50%">

<img src="https://user-images.githubusercontent.com/35271042/118224532-3842c400-b438-11eb-923d-a5f66fa6785a.png" width="100%">

</div>
</div>

<!--s-->

<div class="header-slide">

# QUEST

</div>

<!--s-->

## QUEST | Agenda

<div class = "col-wrapper">

<div class="c1" style = "width: 70%">

1. What is QUEST?
2. Why Use QUEST?
3. Accessing QUEST
4. Running Jobs on QUEST

</div>

<div class="c2" style = "width: 30%">

<img src="https://storage.googleapis.com/slide_assets/quest.png" width="100%">

</div>

</div>

<!--s-->

## What is QUEST?

**QUEST** is Northwestern's high-performance computing cluster. It is a shared resource that is available to all Northwestern researchers. We have been given an allocation on QUEST to run computationally intensive tasks, such as training deep learning models.

<!--s-->

## Why Use QUEST?

QUEST provides several benefits for running computationally intensive tasks:

1. **High Performance**: QUEST is designed to handle large-scale computations efficiently.
2. **Resource Sharing**: QUEST is a shared resource, allowing multiple users to run jobs simultaneously.
3. **Job Scheduling**: QUEST uses the **slurm** job scheduler to manage job submissions and resource allocation. 🔥
4. **Singularity Containers**: QUEST supports Singularity containers, which allow you to run applications in isolated environments.
  - Note: Docker containers are easily & automatically converted to Singularity containers on Quest.

Most importantly, getting used to running your training jobs in containers on a high-performance computing cluster is a valuable skill that will serve you well in your research and industry careers.

<!--s-->

## Accessing QUEST

QUEST is best accessed via SSH. SSH (Secure Shell) is a network protocol that allows you to securely connect to a remote computer. You can use an SSH client to connect to QUEST from your local machine.

To access QUEST, you will need to use your NetID and password. You can access QUEST from any computer with an internet connection.

<!--s-->

## Running Jobs on QUEST

To run a job on QUEST, you will need to:

1. Write a <span class="code-span">slurm script</span> that specifies the job requirements.
2. Submit the job using the <span class="code-span">sbatch</span> command.
3. Monitor the job using the <span class="code-span">squeue</span> command.

The <span class="code-span">slurm script</span> specifies the resources required for the job, such as the number of CPUs, memory, and runtime. The <span class="code-span">sbatch</span> command submits the job to the slurm scheduler, which allocates the necessary resources and runs the job.

<!--s-->

## QUEST | Slurm Script Example

Here is an example of a <span class="code-span">slurm script</span> that specifies the resources required for a job:

```bash

#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=my_job.out
#SBATCH --error=my_job.err

# Pull docker image
singularity pull docker://continuumio/miniconda3

# Run the container
singularity exec miniconda3_latest.sif python my_script.py

```

This script requests 1 node, 1 task, 4 CPUs per task, 8GB of memory, and a runtime of 1 hour. It runs a Python script called <span class="code-span">my_script.py</span> in a Singularity container based on the Miniconda3 Docker image.

<!--s-->

## QUEST | Running a Job

To submit a job on QUEST, you can use the <span class="code-span">sbatch</span> command with the <span class="code-span">slurm script</span> as an argument:

```bash

sbatch my_job.sh

```

This command submits the job to the slurm scheduler, which allocates the necessary resources and runs the job. You can monitor the job using the <span class="code-span">squeue</span> command.

<!--s-->

## QUEST | Monitoring a Job

To monitor a job on QUEST, you can use the <span class="code-span">squeue</span> command:

```bash

squeue -u your_netid

```

This command displays information about the jobs you have submitted, including the job ID, name, status, and runtime. You can use this information to track the progress of your job and troubleshoot any issues that arise.

<!--s-->

<div class="header-slide">

# Questions?

</div>

<!--s-->






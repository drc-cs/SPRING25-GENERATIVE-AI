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
  ## L.02 | Machine Learning Operations

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
  ## Please check in by entering the code on the chalkboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs.github.io/WINTER25-CS326/lectures/index.html?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

Objectives: Distributed data preprocessing (SnowFlake), containerized training (Docker), hyperparameter tuning and distributed training strategies

Agenda:

1. Welcome!
2. Distributed Data Preprocessing
    a. SnowFlake
    b. Data Pipelines
    c. Data Versioning
3. Containerized Training
    a. Docker
    b. Running your Models
    c. Containerization Best Practices
4. Hyperparameter Tuning Strategies
    a. Grid Search
    b. Random Search
    c. Bayesian Optimization
    d. Hyperband
5. Distributed Training Strategies
    a. Data Parallelism
    b. Model Parallelism
    c. Synchronous vs. Asynchronous Training
    d. Gradient Accumulation
    e. Gradient Compression
    f. Model Sharding
    g. Pipeline Parallelism
    h. Hybrid Parallelism
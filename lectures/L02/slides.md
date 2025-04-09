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
  ## L.02 | MLOps Essentials

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 30%">

  <iframe src="https://lottie.host/embed/e7eb235d-f490-4ce1-877a-99114b96ff60/OFTqzm1m09.json" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Welcome to Generative AI.
  ## Please check in by entering the provided code.

  </div>
  </div>

  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 50%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Enter Code" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with MLOps?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

# Agenda 


<div class = "col-wrapper" style = "font-size: 0.8em">
<div class="c1" style = "width: 50%">

## 1. Data Preprocessing
### Local Solutions / Python Multiprocessing
### Remote Solutions / Columnar DBs

## 2. Containerized Training
### Using pre-built containers
### Building your own container
### Running on SLURM

## 3. Hyperparameter Tuning Strategies
### Grid Search
### Random Search
### Hyperband

</div>
<div class="c2" style = "width: 50%">


## 4. Distributed Training Strategies
### Data Parallelism
### Model Parallelism

## 5. Monitoring and Logging
### Monitoring with TensorBoard
### Monitoring with Weights & Biases

## 6. Project 1 (Image Generation)


</div>
</div>

<!--s-->

<div class="header-slide">

# Parallel & Distributed Data Preprocessing

</div>

<!--s-->

## Why Separate Data Preprocessing from Model Training?

Separating data preprocessing from model training is a crucial practice in machine learning workflows.

- ### Different Resource Requirements
- ### Scalability
- ### Modularity and Reusability
- ### Improved Debugging and Maintenance
- ### Consistency and Reproducibility

<!--s-->

## Different Resource Requirements

By separating these tasks, we can optimize resource usage and reduce costs.

<div class="col-wrapper">
  <div class="c1" style="width: 50%">
    <h3>Data Preprocessing</h3>
    <p>Often requires parallelization on CPUs due to the nature of tasks like data cleaning, transformation, and augmentation.</p>
  </div>
  <div class="c2" style="width: 50%">
    <h3>Model Training</h3>
    <p>Typically requires GPUs to handle the computationally intensive process of training deep learning models.</p>
  </div>
</div>

<!--s-->

## Scalability

This separation allows each process to be scaled according to its specific needs.

<div class="col-wrapper">
  <div class="c1" style="width: 50%">
    <h3>Data Preprocessing</h3>
    <p>Can be scaled independently using distributed computing frameworks like Apache Spark or Dask.</p>
  </div>

  <div class="c2" style="width: 50%">
    <h3>Model Training</h3>
    <p>Can be scaled using specialized hardware like multiple GPUs with NVLink for faster data transfer.</p>
  </div>
</div>

<!--s-->

## Modularity and Reusability

### Modularity

Separating preprocessing from training allows for the creation of modular pipelines where preprocessing steps can be reused across different models and experiments.

### Reusability

Preprocessed data can be stored and reused, saving time and computational resources in future experiments.

<!--s-->

## Improved Debugging and Maintenance

### Isolation of Issues

By separating preprocessing and training, it becomes easier to isolate and debug issues in each stage.

### Maintainability

Modular code is easier to maintain and update, ensuring that changes in preprocessing do not inadvertently affect model training and vice versa.

<!--s-->

## Consistency and Reproducibility

### Consistent Preprocessing

Ensures that the same preprocessing steps are applied consistently across different training runs, leading to more reproducible results.

### Reproducible Experiments

By storing preprocessed data, experiments can be reproduced exactly, facilitating better comparison and validation of models.

<!--s-->

## Example Workflow

This workflow ensures that each stage is optimized and managed independently, leading to more efficient and effective machine learning pipelines.

<div class="col-wrapper">
  <div class="c1" style="width: 50%">

### Data Preprocessing
  - Load raw data
  - Clean and transform data
  - Store preprocessed data

  </div>
  <div class="c2" style="width: 50%">

### Model Training
- Load preprocessed data
- Train model
- Evaluate model
</div>
</div>

<!--s-->

## L.02 | Q.01

Which of the following is **not** a benefit of separating data preprocessing from model training?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

A. Improved Debugging and Maintenance<br><br>
B. Scalability<br><br>
C. Reduced Model Training Time<br><br>
D. Consistency and Reproducibility

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.01' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Local Data Preprocessing with Python Multiprocessing

</div>

<!--s-->

## Local Data Preprocessing with Python Multiprocessing
Fast local data preprocessing is essential for efficient machine learning workflows. It allows for quick data preparation, enabling faster iterations and experimentation. One of the key tools for achieving this in Python is the built-in <span class="code-span">multiprocessing</span> module.

### Benefits of Fast Local Data Preprocessing

- **Reduced Overall Iteration Time**: Efficient preprocessing reduces the overall time required for the data & model lifecycle.
- **Increased Productivity**: Faster data preparation allows for more iterations and experiments within the same timeframe.
- **Resource Optimization**: Utilizes local CPU resources effectively, reducing the need for expensive cloud-based solutions.

<!--s-->

## Python's <span class="code-span">multiprocessing</span> Module

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

The <span class="code-span">multiprocessing</span> module in Python provides a simple and powerful way to parallelize data preprocessing tasks across multiple CPU cores. This can significantly speed up the preprocessing pipeline.

</div>
<div class="c2" style = "width: 30%; text-align: center;">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/640px-Python-logo-notext.svg.png" style="border-radius: 10px;">
<p style="font-size: 0.6em; color: grey;">Python</p>

</div>
</div>

<!--s-->

## Key Features of <span class="code-span">multiprocessing</span>

<div class = "col-wrapper">
<div class="c1" style = "width: 60%">

- **Process-based Parallelism**: Leverages multiple processes to bypass the Global Interpreter Lock (GIL) and achieve true parallelism.
- **Simple API**: Easy to use with constructs like <span class="code-span">Pool</span>, <span class="code-span">Process</span>, and <span class="code-span">Queue</span>.
- **Scalability**: Can scale with the number of available CPU cores, making it suitable for both small and large datasets.

</div>
<div class="c2" style = "width: 30%; text-align: center;">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/640px-Python-logo-notext.svg.png" style="border-radius: 10px;">
<p style="font-size: 0.6em; color: grey;">Python</p>

</div>
</div>

<!--s-->

## Example: Using <span class="code-span">multiprocessing</span> for Data Preprocessing

```python
import multiprocessing as mp
import pandas as pd

# Function to preprocess a chunk of data
def preprocess_chunk(chunk):
  # Example preprocessing steps
  chunk['processed'] = chunk['raw'].apply(lambda x: x * 2)
  return chunk

# Function to split data into chunks and process in parallel
def parallel_preprocess(data, num_chunks):
  chunks = np.array_split(data, num_chunks)
  pool = mp.Pool(processes=num_chunks)
  processed_chunks = pool.map(preprocess_chunk, chunks)
  pool.close()
  pool.join()
  return pd.concat(processed_chunks)

# Example usage
if __name__ == '__main__':
  data = pd.DataFrame({'raw': list(range(1000000))})
  num_chunks = mp.cpu_count()
  processed_data = parallel_preprocess(data, num_chunks)
  print(processed_data.head())
```

<!--s-->

## Best Practices for Using <span class="code-span">multiprocessing</span>


<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Chunk Size
Choose an appropriate chunk size to balance between parallelism and the overhead of process management.

### Resource Monitoring
Monitor CPU and memory usage to avoid overloading the system.

</div>
<div class="c2" style = "width: 50%">

### Error Handling
Implement robust error handling to manage exceptions in parallel processes.

### Profiling
Profile the preprocessing pipeline to identify and address bottlenecks.

</div>
</div>



<!--s-->

## Other ~ Local Approaches

### CPUs
- **Dask**: A flexible parallel computing library that provides parallelized data structures and task scheduling.
- **Joblib**: A library that provides simple tools for parallelizing Python functions using <span class="code-span">multiprocessing</span>.
- **PySpark**: A distributed computing framework that provides parallel data processing using Apache Spark, but can be used locally as well.

### CPUs and GPUs
- **Ray**: A distributed computing framework that enables parallel and distributed Python applications. 🔥
- **Numba**: A just-in-time compiler that accelerates Python functions using the LLVM compiler infrastructure.
- **Jax**: A library for numerical computing that provides automatic differentiation and GPU/TPU acceleration.

<!--s-->

## L.02 | Q.02

Let's say you have a dataset with 9,000 rows. You want to preprocess this data using Python's <span class="code-span">multiprocessing</span> module. Your machine has 8 CPU cores. What is the **max** number of chunks you should split the data into? Ignore any OS or application overhead.

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

A. 6<br><br>
B. 8<br><br>
C. 9<br><br>
D. 10<br><br>

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.02' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Remote Data Preprocessing with Columnar Databases (OLAP)

</div>

<!--s-->

## Columnar Databases

**Columnar Databases** (as opposed to traditional OLTP systems) store data tables primarily by column rather than row. This storage approach is ideal for OLAP scenarios as it dramatically speeds up the data manipulation and retrieval process.

<div class="col-wrapper col-centered">
<img src = "https://storage.googleapis.com/gweb-cloudblog-publish/images/BigQuery_Explained_storage_options_2.max-700x700.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Thallum, 2020</p>
</div>

<!--s-->

## Why Column-Based Databases for OLAP?

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Row-based databases

Row-based databases store data in rows, which is efficient for transactional workloads but can be inefficient for analytical queries that often require scanning large amounts of data across multiple rows.

### Column-based databases

Column-based databases provide faster data retrieval and more effective data compression than traditional row-oriented databases, especially suited for read-oriented tasks.

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src = "https://storage.googleapis.com/gweb-cloudblog-publish/images/BigQuery_Explained_storage_options_2.max-700x700.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Thallum, 2020</p>

</div>
</div>

<!--s-->

## Why Column-Based Databases for OLAP?

### Advantages of Columnar Storage

1. **Faster Query Performance**: Only the necessary columns are read, reducing I/O operations.

2. **Better Compression**: Similar data types are stored together, allowing for more efficient compression algorithms.

3. **Improved Analytics**: Columnar storage is optimized for analytical queries, making it easier to perform aggregations and calculations. This is critical for data preprocessing.

4. **Scalability**: Columnar databases can handle large volumes of data and scale horizontally by adding more nodes to the cluster.

<!--s-->

## Sidebar: Advances that make Columnar DBs feasible

1. **Data Compression**: Columnar databases may employ techniques such as Run Length Encoding (where repetitions of a value are condensed into a single entry with a count) and Dictionary Encoding (which uses a dictionary of unique values and indices) to conserve storage space and boost query efficiency.

2. **Vectorized Execution**: Vectorized execution refers to the ability to process data in batches or vectors rather than one row at a time. This approach takes advantage of modern CPU architectures and SIMD capabilities, leading to significant performance improvements.

3. **SIMD (Single Instruction, Multiple Data)**: SIMD refers to the hardware architecture that allows for the simultaneous execution of a single operation on multiple data points. This capability is essential for parallel processing in columnar storage, enabling high performance and efficiency.

<!--s-->

## Cloud-based Columnar Data Warehouse Services

<div class = "col-wrapper" style="font-size: 0.8em">
<div class="c1" style = "width: 50%">

### AWS Redshift

- Uses columnar storage, which significantly enhances query performance for analytical workloads.
- Employs massively parallel processing (MPP) to distribute and accelerate query execution across multiple nodes.
- Leverages optimized data compression techniques to reduce storage footprint and improve I/O efficiency.

### GCP BigQuery

- Serverless architecture eliminates infrastructure management overhead, allowing users to focus on data analysis.
- Highly scalable, capable of handling petabyte-scale datasets and dynamically adjusting resources.
- Cost-effective, with a pay-as-you-go pricing model based on query processing and storage usage.
- Strong integration with other GCP services.

</div>
<div class="c2" style = "width: 50%">

### Snowflake

- Unique architecture with separate compute and storage layers, enabling independent scaling and optimization.
- Provides elastic performance, allowing for dynamic resource allocation to handle varying workloads.
- Offers data sharing capabilities.

</div>
</div>



<!--s-->

## Example: Columnar DB Data Preprocessing
### Standardizing a Column using BigQuery

```python
from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Define a query to standardize column1 in a `puppies` table
query = "SELECT ML.STANDARD_SCALER(column1) AS column1_scaled FROM `project.dataset.puppies`"

# Load the query results into a Pandas DataFrame
df = client.query(query).to_dataframe()
```

<!--s-->

## Example: Columnar DB Data Preprocessing
### Standardizing a Column using a Stored Procedure with Snowflake

```text
CREATE OR REPLACE PROCEDURE standardize_salary()
LANGUAGE PYTHON
RUNTIME_VERSION = 3.8
HANDLER = 'main'
AS
$$
import snowflake.connector

def main(session):
    try:
        # Calculate mean and standard deviation
        result = session.sql("SELECT AVG(salary), STDDEV(salary) FROM employees").collect()
        mean_salary, stddev_salary = result[0]

        if stddev_salary == 0:
            return "Standard deviation is zero. Salaries are identical. No standardization performed."

        # Add the standardized_salary column if it doesn't exist
        session.sql("ALTER TABLE employees ADD COLUMN IF NOT EXISTS standardized_salary FLOAT").collect()

        # Update standardized_salary column
        update_query = f"""
        UPDATE employees 
        SET standardized_salary = (salary - {mean_salary}) / {stddev_salary}
        """
        session.sql(update_query).collect()

        return "Salary standardization completed. New standardized_salary column added."

    except Exception as e:
        return f"Error: {str(e)}"
$$;
```

```sql
CALL standardize_salary();
```


<!--s-->

<div class="header-slide">

# Containerized Training

</div>

<!--s-->

<div class="header-slide">

## Using Pre-built Containers for Deep Learning Training

</div>

<!--s-->


## Examples of Pre-built Containers

These images come with popular frameworks like TensorFlow, PyTorch, and Jax/Flax with CUDA support.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### AWS

AWS provides [Docker images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) optimized for deep learning workloads on AWS.

```bash
# Pulling a TensorFlow container from AWS.
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-cpu-py310-ubuntu22.04-ec2
```

### GCP

Google Cloud offers [Docker images](https://cloud.google.com/deep-learning-containers/docs/getting-started-local) for deep learning, integrated with Google Cloud services.

```bash
# Pulling a PyTorch container from Google Cloud
docker pull us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310
```

</div>
<div class="c2" style = "width: 50%; margin-right: 2em;">

### DockerHub

DockerHub hosts a variety of official deep learning containers.

```bash
# Pulling an official TensorFlow container from DockerHub
docker pull tensorflow/tensorflow:latest-gpu
```

</div>
</div>

<!--s-->

## Pre-built Containers | Pros and Cons

| Pros | Cons | 
| --- | --- |
| Quick setup and deployment | Limited customization options |
| Consistent and reproducible environments | Potential security vulnerabilities |
| Pre-configured with popular frameworks and libraries | Dependency on container provider for updates |
| Simplifies dependency management | Performance overhead compared to bare-metal setups |
| Reduces setup errors and conflicts | May include unnecessary components, increasing image size |
| May have built-in operations for various cloud platforms | |

<!--s-->

## Best Practices for using Pre-built Containers

- **Regular Updates**: Ensure containers are regularly updated to include the latest security patches and framework versions.
- **Customization**: Use pre-built containers as a base and customize them to fit specific needs.
- **Resource Management**: Monitor resource usage and optimize container configurations for better performance.
- **Security**: Scan containers for vulnerabilities and follow best practices for container security.

<!--s-->

## Building Your Own Container.

To build your own container, you need to create a Dockerfile that specifies the base image, dependencies, and commands to run. We covered some basic Docker in L.01. Here's an example:

```Dockerfile
# Use an official TensorFlow image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run the train.py script when the container launches
CMD ["python", "train.py"]
```

### Build the Docker image

```bash
docker build -t my_tensorflow_image .
```
### Run the Docker container

```bash
docker run -t my_tensorflow_image
```

<!--s-->

## Building your own container | Pros and Cons

| Pros | Cons |
| --- | --- |
| Full control over the environment | Requires more effort and expertise |
| Customizable to specific requirements | May introduce errors or conflicts |
| Can include only necessary components | Time-consuming to set up and maintain |
| Greater flexibility for optimization | Security vulnerabilities if not properly configured |

<!--s-->

## Running on SLURM

SLURM (Simple Linux Utility for Resource Management) is a popular job scheduler used in high-performance computing (HPC) environments. As a reminder, SLURM allows you to submit and manage jobs on a cluster of compute nodes. We have access to the Northwestern Quest cluster, which uses SLURM.

<!--s-->

## Running on SLURM | Example

Here's an example of how you can run a deep learning training job within a pre-built container on a SLURM cluster. 

⚠️ ⚠️ This script was built using information at this [link](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1748), but it isn't working as of April 2025. If you want to run a job on SLURM, I recommend the [conda approach](https://drc-cs.github.io/SPRING25-GENERATIVE-AI/lectures/L01/#/68) we covered in L01.

```bash
#!/bin/bash
#SBATCH --account=p32562
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --output=/projects/p32562/gpu-run.log

# Load the required modules.
module load singularity

# Use a temporary directory when pulling container image
export SINGULARITY_CACHEDIR=$TMPDIR

# Pull the TensorFlow container.
singularity pull docker://pytorch/pytorch:latest

# Run your torch script inside the container.
singularity exec pytorch_latest.sif python -c "import torch; print('Num GPUs Available:', torch.cuda.device_count())"

```


<!--s-->

<div class="header-slide">

# Hyperparameter Tuning Strategies

</div>

<!--s-->

## Hyperparameter Tuning Strategies

Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance.

We will cover 3 strategies in detail:

1. Grid Search
2. Random Search
3. Hyperband

<!--s-->

## Goal

The goal of hyperparameter tuning is to find the optimal set of hyperparameters that minimize the model's loss on a validation set.

`$$ \text{argmin}_{\theta} \sum_{i=1}^{n} L(y_i, f(x_i, \theta)) $$`

where:

- $ \theta $ is the hyperparameter vector
- $ y_i $ is the true label
- $ f(x_i, \theta) $ is the predicted label
- $ L $ is the loss function
- $ n $ is the number of samples

<!--s-->

## Grid Search

Grid search is a brute-force approach to hyperparameter tuning. It involves defining a grid of hyperparameter values and evaluating the model's performance for each combination of hyperparameters.

```python
for learning_rate in [0.01, 0.1, 1]:
    for batch_size in [16, 32, 64]:
        model = create_model(learning_rate, batch_size)
        model.fit(X_train, y_train)
        score = model.evaluate(X_val, y_val)
        store_results(learning_rate, batch_size, score)
```

<!--s-->

## Grid Search

| Pros | Cons |
| --- | --- |
| Simple to implement | Computationally expensive |
| Easy to understand | Not suitable for large hyperparameter spaces |
| Guarantees finding the optimal hyperparameters | Can be inefficient if the grid is not well-defined |

<!--s-->

## Random Search

Random search is a more efficient alternative to grid search. Instead of evaluating all combinations of hyperparameters, it randomly samples a fixed number of combinations and evaluates their performance.

Random search has been shown to be more efficient than grid search in practice, especially for large hyperparameter spaces [[citation]](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

```python
# Create random hyperparameter combinations.
hyperparameter_space = {
    'learning_rate': [0.01, 0.1, 1],
    'batch_size': [16, 32, 64, 128, 256]
}

random_combinations = random.sample(list(itertools.product(*hyperparameter_space.values())), k=10)

for learning_rate, batch_size in random_combinations:
    model = create_model(learning_rate, batch_size)
    model.fit(X_train, y_train)
    score = model.evaluate(X_val, y_val)
    store_results(learning_rate, batch_size, score)
```


<!--s-->

# Random Search


| Pros | Cons |
| --- | --- |
| More efficient than grid search | No guarantee of finding the optimal hyperparameters |
| Can be parallelized | Randomness can lead to inconsistent results |
| Suitable for large hyperparameter spaces | Requires careful selection of the number of samples |
| Can find good hyperparameters even with a small number of evaluations | |

<!--s-->

## Hyperband

Hyperband is a more advanced hyperparameter tuning algorithm that combines random search with early stopping. It is based on the idea of running multiple random search trials in parallel and stopping the least promising trials early.

This smart resource allocation strategy allows Hyperband to find good hyperparameters with fewer evaluations than random search. The authors indicate it is also superior to Bayesian optimization [[citation]](https://arxiv.org/abs/1603.06560).


```python

from keras_tuner import Hyperband

tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='helloworld',

)

tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
```

<!--s-->

## Hyperband Pseudocode

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/hyperband.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Li 2018</p>
</div>

<!--s-->

## Hyperband Explanation

$n$ is the number of configurations to evaluate in each bracket. It ensures that the total budget $B$ is distributed across all brackets, taking into account the reduction factor $η$ and the number of brackets $s$.

**Number of configurations (n):**
$$ n = \left\lceil \frac{B}{R} \cdot \frac{\eta^s}{s + 1} \right\rceil $$

Where:
  - $B$ is the total budget
  - $R$ is the resources allocated to each configuration
  - $s$ is the number of brackets
  - $η$ is the reduction factor (e.g., 3)

<!--s-->

## Hyperband Explanation

This formula determines the initial amount of resources allocated to each configuration in a given bracket. As $s$ decreases, the resources per configuration increase.

**Resources per configuration (r):**
$$ r = R \cdot \eta^{-s} $$

Where:
  - $R$ is the total budget
  - $s$ is the number of brackets
  - $η$ is the reduction factor (e.g., 3)
  
<!--s-->

## Hyperband Example

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/hyperband.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Li 2018</p>
</div>

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/hyperband_example.png' style='border-radius: 10px;'>
</div>

</div>
</div>


<!--s-->

## Hyperband

| Pros | Cons |
| --- | --- |
| Efficient resource allocation | Requires hyperparameter tuning |
| Combines exploration and exploitation | Performance depends on the choice of initial configurations |
| Suitable for large hyperparameter spaces | May not always find the global optimum |
| Can find good hyperparameters with fewer evaluations compared to random search | |

<!--s-->

## Other Hyperparameter Tuning Strategies


| Strategy | Description |
| --- | --- |
| Bayesian Optimization | Uses probabilistic models to find the optimal hyperparameters by balancing exploration and exploitation. |
| Genetic Algorithms | Uses evolutionary algorithms to optimize hyperparameters by mimicking natural selection processes. |
| Tree-structured Parzen Estimator (TPE) | A Bayesian optimization method that models the distribution of good and bad hyperparameters separately. |
| Reinforcement Learning | Uses reinforcement learning algorithms to optimize hyperparameters by treating the tuning process as a sequential decision-making problem.

<!--s-->

## L.02 | Q.03

Let's say you have a deep learning model with many hyperparameters and an expensive fitness score (e.g., long training time). Which hyperparameter tuning strategy would you choose?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

A. Grid Search<br><br>
B. Random Search<br><br>
C. Hyperband<br><br>

</div>

<div class='c2' style = 'width: 50%;'>

<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.03' width = '100%' height = '100%'></iframe>

</div>

<!--s-->

<div class="header-slide">

# Distributed Training Strategies

</div>

<!--s-->

<div class="header-slide">

## Data Parallelism

</div>

<!--s-->

## Data Parallelism

Data parallelism is a technique used to distribute the training of a machine learning model across multiple devices, such as GPUs or CPUs. The main idea is to split the training data into smaller batches and process them in parallel on different devices.

<div style="text-align: center;">
  <img src="https://storage.googleapis.com/slide_assets/parallelism.png" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Scalar Topics (2023)</p>
</div>

<!--s-->

## Advantages and Challenges of Data Parallelism

| Advantages | Challenges |
| --- | --- |
| Scalability (can leverage multiple devices) | Communication overhead (synchronizing gradients) |
| Fault tolerance (if one device fails, others can continue) | Load balancing (ensuring equal workload across devices) |
| Flexibility (can use different devices) | Complexity (requires careful implementation) |
| Improved training speed | Memory constraints (limited by the smallest device) |


<!--s-->

## Pseudocode for Data Parallelism

```
1. Initialize:
  - D: training dataset
  - M: model
  - N: number of devices

2. Split D into N subsets: D1, D2, ..., DN

3. For each device i in {1, 2, ..., N}:
  - Copy model M to device i
  - Train model Mi on data subset Di
  - Compute gradients Gi

4. Aggregate gradients: G = (G1 + G2 + ... + GN) / N

5. Update global model M using aggregated gradients G
```

<!--s-->

## Data Parallelism in Practice

**TensorFlow**: Offers data parallelism with <span class = 'code-span'>tf.distribute.Strategy </span>.
<br>**PyTorch**: Offers data parallelism through <span class = 'code-span'>torch.nn.parallel.DistributedDataParallel</span>.

### TensorFlow Example

```python
import tensorflow as tf

# Define model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Define strategy
    strategy = tf.distribute.MirroredStrategy()

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

    # Compile and train model within strategy scope
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(dataset, epochs=10)
```

<!--s-->

## Data Parallelism | Best Practices

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Batch Size
Choose an appropriate batch size to balance between computation and communication. For example, a larger batch size can reduce the number of communication rounds.

### Synchronization
Use efficient synchronization techniques to minimize communication overhead. For example, using gradient compression.

</div>
<div class="c2" style = "width: 50%">

### Profiling
Profile the training process to identify and address bottlenecks. For example, use TensorBoard to visualize the training process and identify slow operations.

</div>
</div>


<!--s-->

<div class="header-slide">

# Model Parallelism

</div>

<!--s-->

## Model Parallelism

Model parallelism is a technique used to distribute the training of a machine learning model across multiple devices by splitting the model itself, rather than the data.

<div style="text-align: center;">
  <img src="https://storage.googleapis.com/slide_assets/parallelism.png" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Scalar Topics (2023)</p>
</div>

<!--s-->

## Types of Model Parallelism

There are several methods to achieve model parallelism.

<div class = "col-wrapper" style = "font-size: 0.8em;">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Layer-wise Parallelism
Splitting model layers across different devices. For example, one group of layers might be placed on one GPU, while the subsequent layers are placed on another. This is often called naive model parallelism, and results in idle GPUs.

### Pipeline Parallelism
Distributes the model across multiple devices in a pipeline fashion, where each device processes a different stage of the model. Through the use of micro-batching, this can help reduce idle time.

</div>
<div class="c2" style = "width: 50%">

### Tensor Parallelism
Distributes individual tensors across multiple devices, often by splitting the tensors themselves (e.g., slices or chunks of matrices/vectors). Commonly used for distributing large matrices involved in operations like matrix multiplication or transformation (e.g., in transformer models).

</div>
</div>


<!--s-->

## Layer-Wise Parallelism vs Pipeline Parallelism Resource Usage

On the top we can see an implementation of layer-wise parallelism, where the GPUs are idle while waiting for the previous GPU to finish processing. 

On the bottom we can see an implementation of pipeline parallelism, where the GPUs are used more efficiently through the use of micro-batching.

<div style="text-align: center;">
  <img src="https://1.bp.blogspot.com/-fXZxDPKaEaw/XHlt7OEoMtI/AAAAAAAAD0I/hYM_6uq2BTwaunHZRxWd7JUJV43fEysvACLcBGAs/s640/image2.png" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Google Research (2019)</p>
</div>

<!--s-->

## Pseudocode for Model Parallelism

```text
1. Initialize:
  - M: model
  - N: number of devices

2. Partition M into N parts: M1, M2, ..., MN

3. Assign each part Mi to device i

4. Forward pass:
  - For each device i in {1, 2, ..., N}:
    - Compute forward pass for Mi
    - Send intermediate results to device i+1

5. Backward pass:
  - For each device i in {N, N-1, ..., 1}:
    - Compute gradients for Mi
    - Send gradients to device i-1

6. Aggregate gradients and update model parameters
```

<!--s-->

## Advantages and Challenges of Model Parallelism

| Advantages | Challenges |
| --- | --- |
| Memory Efficiency: Allows training of very large models that do not fit into the memory of a single device. | Complexity: More complex to implement compared to data parallelism. |
| Scalability: Can leverage multiple devices to speed up training. | Communication Overhead: Requires efficient communication of intermediate results between devices. |
| | Load Balancing: Ensuring that each device has an equal amount of work can be challenging. |

<!--s-->

## Model Parallelism in Practice

The implementation of model parallelism can vary significantly depending on the framework used. Below is an example of model parallelism in TensorFlow.

### Example (TensorFlow)

```python
import tensorflow as tf
import numpy as np

# Define model layers
class ModelPart1(tf.keras.layers.Layer):
  def __init__(self):
    super(ModelPart1, self).__init__() # init parent class.
    self.dense1 = tf.keras.layers.Dense(128, activation='relu')

  def call(self, inputs):
    return self.dense1(inputs)

class ModelPart2(tf.keras.layers.Layer):
  def __init__(self):
    super(ModelPart2, self).__init__()
    self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

  def call(self, inputs):
    return self.dense2(inputs)

# Create model parts and place them on different devices
with tf.device('/GPU:0'):
  model_part1 = ModelPart1()

with tf.device('/GPU:1'):
  model_part2 = ModelPart2()

# Define the full model
class ParallelModel(tf.keras.Model):
  def __init__(self, model_part1, model_part2):
    super(ParallelModel, self).__init__()
    self.model_part1 = model_part1
    self.model_part2 = model_part2

  def call(self, inputs):
    x = self.model_part1(inputs)
    return self.model_part2(x)

# Create the parallel model
parallel_model = ParallelModel(model_part1, model_part2)

# Compile the model
parallel_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create dummy data
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))

# Train the model
parallel_model.fit(x_train, y_train, epochs=10, batch_size=32)
```

<!--s-->

## Best Practices for Model Parallelism

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Partitioning

Carefully partition the model to balance the workload across devices -- you don't want one device to be idle while another is overloaded.

### Communication

Optimize communication between devices to minimize overhead -- this is done through efficient data transfer and synchronization techniques. Hardware accelerators like NVIDIA NVLink can help.

</div>
<div class="c2" style = "width: 50%">

### Profiling

Profile the training process to identify and address bottlenecks -- this can be done using tools like TensorBoard or PyTorch Profiler.

</div>
</div>

<!--s-->

<div class="header-slide">

# Monitoring and Logging

</div>

<!--s-->

## TensorBoard

TensorBoard is a powerful visualization tool for TensorFlow that allows you to monitor and visualize various aspects of your machine learning model during training. It provides a suite of tools to help you understand, debug, and optimize your model.

<div style="text-align: center;">
  <img src="https://www.tensorflow.org/static/tensorboard/images/tensorboard.gif" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">TensorBoard (2023)</p>
</div>

<!--s-->

## TensorBoard | Example

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
...

# Define model
def create_model():
    ...
    return model

# Create TensorBoard callback
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile and train model
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

```

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src = "https://www.tensorflow.org/static/tensorboard/images/tensorboard.gif" width = "100%" style="border-radius: 10px;">
<p style = "font-size: 0.6em; color: grey;">TensorBoard (2025)</p>

</div>
</div>

<!--s-->

## L.02 | Q.04

The Tensorflow Profiler is a tool that helps you analyze the performance of your TensorFlow models. It provides insights into the execution time of different operations, memory usage, and other performance metrics.

What's the bottleneck here?

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 60%; justify-content: start;">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/profile.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>TensorFlow 2025</p>
</div>

</div>
<div class="c2" style = "width: 40%; justify-content: start; padding-bottom: 20%;">

<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.04' width = '100%' height = '100%'></iframe>

</div>
</div>

<!--s-->

## TensorBoard | Features

| Feature | Description | Why is it useful? |
| --- | --- | --- |
| Scalars | Visualize scalar values (e.g., loss, accuracy) over time | Helps track model performance and identify issues |
| Histograms | Visualize the distribution of weights and biases | Helps understand model behavior and identify overfitting / regularization issues |
| Graphs | Visualize the computation graph of the model | Helps understand model architecture and identify bottlenecks |
| Embeddings | Visualize high-dimensional data in lower dimensions | Helps understand data distribution and clustering |
| Images | Visualize images during training | Helps track data augmentation and preprocessing |
| Text | Visualize text data during training | Helps track data preprocessing and augmentation |


<!--s-->

## Weights & Biases

Weights & Biases (W&B) is a popular tool for experiment tracking, model management, and collaboration in machine learning projects. It provides a suite of tools to help you visualize, compare, and share your machine learning experiments.

<div style="text-align: center;">
  <img src="https://help.ovhcloud.com/public_cloud-ai_machine_learning-notebook_tuto_03_weight_biases-images-overview_wandb.png" width="800%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Weights & Biases (2023)</p>
</div>

<!--s-->

## Weights & Biases | Example

Weights & Biases works through a simple REST API that integrates with popular machine learning frameworks like TensorFlow, PyTorch, and Flax. W&B is excellent because of how flexible it is.

<div class = "col-wrapper">

<div class="c1 col-centered" style = "width: 50%; justify-content: start;">

```python
import wandb

wandb.init(config=args)

model = ...  # set up your model

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        wandb.log({"loss": loss})
```
<p style="font-size: 0.6em; color: grey;">Weights & Biases (2025)</p>

</div>
<div class="c2 col-centered" style = "width: 50%; justify-content: start;">

<div style="text-align: center;">
  <img src="https://help.ovhcloud.com/public_cloud-ai_machine_learning-notebook_tuto_03_weight_biases-images-overview_wandb.png" width="100%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Weights & Biases (2023)</p>
</div>

</div>
</div>

<!--s-->

## Weights & Biases | Features

| Feature | Description | Why is it useful? |
| --- | --- | --- |
| Experiment Tracking | Track and visualize experiments, hyperparameters, and metrics | Helps compare and analyze different experiments |
| Model Management | Version control for models and datasets | Helps manage and share models |
| Collaboration | Share experiments and results with team members | Facilitates collaboration and knowledge sharing |
| Integration | Integrates with popular machine learning frameworks | Easy to use with existing projects |

<!--s-->

## Summary

<div class="col-wrapper" style = "font-size: 0.75em">
<div class="c1" style="width: 50%; margin-right: 2em;">

### Data Preprocessing
**Fast Local Preprocessing**: Utilize Python's <span class='code-span'>multiprocessing</span> module for efficient local data preprocessing.
**Remote Preprocessing**: Leverage columnar databases (OLAP) for scalable and efficient remote data preprocessing.

### Model Training
**Pre-built Containers**: Simplify training with pre-configured containers for deep learning.
**Custom Containers**: Build tailored containers to meet specific training requirements.

### Hyperparameter Tuning
**Grid Search**: Exhaustive search over a predefined hyperparameter grid.
**Random Search**: Efficient sampling of hyperparameter combinations.
**Hyperband**: Combines random search with early stopping for resource-efficient tuning.

</div>

<div class="c2" style="width: 50%;">

### Distributed Training
**Data Parallelism**: Split data across devices to accelerate training.
**Model Parallelism**: Distribute model components across devices for memory efficiency.

### Monitoring and Logging
**TensorBoard**: Visualize training metrics, model graphs, and more.
**Weights & Biases**: Track experiments, manage models, and collaborate effectively.  

</div>

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with MLOps?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Project 1 | Image Generation

</div>

<!--s-->

## Projects

The course will have two projects, each focusing on a different aspect of generative AI. The first project will focus on **image generation**, while the second project will focus on **text generation**. Both projects will have a proposal and a final report.

We will vote on the best projects at the end of the quarter, and the top-voted projects will be presented on our last day of class. These top-voted projects get a large curve on their project grade.

<!--s-->

## Project Grading

Gen AI is rapidly evolving, and companies that implement a little extra effort are being rewarded. In that spirit, we will have **baseline project criteria** (85%) and **extra project criteria** (15%).

Baseline project criteria includes clean and reproducible code, as well as general ambition and quality. Perfect code and meeting the baseline criteria (a simple generative model that produces images, for example) will earn you an 85% on the project.

To get the extra 15%, additional project criteria is outlined on the next slide.

<!--s-->

## Project Grading | Extra Project Criteria

In your project proposal, please outline any additional features or improvements you plan to implement that go beyond the baseline project criteria. This could include:

### Image Generation (Project 1)

1. ML operations
    - Distributed preprocessing or training pipelines
    - Metrics training & evaluation tracking
2. Hyperparameter tuning strategies
3. Creative latent space exploration
4. Gallery GUI

<!--s-->

## Project Proposal

1. Title -- A short, clean title to identify your project.
2. Images source -- Where are you getting your original images? What are you trying to generate?
3. Model architecture(s) -- What model architecture do you plan to use? Please choose from AE/VAE, GAN, or diffusion architectures.
4. Extra Criteria -- What "Extra Criteria" are you pursuing? 

<!--s-->

## Project Proposal Example

1. Title -- Flower Generator
2. Images source -- I'll be using the [Oxford 102 Flower Dataset](https://paperswithcode.com/dataset/oxford-102-flower).
3. Model architecture(s) -- Training a diffusion model to generate flowers.
4. Extra Criteria -- Hyperband optimization of parameters & data-parallelism for efficient training.

<!--s-->
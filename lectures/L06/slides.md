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
  ## L.06 | Autoregressive Text

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
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Enter Code" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. Transformer Architectures
  2. Retrieval-Augmented Generation (RAG)
  3. Streamlit GUIs
  4. Model Context Protocol (MCP)

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Announcements

- Project 1 (Image Generation) is due tonight at 11:59PM.
   - Next week (after we have had the chance to review your code) we will do a 1 minute code check with everyone.
   - The process should take ~ 30 minutes at the end of lecture.

- Project 2 (Text Generation) will be due on 05.28.2025 @ 11:59PM.
   - The project will be similar to Project 1, but with a focus on text generation.
   - A proposal will be due next week (**05.14.2025**) at 11:59PM.

<!--s-->

# Midterm Feedback Action Items

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Less Theory, More Applications
Generative AI requires some theoretical background. But, it is a very applied field too. I'll find a better balance for the remaining lectures.

### PDFs of Lecture Slides
You can download lectures as PDFs here. I'll get them uploaded before class.

</div>
<div class="c2" style = "width: 50%">

### Textbook & Code Snippets
We don't have a textbook for this class -- but if we did, Generative Deep Learning by David Foster is a great book! Most of the lecture code & examples come from this book, which you can purchase [here](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/). 

</div>
</div>

<!--s-->

<div class="header-slide">

# Autoregressive Text

</div>

<!--s-->

# Agenda

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

## Part I. 

Text Generation Applications<br><br>
Autoregressive Text Modeling
   - Markov Chains
   - RNN
   - LSTM
   - Transformers

</div>
<div class="c2" style = "width: 50%">

## Part II.

Retrieval-Augmented Generation (RAG)
- Building a RAG chatbot.

Model Context Protocol (MCP)
- Augmenting Claude w/ Plugins.

</div>
</div>

<!--s-->

<div class="header-slide">

# Text Generation Applications

</div>

<!--s-->

## Text Generation Applications

| **Application**       | **Description**   |
|-----------------------|------------------|
| **Story Generation**  | Create coherent and engaging narratives.|
| **Content Creation**  | Automate the creation of articles, blog posts, and news.  |
| **Chatbots**          | Engage users in conversations.             |
| **Code Generation**   | Assist in writing code snippets.           |
| **Summarization**     | Condense long texts into shorter summaries. |
| **Translation**       | Translate text from one language to another.  |
| **Dialogue Systems**  | Enable natural language interactions.      |
| **Creative Writing**  | Generate poetry, lyrics, and other creative content.      |
| ... | ... |

<!--s-->

<div class="header-slide">

# Autoregressive Text Modeling

</div>

<!--s-->

## Autoregressive Text Modeling

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Markov Chains
Markov chains are a simple probabilistic model for text generation. They use the current word to predict the next word based on transition probabilities.

### RNN
Recurrent Neural Networks (RNNs) are a class of neural networks designed for sequential data. They maintain a hidden state that captures information about previous inputs, allowing them to model dependencies over time.
</div>
<div class="c2" style = "width: 50%">

### LSTM
Long Short-Term Memory (LSTM) networks are a type of RNN that aims to address the vanishing gradient problem, allowing them to learn long-term dependencies.

### Transformers
Transformers are a type of neural network architecture that uses self-attention mechanisms to model dependencies between words. They have become the dominant architecture for text generation tasks due to their ability to capture long-range dependencies and parallelize training.
</div>
</div>

<!--s-->

<div class="header-slide">

# Markov Chains

</div>

<!--s-->

## Markov Chains

Markov Chains are probabilistic models used for generating text. By modeling the context of words with historical patterns, Markov chains can simulate text generation processes.

<div class="col-centered">
<img src="https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2F2.bp.blogspot.com%2F-U2fyhOJ7bN8%2FUJsL23oh3zI%2FAAAAAAAADRs%2FwZNWvVR-Jco%2Fs1600%2Ftext-markov.png" width="500" style="margin: 0; padding: 0; display: block; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">awalsh128.blogspot.com</span>
</div>

<!--s-->

## Markov Chains

In the context of text generation, a Markov chain uses a finite set of states (words) and transitions between these states based on probabilities.

### Key Elements

- **States**: Words or tokens in a text.

- **Transition Probabilities**: The probability of moving from one word to another. 

- **Order**: Refers to how many previous states (words) influence the next state.

<!--s-->

## Markov Chains

Consider a simple first-order Markov Chain, which uses the current word to predict the next word.


### Transition Matrix

A transition matrix represents the probabilities of transitioning from each word to possible subsequent words.

### Markov Process (First Order)

$$ P(w_{t+1} | w_t) = P(w_{t+1} | w_t, w_{t-1}, \ldots, w_1) $$

tldr; the probability of the next word depends only on the current word.

<!--s-->

## Markov Chains

Let's say we have the following text:

> "The quick brown fox jumps over the lazy dog."

|  | The | quick | brown | fox | jumps | over | lazy | dog |
|--------------|-------|---------|---------|-------|---------|--------|--------|-------|
| The        | 0.0   | 0.5     | 0.0     | 0.0   | 0.0     | 0.0    | 0.5    | 0.0   |
| quick      | 0.0   | 0.0     | 1.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| brown      | 0.0   | 0.0     | 0.0     | 1.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| fox        | 0.0   | 0.0     | 0.0     | 0.0   | 1.0     | 0.0    | 0.0    | 0.0   |
| jumps      | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 1.0    | 0.0    | 0.0   |
| over       | 1.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| lazy       | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 1.0   |
| dog        | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |

Using these probabilities, you can generate new text by predicting each subsequent word based on the current word.

<!--s-->

## Markov Chains

Increasing the order allows the model to depend on more than one preceding word, creating more coherent and meaningful sentences.

### Second-Order Markov Chain Example

Given bi-grams:

> "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"

The transition probability now depends on pairs of words:

$$ P(w_3 | w_1, w_2) $$

This provides better context for the generated text, but can also reduce flexibility and introduction of new combinations.

<!--s-->

<div class="header-slide">

# RNN

</div>

<!--s-->

## RNN

Recurrent Neural Networks (RNNs) are a class of neural networks designed for sequential data. They maintain a hidden state that captures information about previous inputs, allowing them to model dependencies over time. Back in the day, RNNs were the go-to architecture for text generation.

<div style='text-align: center;'>
   <img src='https://miro.medium.com/v2/resize:fit:1254/format:webp/1*go8PHsPNbbV6qRiwpUQ5BQ.png' style='border-radius: 10px; width: 70%;'>
   <p style='font-size: 0.6em; color: grey;'>Mitall 2019</p>
</div>

<!--s-->

<div class="header-slide">

# LSTM

</div>

<!--s-->

## LSTM

Long Short-Term Memory (LSTM) networks are a type of RNN that aims to address the vanishing gradient problem, allowing them to learn long-term dependencies.

<div class = "col-wrapper">
<div class="c1" style = "width: 40%">

LSTMs use a cell state and three gates (input, forget, and output) to control the flow of information.

</div>
<div class="c2" style = "width: 60%">

<div style='text-align: center;'>
   <img src='https://thorirmar.com/post/insight_into_lstm/featured.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Ingolfsson 2021</p>
</div>

</div>
</div>

<!--s-->

<div class="header-slide">

# Transformers

</div>

<!--s-->

## Transformers 

Transformers are a type of neural network architecture that uses attention mechanisms to model dependencies between words. They have become the dominant architecture for text generation tasks due to their ability to capture long-range dependencies and parallelize training. Transformers were introduced by the Vaswani et al. paper "Attention is All You Need" in 2017.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

In order to understand how transformers work, we need to understand the **attention mechanism**.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center; width: 70%;'>
   <img src='https://storage.googleapis.com/slide_assets/transformer.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Vashwani 2017</p>
</div>

</div>
</div>

<!--s-->

## L.06 | Q.01


<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

Desribe what <span class="code-span">k</span> is in the context of self-attention.

</div>
<div class='c2' style = 'width: 50%;'>

<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.06 | Q.01' width = '100%' height = '100%'></iframe>

</div>
</div>

<!--s-->

## Transformers | Self-Attention

Self-attention is a mechanism that allows the model to weigh the importance of different words in a sequence when generating text. It computes attention scores for each word based on its relationship with other words in the sequence. It does this by computing three vectors for each word: the query (q), key (k), and value (v) vectors.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

**Query (q)**: Represents the word for which we are computing attention.

**Key (k)**: Represents the words we are attending to.

**Value (v)**: Represents the information we want to extract from the attended words.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://www.mdpi.com/applsci/applsci-11-01548/article_deploy/html/images/applsci-11-01548-g001.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Jung 2021</p>
</div>

</div>
</div>

<!--s-->

## Transformers | Self-Attention

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

The attention score is computed as the dot product of the query and key vectors, followed by a softmax operation to normalize the scores. Value vectors are then weighted by these scores to produce the final output.

$$ \text{Attention}(q, k, v) = \text{softmax}\left(\frac{q \cdot k^T}{\sqrt{d_k}}\right) v $$

where: 
- $q$ is the query vector
- $k$ is the key vector
- $v$ is the value vector
- $d_k$ is the dimension of the key vector

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/attention.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>

</div>
</div>
</div>

<!--s-->

## Transformers | Self-Attention

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/attention.png' style='border-radius: 10px; width: 60%; height: 80%'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>

</div>

<!--s-->

## Transformers | Multi-Head Attention

Multi-head attention is an extension of the self-attention mechanism that allows the model to focus on different parts of the input sequence simultaneously. It does this by using multiple sets of query, key, and value vectors, each with its own learned parameters.

<div style='text-align: center;'>
   <img src='https://sanjayasubedi.com.np/assets/images/deep-learning/mha-scratch/mha_dp_fig.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

<!--s-->

## Transformers | Architecture

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/transformer.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey;'>Vashwani 2017</p>
</div>

<!--s-->

## Transformers | Tokenization | Why Tokenize?

Tokenization is the process of converting text into smaller units (tokens) that can be processed by machine learning models. It is a crucial step in natural language processing (NLP) and has several important purposes:

- **Efficiency**: Reduce the vocabulary size to improve model efficiency.
- **Generalization**: Handle out-of-vocabulary words by breaking them into subwords or characters.
- **Context Preservation**: Maintain the context of words and phrases.
- **Flexibility**: Allow for dynamic vocabulary sizes based on the training data.

<!--s-->

## Transformers | Tokenization | Byte Pair Encoding (BPE)

BPE is a subword tokenization algorithm that builds a vocabulary of subwords by iteratively merging the most frequent pairs of characters. BPE is a powerful tokenization algorithm because it can handle rare words and out-of-vocabulary words. It is used by many large language models, including GPT-4. The algorithm is as follows:

```text
1. Initialize the vocabulary with all characters in the text.
2. While the vocabulary size is less than the desired size:
    a. Compute the frequency of all character pairs.
    b. Merge the most frequent pair.
    c. Update the vocabulary with the merged pair.
```

<!--s-->

## Byte Pair Encoding (BPE) with TikToken

One BPE implementation can be found in the `tiktoken` library, which is an open-source library from OpenAI.

```python

import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # Get specific encoding used by GPT-4.
enc.encode("Hello, world!") # Returns the tokenized text.

>> [9906, 11, 1917, 0]

```

<!--s-->

## Transformers | Padding & Truncation

The pad & truncate step ensures that all input sequences have the same length, and takes place after tokenization. This is important for batch processing and model training.

<div style='text-align: center;'>
   <img src='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*I0JbAgArgFzMCWiOB1o9CQ.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Lokare 2023</p>
</div>

<!--s-->

## Transformers | Embedding Layer

The embedding layer is a crucial component of neural networks, especially in natural language processing (NLP) tasks. It transforms discrete tokens into continuous vector representations, allowing the model to learn semantic relationships between tokens. 

The embedding layer is backpropagated through, meaning that the embeddings are learned during training.

<div style='text-align: center;'>
   <img src='https://cdn.prod.website-files.com/6064b31ff49a2d31e0493af1/66d06d2f219c5eab928c6b5b_AD_4nXdL2BUY6asFzNdhx_FYCFp6DNBRCwLx_XCALqjkUueNttpIa0WPQWRzUxSNvDBdyU6U3r5unc1OJu4-4DxbNAXeaXlS7Y9BW2-igGX91VVZHRSlwLYYw0dE0m5DPrw29A3RNnVp5hV9S3ljW7GcLYRMPaX0.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Airbyte 2025</p>
</div>

<!--s-->


## Transformers | Positional Encoding

Positional encoding is a technique used in transformer models to add positional information to tokens. This is important because transformers do not have a built-in notion of order, unlike recurrent neural networks (RNNs). Positional encoding allows the model to understand the order of tokens in a sequence.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/positional_embedding.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## Transformers | Positional Encoding | Methods

In practice, these actually perform similarly! Vashwani et al. (2017) found that sinusoidal positional encoding performed better than learned positional embeddings in some cases, but the difference is often negligible. Modern architectures will actually use something called Rotary Positional Embedding (RoPE), but I'll leave that to Professor Demeter in NLP.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Learned Positional Embeddings
Learn a separate embedding for each position in the sequence.

</div>
<div class="c2" style = "width: 50%">

### Sinusoidal Positional Encoding
Use sine and cosine functions to generate positional embeddings.

<!--s-->

## Transformers | Learned Positional Embeddings

```python
class TokenAndPositionEmbedding(layers.Layer):
   def __init__(self, maxlen, vocab_size, embed_dim):
      super().__init__()
      self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
      self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

   def call(self, x):
      maxlen = ops.shape(x)[-1]
      positions = ops.arange(0, maxlen, 1)
      positions = self.pos_emb(positions)
      x = self.token_emb(x)
      return x + positions

```

<div style='text-align: center;'>
   <p style='font-size: 0.6em;'>

   [Nandan 2020](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)

   </p>
</div>


<!--s-->

## Transformers | Sinusoidal Positional Encoding

```python
def positional_encoding(length, depth):
   depth = depth/2
   positions = np.arange(length)[:, np.newaxis]
   depths = np.arange(depth)[np.newaxis, :]/depth
   angle_rates = 1 / (10000**depths)
   angle_rads = positions * angle_rates
   pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
   return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
   def __init__(self, vocab_size, d_model):
      super().__init__()
      self.d_model = d_model
      self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
      self.pos_encoding = positional_encoding(length=2048, depth=d_model)

   def compute_mask(self, *args, **kwargs):
      return self.embedding.compute_mask(*args, **kwargs)

   def call(self, x):
      length = tf.shape(x)[1]
      x = self.embedding(x)
      # This factor sets the relative scale of the embedding and positonal_encoding.
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      x = x + self.pos_encoding[tf.newaxis, :length, :]
      return x

```

<div style='text-align: center;'>
   <p style='font-size: 0.6em; color: grey;'>
   
   [Tensorflow Tutorial](https://www.tensorflow.org/text/tutorials/transformer?hl=en)

   </p>
</div>

<!--s-->

## Transformers | Causal Masking

Causal masking is a technique used in autoregressive models to prevent the model from seeing future tokens during training. This is important for tasks like text generation, where the model should only use past tokens to predict the next token. Otherwise, the model would learn to cheat by looking at future tokens.

Causal masking is important for decoder-only architectures, such as GPT-4.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/causal_mask.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>

<!--s-->

## L.06 | Q.02

What type of normalization is used in transformers?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

A. Batch Normalization<br>
B. Layer Normalization<br>
C. Instance Normalization<br>
D. Group Normalization<br>

</div>

<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.06 | Q.02' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

## Transformers | Normalization

Layer normalization normalizes the inputs to each layer, ensuring that they have a mean of 0 and a standard deviation of 1. This helps to mitigate the effects of internal covariate shift and improves convergence.
 
Compared to batch normalization, layer normalization is applied to each individual sample rather than across a batch.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/layer_normalization.png' style='border-radius: 10px; width: 50%;'>
   <p style='font-size: 0.6em; color: grey;'>Foster 2024</p>
</div>


<!--s-->

## Transformer Architecture Recap

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/transformer.png' style='border-radius: 10px; width: 40%;'>
   <p style='font-size: 0.6em; color: grey;'>Vashwani 2017</p>
</div>

<!--s-->

## L.06 | Q.03

What type of transformer architecture is BERT?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

A. Encoder-Only<br>
B. Decoder-Only<br>
C. Encoder-Decoder<br>

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.06 | Q.03' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

## Encoder, Decoder, and Encoder-Decoder

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

**Encoder-Decoder Models**: T5, BART.

Encoder-decoder models generate text by encoding the input text into a fixed-size vector and then decoding the vector into text. Used in machine translation and text summarization.

**Encoder-Only**: BERT

Encoder-only models encode the input text into a fixed-size vector. These models are powerful for text classification tasks but are not typically used for text generation.

**Decoder-Only**: GPT-4, GPT-3, Gemini

Autoregressive models generate text one token at a time by conditioning on the previous tokens. Used in text generation, language modeling, and summarization.

</div>
<div class="c2 col-centered" style = "width: 40%">

<div>
<img src='https://storage.googleapis.com/slide_assets/transformer.png' style="margin: 0; padding: 0; ">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Vaswani, 2017</span>
</div>
</div>
</div>

<!--s-->

## Building a Text Gerator with Keras

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile("adam", loss=[loss_fn, None],)
    return model
```

<div style='text-align: center;'>
   <p style='font-size: 0.6em;'>

   [Nandan 2020](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)

   </p>
</div>



<!--s-->

<div class="header-slide">

# Retrieval Augmented Generation (RAG)

</div>

<!--s-->

## Retrieval Augmented Generation (RAG)

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Chunking

Split text into smaller chunks for efficient retrieval.

### Indexing
Create an index of chunks for fast retrieval.

</div>
<div class="c2" style = "width: 50%">

### Retrieval
Retrieve relevant chunks based on the input query.

### Generation
Generate text based on the retrieved chunks and the input query.

</div>
</div>

<!--s-->

## Motivation | RAG

Large language models (LLMs) have revolutionized natural language processing (NLP) by achieving state-of-the-art performance on a wide range of tasks. We will discuss LLMs in more detail later in this lecture. However, for now it's important to note that modern LLMs have some severe limitations, including:

- **Inability to (natively) access external knowledge**
- **Hallucinations** (generating text that is not grounded in reality)

Retrieval-Augmented Generation (RAG) is an approach that addresses these limitations by combining the strengths of information retrieval systems with LLMs.

<!--s-->

## Motivation | RAG

So what is Retrieval-Augmented Generation (RAG)?

1. **Retrieval**: A storage & retrieval system that obtains context-relevant documents from a database.
2. **Generation**: A large language model that generates text based on the obtained documents.

<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/rag-pipeline-ingest-query-flow-b.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">NVIDIA, 2023</span>

<!--s-->

## Motivation | Creating an Expert Chatbot 🤖

<div style = "font-size: 0.8em;">

Our goal today is to build a RAG system that will answer questions about Northwestern's policy on academic integrity. To do this, we will:

1. **Chunk** the document into smaller, searchable units.<br>
Chunking is the process of creating windows of text that can be indexed and searched. We'll learn how to chunk text to make it compatible with a vector database.

2. **Embed** the text chunks.<br>
Word embeddings are dense vector representations of words that capture semantic information. We'll learn how to embed chunks using OpenAI's embedding model (and others!).

3. **Store and Retrieve** the embeddings from a vector database.<br>
We'll store the embeddings in a vector database and retrieve relevant documents based on the current context of a conversation. We'll demo with chromadb.

4. **Generate** text using the retrieved chunks and conversation context.<br>
We'll generate text with GPT-4 based on the retrieved chunks and a provided query, using OpenAI's API.

</div>

<!--s-->

<div class="header-slide">

# Chunk

</div>

<!--s-->

## Chunk | 🔥 Tips & Methods

<div style = "font-size: 0.8em;">

Chunking is the process of creating windows of text that can be indexed and searched. Chunking is essential for information retrieval systems because it allows us to break down large documents into smaller, searchable units.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; height: 100%; margin-right: 2em;">


### Sentence Chunking

Sentence chunking is the process of breaking text into sentences.

E.g. <span class="code-span">"Hello, world! How are you?" -> ["Hello, world!", "How are you?"]</span>

### Paragraph Chunking

Paragraph chunking is the process of breaking text into paragraphs.

E.g. <span class="code-span">"Hello, world! \n Nice to meet you." -> ["Hello, world!", "Nice to meet you."]</span>

### Agent Chunking

Agent chunking is the process of breaking text down using an LLM.

</div>

<div class="c2" style = "width: 50%; height: 100%;">

### Sliding Word / Token Window Chunking

Sliding window chunking is a simple chunking strategy that creates windows of text by sliding a window of a fixed size over the text.

E.g. <span class="code-span">"The cat in the hat" -> ["The cat in", "cat in the", "in the hat"]</span>

### Semantic Chunking

Semantic chunking is the process of breaking text into semantically meaningful units.

E.g. <span class="code-span">"The cat in the hat. One of my favorite books." -> ["The cat in the hat.", "One of my favorite books."]</span>

</div>
</div>

<!--s-->

## Chunk | NLTK Sentence Chunking

NLTK is a powerful library for natural language processing that provides many tools for text processing. NLTK provides a sentence tokenizer that can be used to chunk text into sentences.

### Chunking with NLTK

```python
from nltk import sent_tokenize

# Load Academic Integrity document.
doc = open('/Users/joshua/Desktop/academic_integrity.md').read()

# Split the document into sentences.
chunked_data = sent_tokenize(doc)
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; height: 100%;">

### Input: Original Text

```text
The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and
throughout the larger intellectual community of which we are members. Please read this booklet carefully,
as you will be held responsible for its contents. It describes the ways in which common sense and decency apply
to academic conduct. When you applied to Northwestern, you agreed to abide by our principles of academic integrity;
these are spelled out on the first three pages. The balance of the booklet provides information that will help you avoid
violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who 
can give you further information and counseling within the undergraduate schools.
```

</div>
<div class="c2" style = "width: 50%; height: 100%;">

### Output: Chunked Text (by Sentence)
```text
[
    "The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and throughout the larger intellectual community of which we are members."
    "Please read this booklet carefully, as you will be held responsible for its contents."
    "It describes the ways in which common sense and decency apply to academic conduct."
    "When you applied to Northwestern, you agreed to abide by our principles of academic integrity; these are spelled out on the first three pages."
    "The balance of the booklet provides information that will help you avoid violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who can give you further information and counseling within the undergraduate schools."
]

```
</div>
</div>

<!--s-->

<div class="header-slide">

# Embed

</div>

<!--s-->

## Embed

Word embeddings are dense vector representations of words that capture semantic information. Word embeddings are essential for many NLP tasks because they allow us to work with words in a continuous and meaningful vector space.

**Traditional embeddings** such as Word2Vec are static and pre-trained on large text corpora.

**Contextual embeddings** such as those produced by BERT (encoder-only Transformer model) are dynamic and depend on the context in which the word appears. Contextual embeddings are essential for many NLP tasks because they capture the *contextual* meaning of words in a sentence.

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Google</span>

<!--s-->

## Embed | Contextual Word Embeddings

Contextual word embeddings are word embeddings that are dependent on the context in which the word appears. Contextual word embeddings are essential for many NLP tasks because they capture the *contextual* meaning of words in a sentence.

For example, the word "bank" can have different meanings depending on the context:

- **"I went to the bank to deposit my paycheck."**
- **"The river bank was covered in mud."**

[HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) contains a [MTEB](https://arxiv.org/abs/2210.07316) leaderboard for some of the most popular contextual word embeddings:

<img src="https://storage.googleapis.com/cs326-bucket/lecture_14/leaderboard.png" style="margin: 0 auto; display: block; width: 50%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">HuggingFace, 2024</span>

<!--s-->
<div class="header-slide">

# Embed | BERT Demo

</div>

<!--s-->

## Embed | OpenAI's Embedding Model

OpenAI provides an embedding model via API that can embed text into a dense vector space. The model is trained on a large text corpus and can embed text into a n-dimensional vector space.

```python
import openai

openai_client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])
embeddings = openai_client.embeddings.create(model="text-embedding-3-large", documents=chunked_data)
```

🔥 Although they do not top the MTEB leaderboard, OpenAI's embeddings work well and the convenience of the API makes them a popular choice for many applications.

<!--s-->

<div class="header-slide">

# Retrieve

</div>

<!--s-->

## Store & Retrieve

A vector database is a database that stores embeddings and allows for fast similarity search. Vector databases are essential for information retrieval systems because they enable us to *quickly* retrieve relevant documents based on their similarity to a query. 

This retrieval process is very similar to a KNN search! However, vector databases will implement Approximate Nearest Neighbors (ANN) algorithms to speed up the search process -- ANN differs from KNN in that it does not guarantee the exact nearest neighbors, but rather a set of approximate nearest neighbors.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There are many vector databases options available, such as:

- [ChromaDB](https://www.trychroma.com/)
- [Pinecone](https://www.pinecone.io/product/)
- [Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Postgres with PGVector](https://github.com/pgvector/pgvector)
- [FAISS](https://ai.meta.com/tools/faiss/)
- ...

</div>
<div class="c2" style = "width: 50%">

<img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bg8JUIjbKncnqC5Vf3AkxA.png" style="margin: 0 auto; display: block; width: 80%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Belagotti, 2023</span>

</div>
</div>

<!--s-->

## Store & Retrieve | ChromaDB

<div style="font-size: 0.9em">

ChromaDB is a vector database that stores embeddings and allows for fast text similarity search. ChromaDB is built on top of SQLite and provides a simple API for storing and retrieving embeddings.

### Initializing

Before using ChromaDB, you need to initialize a client and create a collection.

```python
import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection('academic_integrity_nw')
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Storing Embeddings

Storing embeddings in ChromaDB is simple. You can store embeddings along with the original documents and ids.

```python
# Store embeddings in chromadb.
collection.add(embeddings = embeddings, documents = chunked_data, ids = [f"id.{i}" for i in range(len(chunked_data))])
```

</div>
<div class="c2" style = "width: 50%">

### Retrieving Embeddings

You can retrieve embeddings from ChromaDB based on a query. ChromaDB will return the most similar embeddings (and the original text) to the query.

```python
# Get relevant documents from chromadb, based on a query.
query = "Can a student appeal?"
relevant_chunks = collection.query(query_embeddings = embedding_function([query]), n_results = 2)['documents'][0]

>>> ['A student may appeal any finding or sanction as specified by the school holding jurisdiction.',
     '6. Review of any adverse initial determination, if requested, by an appeals committee to whom the student has access in person.']

```

</div>
</div>
</div>

<!--s-->

## Retrieving Embeddings | 🔥 Tips & Re-Ranking

In practice, the retrieved documents may not be in the order you want. While a vector db will often return documents in order of similarity to the query, you can re-rank documents based on a number of factors. Remember, your chatbot is paying per-token on calls to LLMs. You can cut costs by re-ranking the most relevant documents first and only sending those to the LLM.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; margin-right: 2em;">

### Multi-criteria Optimization

Consideration of additional factors beyond similarity, such as document quality, recency, and 'authoritativeness'.

### User Feedback

Incorporate user feedback into the retrieval process. For example, if a user clicks on a document, it can be re-ranked higher in future searches.

</div>

<div class="c2" style = "width: 50%">

### Diversification

Diversify the search results by ensuring that the retrieved documents cover a wide range of topics.

### Query Expansion & Rephrasing

For example, if a user asks about "academic integrity", the system could expand the query to include related terms like "plagiarism" and "cheating". This will help retrieve more relevant documents.

<!--s-->

<div class="header-slide">

# Generate

</div>

<!--s-->

## Generate

Once we have retrieved the relevant chunks based on a query, we can generate text using a large language model. Large language models can be used for many tasks -- including text classification, text summarization, question-answering, multi-modal tasks, and more.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There are many large language models available at platforms like:

- [OpenAI GPT-4o](https://platform.openai.com/)
- [Google Gemini](https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=CjwKCAiAudG5BhAREiwAWMlSjKXwuvq9JRRX0xxXaS7yCSn-NWo3e4rso3D-enl2IblIH09phtCvSxoCJhoQAvD_BwE)
- [Anthropic Claude](https://claude.ai/)
- [HuggingFace (Many)](https://huggingface.co/)
- ...


</div>
<div class="c2" style = "width: 50%">

<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcce3c437-4b9c-4d15-947d-7c177c9518e5_4258x5745.png" style="margin: 0 auto; display: block; width: 80%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Raschka, 2023</span>

</div>
</div>

<!--s-->

## Generate | GPT-4 & OpenAI API

What really sets OpenAI apart is their extremely useful and cost-effective API. This puts their LLM in the hands of users with minimal effort. Competitors liuke Anthropic and Google have similar APIs now.

```python

import openai

openai_client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, GPT-4!"}
    ]
)

```

<!--s-->

## Generate | Prompt Engineering 🔥 Tips

<div class = "col-wrapper" style = "font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Memetic Proxy

A memetic proxy is a prompt that provides context to the LLM by using a well-known meme or phrase. This can help the LLM derive the context of the conversation and generate more relevant responses. [McDonell 2021](https://arxiv.org/pdf/2102.07350).

### Few-Shot Prompting
Few-shot prompting is a technique that provides the LLM with a few examples of the desired output. This can help the LLM understand the context and generate more relevant responses. [OpenAI 2023](https://arxiv.org/abs/2303.08774).

</div>
<div class="c2" style = "width: 50%">

### Chain-of-Thought Prompting
Chain-of-thought prompting is a technique that provides the LLM with a series of steps to follow in order to generate the desired output. This can help the LLM understand the context and generate more relevant responses. [Ritter 2023](https://arxiv.org/pdf/2305.14489).

### **TYPOS AND CLARITY**
Typos and clarity are important factors to consider when generating text with an LLM. Typos can lead to confusion and misinterpretation of the text, while clarity can help the LLM with the context and generate more relevant responses.

</div>
</div>

<!--s-->

<div class="header-slide">

# Putting it All Together

</div>

<!--s--> 

## Putting it All Together

Now that we have discussed the components of Retrieval-Augmented Generation (RAG), let's use what we have learned to build an expert chatbot that can answer questions about Northwestern's policy on academic integrity.

<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/rag-pipeline-ingest-query-flow-b.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">NVIDIA, 2023</span>

<!--s-->

## Putting it All Together | Demo Copied Here

```python[1-10 | 12-13 | 15-16 | 18-19 | 21-23 | 25-26 | 28 - 39 | 40 - 42 | 44-46 | 48 - 51]
import os

import chromadb
import openai
from chromadb.utils import embedding_functions
from nltk import sent_tokenize

# Initialize clients.
chroma_client = chromadb.Client()
openai_client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])

# Create a new collection.
collection = chroma_client.get_or_create_collection('academic_integrity_nw')

# Load academic integrity document.
doc = open('/Users/joshua/Documents/courses/SPRING25-GENERATIVE-AI/docs/academic_integrity.md').read()

# Chunk the document into sentences.
chunked_data = sent_tokenize(doc)

# Embed the chunks.
embedding_function = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002", api_key=os.environ['OPENAI_API_KEY'])
embeddings = embedding_function(chunked_data)

# Store embeddings in ChromaDB.
collection.add(embeddings = embeddings, documents = chunked_data, ids = [f"id.{i}" for i in range(len(chunked_data))])

# Create a system prompt template.

SYSTEM_PROMPT = """

You will provide a response to a student query using exact language from the provided relevant chunks of text.

RELEVANT CHUNKS:

{relevant_chunks}

"""

# Get user query.
user_message = "Can I appeal?"
print("User: " + user_message)

# Get relevant documents from chromadb.
relevant_chunks = collection.query(query_embeddings = embedding_function([user_message]), n_results = 2)['documents'][0]
print("Retrieved Chunks: " + str(relevant_chunks))

# Send query and relevant documents to GPT-4.
system_prompt = SYSTEM_PROMPT.format(relevant_chunks = "\n".join(relevant_chunks))
response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
print("RAG-GPT Response: " + response.choices[0].message.content)

```

```text
User: Can a student appeal?
Retrieved Chunks: ['A student may appeal any finding or sanction as specified by the school holding jurisdiction.', '6. Review of any adverse initial determination, if requested, by an appeals committee to whom the student has access in person.']
RAG-GPT Response: Yes, a student may appeal any finding or sanction as specified by the school holding jurisdiction.
```

<!--s-->

<div class="header-slide">

# Wrapping RAG in a pretty GUI
## Streamlit App

</div>

<!--s-->

<div class="header-slide">

# Model Context Protocol

</div>

<!--s-->

## Model Context Protocol (MCP)

"MCP is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools." 

<div style="text-align: right; margin-right: 2em;">
   <a href="https://docs.anthropic.com/en/docs/agents-and-tools/mcp">Anthropic, 2025</a>
</div>

<!--s-->

## Model Context Protocol (MCP) | Overview

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/mcp.png' style='border-radius: 10px; width: 70%;'>
   <p style='font-size: 0.6em; color: grey;'>Anthropic 2025</p>
</div>

<!--s-->

<div class="header-slide">

# Demo of MCP
## [File System](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) & Claude

</div>

<!--s-->

<div class="header-slide">

# Demo of BYO-MCP
## Weather API Integration

</div>

<!--s-->

<div class="header-slide">

# Demo of MCP
## Spotify API & Claude

</div>

<!--s-->

## MCP Note

MCP is a push from Anthropic to establish a standard for how LLMs can interact with external data sources and tools. It is under active development! Here is a post from 05.01.2025 that describes using [remote MCP servers](https://www.anthropic.com/news/integrations).

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. Transformer Architectures
  2. Retrieval-Augmented Generation (RAG)
  3. Streamlit GUIs
  4. Model Context Protocol (MCP)

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->
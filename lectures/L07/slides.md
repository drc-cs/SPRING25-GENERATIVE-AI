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
  ## L.07 | Forecasting

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

## Project 2 | Image Generation Grading Criteria

[Project 2 Proposal](https://canvas.northwestern.edu/courses/230487/assignments/1590622) is due **tonight**, 05.14.2025 at 11:59PM. Please note, I will be cross-referencing this proposal with your projects for other classes.

Project 2 is due on 06.04.2025 and is worth 100 points.

| Criteria | Points | Description |
| -------- | ------ | ----------- |
| Generation of Text | 40 | Your model should be capable of generating text. |
| Code Quality | 20 | Your code should be well-organized and easy to read. Please upload to GitHub and share the link. Notebooks are fine but **must** be tidy. |
| Code Explanation | 25 | You should know your code inside and out. Please do not copy and paste from other sources (including GPT). Xinran and I will conduct an oral exam for your code. |
| Extra Criteria | 15 | Extra criteria is defined in the [README](https://github.com/drc-cs/SPRING25-GENERATIVE-AI?tab=readme-ov-file#extra-project-criteria). |

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident would you be working with forecasting models?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Forecasting

</div>

<!--s-->

# Agenda

<div class = "col-wrapper">
<div class="c1" style = "width: 50%;">

### Forecasting

- Covariates (Past / Future / Static)
- Train & Forecast on Multiple Series
- Probabilistic Forecasting

### Forecasting Models
  - SARIMAX (classical time series)
  - XGBoost (re-appropriated regression models)
  - N-Linear (modern time series)
  - TiDE (modern time series)

</div>
<div class="c2" style = "width: 50%;">

### Forecasting Competition & Project Review
  - Train & forecast in loop w/ Darts

</div>
</div>




<!--s-->

<div class="header-slide">

# Forecasting

</div>

<!--s-->

## Forecasting

Forecasting is the process of analyzing time-ordered data to extract meaningful patterns and make predictions about future values. It involves understanding the underlying structure of the data, identifying trends, seasonality, and other patterns, and using this information to build models that can forecast future values. Today we're breaking down models into three categories:

- **Classical Models**: These models are based on statistical techniques and assumptions about the data. Examples include ARIMA, SARIMA, and Exponential Smoothing.

- **ML / Regression Models**: These models use machine learning algorithms to learn patterns in the data. These models were often not originally designed for time series data, but they can be adapted to work with time series features. Examples include XGBoost and Random Forests.

- **Deep Learning Models**: These models use deep learning techniques to learn complex patterns in the data. Examples include N/D-Linear and TiDE.

<!--s-->

## Forecasting | Prediction Task

The prediction task in time series modeling involves forecasting future values. Values that can be used to predict future values can be categorized into three types:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

- **Past Values**: These are the historical values of the time series.

- **Future Values**: These are the values that will be observed in the future.

- **Static Covariates**: These are additional features that do not change over time.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/covariates.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Darts 2025</p>
</div>

</div>
</div>

<!--s-->

## Forecasting | Past Covariates

These are the most intuitive and commonly used covariates in time series modeling. They are the historical values of the time series itself, which can be used to predict future values. Consider the following example, which only uses past covariates to predict future values:

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/forecast.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Artley 2022</p>
</div>

<!--s-->

## Forecasting | Future Covariates

Future covariates are values that will be observed in the future. They can be used to improve the accuracy of predictions, especially when there are known future events that will impact the time series. For example, if we know that a holiday will occur in the future, we can use that information to improve our predictions.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/covariates.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Darts 2025</p>
</div>

<!--s-->

## Forecasting | Static Covariates

Static covariates are additional features that do not change over time. They can provide valuable context for the time series and improve the accuracy of predictions. For example, if we are predicting sales for a retail store, we might include static covariates such as the store's location or the type of products sold. They are most useful in **multivariate** settings.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/covariates.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Darts 2025</p>
</div>

<!--s-->

## Forecasting | Train & Forecast on Multiple Series

In many real-world applications, we have multiple time series that are related to each other. For example, we might have sales data for multiple products in a retail store. In these cases, we can train a model on multiple time series and use it to forecast future values for all of them. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

This is often referred to as **multivariate time series forecasting**. It allows us to leverage the relationships between different time series to improve our predictions.

</div>
<div class="c2" style= "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/wine_sales.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Cerqueira 2022</p>
</div>

</div>
</div>

<!--s-->

<div class="header-slide">

# Probabilistic Forecasting

</div>

<!--s-->

## Probabilistic Forecasting

Probabilistic forecasting is a technique used to predict future values in a time series by estimating the probability distribution of the future values. This approach provides a range of possible outcomes, rather than a single point estimate, which can be useful for decision-making under uncertainty.

<div style="display: flex; justify-content: center;">
   <div style='text-align: center; width: 70%;'>
      <img src='https://storage.googleapis.com/slide_assets/quantile_forecast.png' style='border-radius: 10px;'>
      <p style='font-size: 0.6em; color: grey;'>Vermorel 2012</p>
   </div>
</div>

<!--s-->

## Probabilistic Forecasting | Quantile Regression

Quantile regression works by fitting a model to the data and estimating the quantiles of the response variable. This can be achieved through something called the pinball loss function.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/quantile_distribution_plot.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Chawla 2024</p>
</div>

<!--s-->

## Pinball Loss Function

The pinball loss function is a loss function used in quantile regression to estimate the conditional quantiles of a response variable. In the case below "$\alpha$" is the quantile level, which can be set to any value between 0 and 1.

[[math, section 3](https://arxiv.org/pdf/2304.11732)]

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/pinball_loss.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Koenker and Bassett Jr (1978)</p>
</div>


<!--s-->

## Probabilistic Forecasting | Quantile Regression

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/quantile_demo_plot.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Chawla 2024</p>
</div>

<!--s-->

## Probabilistic Forecasting | Quantile Regression w/ Time Series

Quantile regression can be applied to forecasts, where the goal is to predict future values at different quantiles. In this case, we can use past values of the time series as predictors. The following is a simple example using <span class="code-span">darts</span>:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
from darts.datasets import AirPassengersDataset
from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models.torch import QuantileRegression

series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]

scaler = Scaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
series = scaler.transform(series)

model = TCNModel(input_chunk_length=30, output_chunk_length=12, likelihood=QuantileRegression(quantiles=[0.25, 0.5, 0.75]))
model.fit(train, epochs=400)
pred = model.predict(n=36, num_samples=500)

series.plot()
pred.plot(label='forecast')
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/darts_quantile_example2.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>
</div>
</div>


<!--s-->

<div class="header-slide">

# Forecasting Models
## ARIMA, SARIMAX

</div>

<!--s-->

## Forecasting Models | ARIMA to SARIMAX

ARIMA (AutoRegressive Integrated Moving Average) is characterized by three main components:

- AR (AutoRegressive): This component captures the relationship between an observation and a number of lagged observations (i.e., past values). $p$ is the number of lag observations included in the model (the order of the AR term)

- I (Integrated): This component involves differencing the raw observations to make the time series stationary, which is a requirement for many time series models. $d$ is the number of times that the raw observations are differenced (the degree of differencing)

- MA (Moving Average): This component captures the relationship between an observation and a residual error from a moving average model applied to lagged observations. $q$ is the size of the moving average window (the order of the MA term)


<!--s-->

## ARIMA Introduction | Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used to capture the periodic structure of the time series.

AR(1) $$ X_t = \phi_1 X_{t-1} + c + \epsilon_t $$

Where:

- $X_t$ is the observed value at time $t$.
- $\phi_1$ is a learnable parameter of the model.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## ARIMA Introduction | Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used to capture the periodic structure of the time series.

AR(p) $$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + c + \epsilon_t $$

Where:

- $X_t$ is the observed value at time $t$.
- $p$ is the number of lag observations included in the model.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the parameters of the model.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## ARIMA Introduction | Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used for capturing the periodic structure of the time series.

$$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + c + \epsilon_t $$

<iframe width = "100%" height = "70%" src="https://storage.googleapis.com/cs326-bucket/lecture_13/ARIMA_1_2.html" title="scatter_plot"></iframe>

<!--s-->

## ARIMA Introduction | Moving Average

**Moving Average (MA) Models**: A type of time series model that predicts future values based on the past prediction errors. A MA model's primary utility is to smooth out noise and short-term discrepancies from the mean.

MA(1) $$ X_t = \theta_1 \epsilon_{t-1} + \mu + \epsilon_t$$

<div class = "col-wrapper" style="font-size: 0.8em;">
<div class="c1" style = "width: 50%">

Where: 

- $X_t$ is the observed value at time $t$.
- $\theta_1$ is a learnable parameter of the model.
- $\mu$ is the mean of the time series.
- $\epsilon_t$ is the white noise at time $t$.

</div>
<div class="c2" style = "width: 50%">

Example with a $\mu = 10 $ and $\theta_1 = 0.5$:

| t | $\widehat{X}_t$ | $\epsilon_t$ | $X_t$ |
|---|------------|--------------|-------|
| 1 | 10         | -2            | 8    |
| 2 | 9         | 1           | 10    |
| 3 | 10.5         | 0            | 10.5    |
| 4 | 10         | 2           | 12     |
| 5 | 11         | -1           | 10    |


</div>
</div>

<!--s-->

## ARIMA Introduction | Moving Average

**Moving Average (MA) Models**: A type of time series model that predicts future values based on the past prediction errors. A MA model's primary utility is to smooth out noise and short-term discrepancies from the mean.

MA(1) $$ X_t = \theta_1 \epsilon_{t-1} + \mu + \epsilon_t$$

<iframe width = "100%" height = "70%" src="https://storage.googleapis.com/cs326-bucket/lecture_13/MA2.html" title="scatter_plot";></iframe>

<!--s-->

## ARIMA Introduction | Moving Average

**Moving Average (MA) Models**: A type of time series model that predicts future values based on the past prediction errors. A MA model's primary utility is to smooth out noise and short-term discrepancies from the mean.

MA(q) $$ X_t = \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q} + \mu + \epsilon_t$$

Where: 

- $X_t$ is the observed value at time $t$.
- $q$ is the number of lag prediction errors included in the model.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the learnable parameters.
- $\mu$ is the mean of the time series.
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## ARIMA Introduction | ARMA

**Autoregressive Models with Moving Average (ARMA)**: A type of time series model that combines autoregressive and moving average components.

The ARMA model is defined as:

$$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q} + c + \epsilon_t $$


Where:

- $X_t$ is the observed value at time $t$.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive parameters.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the moving average parameters.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.


<!--s-->

## ARIMA Introduction | ARIMA

**Autoregressive Integrated Moving Average (ARIMA)**: A type of time series model that combines autoregressive, moving average, and differencing components.

The ARIMA model is defined as: 

$$ y_t' = \phi_1 y_{t-1}' + \phi_2 y_{t-2}' + \ldots + \phi_p y_{t-p}' + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q} + c + \epsilon_t $$

Where:

- $y_t'$ is the differenced observation at time $t$.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive parameters.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the moving average parameters.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Forecasting Models | ARIMA to SARIMAX

SARIMA (Seasonal AutoRegressive Integrated Moving Average) extends the ARIMA model to account for seasonality in the time series data. It includes additional seasonal terms to capture the seasonal patterns in the data. The SARIMA model is typically denoted as SARIMA(p, d, q)(P, D, Q, s), where:

- $P$ is the number of seasonal autoregressive terms,
- $D$ is the number of seasonal differences,
- $Q$ is the number of seasonal moving average terms,
- $s$ is the length of the seasonal cycle (e.g., 12 for monthly data with yearly seasonality).

The SARIMA model is particularly useful for time series data with strong seasonal patterns, as it can capture both the non-seasonal and seasonal components of the data.

<!--s-->

## Forecasting Models | ARIMA to SARIMAX

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) is an extension of the SARIMA model that allows for the inclusion of exogenous variables (covariates) in the model. This is useful when there are additional features that can help improve the accuracy of the forecasts.

<!--s-->

<div class="header-slide">

# Forecasting Models
## Re-Appropriated Regression Models

</div>

<!--s-->

## Forecasting Models | XGBoost

XGBoost is a popular machine learning algorithm that can be used for time series forecasting. It is an ensemble learning method that combines the predictions of multiple weak learners (typically decision trees) to produce a strong learner.

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

XGBoost is not a time series model per se, but it can be adapted for time series forecasting by creating features from the time series data. This involves generating lagged features, rolling statistics, and other relevant features that capture the temporal patterns in the data. 

Then XGBoost is trained on these features to make predictions. This is similar to the Autoregressive models we discussed earlier, but with the added flexibility and power of gradient boosting.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://flower.ai/static/images/blog/content/2023-11-29-xgboost-pipeline.jpg' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Gao 2023</p>
</div>

</div>
</div>



<!--s-->

## Forecasting Models | XGBoost Example (Darts)

```python

from darts.datasets import WeatherDataset
from darts.models import XGBModel
series = WeatherDataset().load()
# predicting atmospheric pressure
target = series['p (mbar)'][:100]
# optionally, use past observed rainfall (pretending to be unknown beyond index 100)
past_cov = series['rain (mm)'][:100]
# optionally, use future temperatures (pretending this component is a forecast)
future_cov = series['T (degC)'][:106]
# predict 6 pressure values using the 12 past values of pressure and rainfall, as well as the 6 temperature
# values corresponding to the forecasted period
model = XGBModel(
    lags=12,
    lags_past_covariates=12,
    lags_future_covariates=[0,1,2,3,4,5],
    output_chunk_length=6,
)
model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
pred = model.predict(6)
pred.values()
```
<!--s-->

<div class="header-slide">

# Forecasting Models (Modern)

</div>

<!--s-->

## Forecasting Models (Modern)

Forecasting models are incredibly valuable tools. As such, there are many modern approaches to time series forecasting that leverage deep learning techniques. These models can capture complex patterns in the data and are often more accurate than traditional statistical or regression models.

Between 2019 and 2022, Transformer-based models became a popular choice for time series forecasting, including popular models such as LogTrans (NeurIPS 2019), Informer (AAAI 2021 Best paper), Autoformer (NeurIPS 2021), Pyraformer (ICLR 2022 Oral), Triformer (IJCAI 2022) and FEDformer (ICML 2022).

<!--s-->

## But ... Are Transformers Effective for Time Series Forecasting? 
[[original_paper](https://arxiv.org/pdf/2205.13504)]

In 2022, a paper titled "Are Transformers Effective for Time Series Forecasting?" was published, which provided a comprehensive analysis of the effectiveness of Transformer-based models for time series forecasting.

<div class = "col-wrapper" style = "margin-top: 0; padding-top: 0;">
<div class="c1" style = "width: 50%; margin-top: 0; padding-top: 0;">

The authors conducted extensive experiments and found that while Transformers can be effective for certain types of time series data, a simple linear model outperformed them in every Long-Term Time Series Forecasting (LSTF) test case. 

Why do you think that is?

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/linear_model.png' style='border-radius: 10px; width: 70%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0; padding: 0;'>Zeng 2022</p>
</div>

</div>
</div>

<!--s-->

## Are Transformers Effective for Time Series Forecasting?

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/linear_comparisons.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Zeng 2022</p>
</div>

<!--s-->

## Vanilla Linear Model

The baseline model Zeng et al (2022) used for comparison is a simple linear model. This model is a straightforward approach to time series forecasting, where the future value is predicted as a linear combination of past values. Two variants of the linear model (N and D) are used to compare the performance of the Transformer-based models.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/linear_model.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0'>Zeng 2022</p>
</div>

<!--s-->

## D-Linear Model
[[original_paper](https://arxiv.org/pdf/2205.13504)]

The D-Linear model is a variant of the linear model that incorporates a trend component. The D-Linear model decomposes a raw data input into a trend component by a moving average kernel and a remainder (seasonal) component. Then, two linear layers are applied to each component, and we sum up the two features to get the final prediction.

D-Linear models are simple and effective, and should be considered when you have a strong trend component in your time series data.

<div style='text-align: center;'>
   <img src='https://images.squarespace-cdn.com/content/v1/678a4c72a9ba99192a50b3fb/b2ba5424-af32-4a97-b1c1-1015c4860c7c/decomp.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Davies</p>
</div>

<!--s-->

## N-Linear Model
[[original_paper](https://arxiv.org/pdf/2205.13504)]

The N-Linear model is another variant of the linear model that handles a distribution shift between training and testing sets. NLinear subtracts the input by the last value of the sequence. Then, the input goes through a linear layer, and the subtracted part is added back before making the final prediction.

N-Linear models are simple and effective, and should be considered when you have a distribution shift between training and testing sets.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/linear_distribution_shift.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Zeng 2022</p>
</div>

<!--s-->

## TiDE
[[original_paper](https://arxiv.org/pdf/2304.08424)]

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

TiDE (Time-series Dense Encoder) is a deep learning model that utilizes a dense encoder architecture specifically designed for time series data. It captures complex patterns in the data and can be used for both univariate and multivariate time series forecasting. The model is designed to be efficient and can handle large datasets with high dimensionality.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/tide.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Das 2024</p>
</div>

</div>
</div>

<!--s-->

## Honorable Mentions

| Model | Paper |
| --- | --- |
| TSMixer | TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting [[paper](https://arxiv.org/pdf/2306.09364v4)] |
| PatchTST | A Time Series is Worth 64 Words: Long Term Forecasting with Transformers [[paper](https://arxiv.org/pdf/2211.14730v2)] |
| N-HiTS | N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting [[paper](https://arxiv.org/abs/2201.12886)] | 
| SegRNN | SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[paper](https://arxiv.org/pdf/2308.11200v1)] |

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident would you be working with forecasting models?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Forecasting Competition & Project Review

## [[click_here_for_colab](https://colab.research.google.com/drive/1VDcldR1CTBkoYpc-JVydRm-8gtpfc9-T?usp=sharing)]

</div>




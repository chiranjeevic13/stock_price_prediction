# Stock Price Prediction Using LSTM

This project focuses on predicting stock prices using Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN) model. The dataset is sourced from Yahoo Finance, which provides historical stock price data. The goal is to train a model that can predict future stock prices based on past data, helping investors make informed decisions.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Stock market prediction is a crucial area of financial studies. Accurate predictions can lead to substantial profits, making this a highly sought-after skill. This project leverages deep learning techniques, specifically LSTM networks, to predict future stock prices using historical data.

## Project Overview

The project follows these main steps:

1. **Data Collection**: Historical stock prices are gathered from Yahoo Finance.
2. **Data Preprocessing**: The data is cleaned, normalized, and split into training and test sets.
3. **Model Building**: An LSTM model is built and trained on the data.
4. **Model Evaluation**: The model's performance is evaluated using various metrics.
5. **API Development**: A RESTful API is developed using FastAPI to serve predictions.
6. **Deployment**: The model and API are deployed for real-time stock price predictions.

## Features

- Fetches historical stock data from Yahoo Finance.
- Predicts future stock prices using an LSTM model.
- Exposes a RESTful API for integrating predictions into other applications.
- Visualizes stock price trends and prediction results.

## Technologies Used

- **Programming Language**: Python 3.7+
- **Libraries**: 
  - `pandas` for data manipulation
  - `numpy` for numerical computations
  - `matplotlib` and `seaborn` for data visualization
  - `scikit-learn` for data preprocessing
  - `tensorflow` and `keras` for building and training the LSTM model
  - `yfinance` for fetching stock data from Yahoo Finance
  - `FastAPI` for building the REST API
  - `Uvicorn` for running the FastAPI application

## Dataset

The dataset is collected from Yahoo Finance, which provides historical stock price data, including:

- **Date**: The date of the recorded stock price.
- **Open**: The price of the stock at the opening of the market.
- **High**: The highest price of the stock during the trading day.
- **Low**: The lowest price of the stock during the trading day.
- **Close**: The price of the stock at market close.
- **Adj Close**: Adjusted close price, accounting for splits and dividends.
- **Volume**: The number of shares traded.

### Example Dataset

| Date       | Open  | High  | Low   | Close | Adj Close | Volume   |
|------------|-------|-------|-------|-------|-----------|----------|
| 2023-01-01 | 150.0 | 155.0 | 149.0 | 154.0 | 154.0     | 1000000  |
| 2023-01-02 | 154.0 | 158.0 | 152.0 | 156.0 | 156.0     | 1100000  |

## Data Preprocessing

1. **Data Cleaning**: Remove missing or null values, if any.
2. **Normalization**: Normalize the stock price data to bring all values into a similar range, which helps improve the model's performance.
3. **Feature Engineering**: Create additional features if necessary, such as moving averages, to enhance model learning.
4. **Train-Test Split**: Split the dataset into training and test sets to evaluate the model's performance.

## Model Architecture

The LSTM model architecture includes:

- **Input Layer**: Takes the stock prices of the previous days as input.
- **LSTM Layers**: Stacked LSTM layers to capture the temporal dependencies in the data.
- **Dense Layer**: Fully connected layer to produce the final output.
- **Output Layer**: Predicts the stock price for the next day.

### Model Summary

- **Input Shape**: Number of previous days considered for prediction (e.g., 60 days)
- **LSTM Layers**: 2 layers with `units=50` each
- **Dropout**: 20% dropout to prevent overfitting
- **Dense Layer**: 1 layer with `units=1` to predict the output price

## Training the Model

- **Loss Function**: Mean Squared Error (MSE) to measure the prediction error.
- **Optimizer**: Adam optimizer for efficient training.
- **Epochs**: 50-100 epochs depending on the training performance.
- **Batch Size**: 32 to process data in small batches.

## Evaluation

- **Mean Squared Error (MSE)**: Measure the average squared difference between predicted and actual prices.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing error in the same unit as the stock prices.
- **Visualization**: Plot the predicted prices against the actual prices to visualize the model's performance.




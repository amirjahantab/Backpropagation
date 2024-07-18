[python]

## README.md

# Neural Network Implementation for Iris Dataset Classification

This project demonstrates the implementation of a simple neural network from scratch using Python and NumPy. The neural network is trained to classify the Iris dataset using backpropagation.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Code Explanation](#code-explanation)
- [Backpropagation Algorithm](#backpropagation-algorithm)
- [Results](#results)

## Overview

This project includes a neural network class that performs the following tasks:
1. Loads the Iris dataset.
2. Preprocesses the data by normalizing features and one-hot encoding the target labels.
3. Splits the data into training and testing sets.
4. Initializes and trains a neural network using backpropagation.
5. Evaluates the neural network's performance on the test data.

## Requirements

- Python 3.x
- NumPy
- scikit-learn
- matplotlib




## Code Explanation

### Neural Network Class

The `NeuralNetwork` class is defined to encapsulate the entire neural network model, including its architecture and learning algorithms. Here's a brief overview of its components:

- **Initialization (`__init__`)**: Sets up the neural network's structure by initializing weights and biases.
- **Activation Functions**: Uses the sigmoid function for activation and its derivative for backpropagation.
- **Feedforward (`feedforward`)**: Computes the outputs of the neural network given an input.
- **Backward (`backward`)**: Updates weights and biases based on the error between predicted and actual outputs using the backpropagation algorithm.
- **Training (`train`)**: Trains the neural network over a specified number of epochs.

### Data Preprocessing

1. **Loading the Iris Dataset**: The Iris dataset is loaded using `scikit-learn`.
2. **One-Hot Encoding**: The target labels are converted to a one-hot encoded format.
3. **Normalization**: Features are standardized to have zero mean and unit variance.
4. **Data Splitting**: The dataset is split into training and testing sets.

### Training and Evaluation

The neural network is trained using the training set, and its performance is evaluated on the testing set. Accuracy is calculated to measure the model's performance.

## Backpropagation Algorithm

Backpropagation is the algorithm used to train the neural network by updating its weights and biases. Here's a detailed explanation of how it works:

1. **Forward Pass**:
    - Calculate the input to the hidden layer: `hidden_activation = X.dot(weights_input_hidden) + bias_hidden`
    - Apply the sigmoid activation function: `hidden_output = sigmoid(hidden_activation)`
    - Calculate the input to the output layer: `output_activation = hidden_output.dot(weights_hidden_output) + bias_output`
    - Apply the sigmoid activation function: `predicted_output = sigmoid(output_activation)`

2. **Backward Pass**:
    - Compute the output error: `output_error = y - predicted_output`
    - Calculate the gradient of the loss with respect to the output layer's activation: `output_delta = output_error * sigmoid_derivative(predicted_output)`
    - Compute the hidden layer error: `hidden_error = output_delta.dot(weights_hidden_output.T)`
    - Calculate the gradient of the loss with respect to the hidden layer's activation: `hidden_delta = hidden_error * sigmoid_derivative(hidden_output)`
    - Update weights and biases:
      ```python
      weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
      bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
      weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
      bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
      ```

The process iterates over a specified number of epochs, gradually reducing the error by adjusting the weights and biases.

## Results

The neural network is evaluated on the test set, and the accuracy of the model is printed.
```
Epoch 0, Loss:0.24193
...
Epoch 999, Loss:0.02549
Accuracy: 98.666667%
```

This shows that the neural network has learned to classify the Iris dataset with high accuracy.


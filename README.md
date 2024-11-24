

# Perceptron Activation Functions Classification

## Overview
This repository contains implementations of different Perceptron models with various activation functions. The project explores the role of activation functions in training a neural network and evaluates their performance on a synthetic binary classification problem.

The Perceptron models are trained using the following activation functions:
1. **Sigmoid**
2. **Tanh**
3. **ReLU**
4. **Leaky ReLU**

Each activation function is used to classify a synthetic dataset, and the accuracy of the models is compared. The dataset is generated randomly and split into training and testing sets.

## Project Structure
The repository includes the following files:

- **`perceptron_activation_functions.py`**: Python code for implementing Perceptron models with various activation functions, training them, and evaluating their performance.
- **`README.md`**: This file containing an overview, instructions, and explanations.
- **`requirements.txt`**: List of Python dependencies required to run the code (e.g., NumPy, Scikit-learn, Matplotlib).

## Installation

To run the code, you need to have Python installed along with the necessary dependencies. You can install the required packages using `pip`.

### Clone the Repository
```bash
git clone https://github.com/Bushra-Butt-17/Perceptron-Activation-Functions-Classification.git
cd Perceptron-Activation-Functions-Classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

To train the perceptron models with different activation functions, run the Python script `perceptron_activation_functions.py`. This script generates synthetic data, splits it into training and testing sets, and trains perceptron models with the specified activation functions.

### Example Command to Run the Script:
```bash
python perceptron_activation_functions.py
```

This will output the classification accuracy of each perceptron model along with plots of the decision boundaries.

## Activation Functions

### 1. Sigmoid
The **Sigmoid** function maps input values to the range between 0 and 1. It is commonly used in binary classification problems where the output represents a probability. It can cause vanishing gradient problems in deep networks but works well for single-layer perceptrons.

**Formula**:  
`sigmoid(x) = 1 / (1 + exp(-x))`

### 2. Tanh
The **Tanh** function maps input values to the range between -1 and 1. It is similar to the Sigmoid function but with better performance because it centers the data around zero. This makes it more suitable for deep networks than Sigmoid.

**Formula**:  
`tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

### 3. ReLU
The **ReLU** (Rectified Linear Unit) function is one of the most popular activation functions due to its simplicity and effectiveness. It returns the input if it is positive, otherwise returns 0. It helps mitigate the vanishing gradient problem, making it useful for deep networks.

**Formula**:  
`relu(x) = max(0, x)`

### 4. Leaky ReLU
**Leaky ReLU** is a variation of ReLU that allows a small negative slope for input values less than 0, which helps avoid "dead neurons" (i.e., neurons that don't update during training).

**Formula**:  
`leaky_relu(x) = x if x > 0 else alpha * x`  
where `alpha` is a small constant (usually 0.01).

## Evaluation

Each model is evaluated on the following metrics:
- **Accuracy**: The proportion of correctly classified instances in the test set.
- **Decision Boundary Visualization**: A plot showing how the perceptron model separates the data points.

## Epochs, Forward Propagation, Backward Propagation

### Epochs
An **epoch** refers to one complete pass through the entire training dataset. In each epoch, the perceptron model updates its weights to minimize the error. Typically, the training process involves multiple epochs to improve model accuracy.

### Forward Propagation
**Forward propagation** refers to the process of passing inputs through the network, layer by layer, to generate an output. In a perceptron, this involves computing a weighted sum of inputs, passing it through an activation function, and producing the output.

### Backward Propagation
**Backward propagation** is the process of adjusting the model's weights based on the error or loss. The error is propagated backward through the network to update the weights using gradient descent.

## XOR Problem

The XOR problem is a well-known binary classification problem that a single-layer perceptron cannot solve. A solution can be implemented using a **Multi-Layer Perceptron** (MLP) with more than one layer, which can handle non-linear separability.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Scikit-learn](https://scikit-learn.org/) for providing machine learning tools and datasets.
- [NumPy](https://numpy.org/) for efficient numerical computations.
- [Matplotlib](https://matplotlib.org/) for plotting decision boundaries.


## Images

### 1. Basic Perceptron Model

![Basic Perceptron](https://github.com/Bushra-Butt-17/Perceptron-Activation-Functions-Classification/raw/main/basic_perceptron_image.png)
_A diagram of a basic perceptron._

### 2. Activation Functions

![Activation Functions](https://github.com/Bushra-Butt-17/Perceptron-Activation-Functions-Classification/raw/main/ActivationFunctions.png)
_A diagram showing decision boundaries of various activation functions: Sigmoid, Tanh, ReLU, and Leaky ReLU._
```

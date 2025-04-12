# -*- coding: utf-8 -*-
"""
simple_ann.py: My From-Scratch Simple ANN Class

Okay, this file holds the core logic for my simple Artificial Neural Network.
It's designed for binary classification with one hidden layer.

Key Components:
- Initialization of weights and biases.
- Activation functions (Tanh for hidden, Sigmoid for output) and their derivatives.
- Forward propagation: Calculating predictions.
- Cost function: Binary Cross-Entropy (Log Loss), possibly with L2 regularization.
- Backward propagation: Calculating gradients using the chain rule.
- Parameter update: Using Gradient Descent.
"""

import numpy as np

class SimpleANN:
    """
    My implementation of a simple ANN with one hidden layer.

    Attributes:
        input_size (int): Number of features in the input data.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of neurons in the output layer (1 for binary classification).
        W1, b1: Weights and bias for the hidden layer.
        W2, b2: Weights and bias for the output layer.
        cache (dict): Stores intermediate values from forward prop for use in backprop.
        losses (list): Stores the loss value during training epochs.
        accuracies (list): Stores the training accuracy during training epochs.
    """

    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Initialize the network's structure and parameters.
        Args:
            input_size (int): Dimension of the input layer (number of features).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Dimension of the output layer (should be 1 for binary).
        """
        if output_size != 1:
            print("Warning: For binary classification, output_size should typically be 1.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # --- Parameter Initialization ---
        # I need to initialize weights and biases.
        # Weights: Small random numbers to break symmetry. Multiplying by 0.01 keeps them small.
        # Biases: Can initialize to zero.
        # Setting a seed for reproducibility during initialization if needed elsewhere.
        # np.random.seed(42) # Optional: for consistent random initialization

        # Layer 1 (Input -> Hidden)
        # W1 shape: (n_hidden, n_input)
        # b1 shape: (n_hidden, 1)
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b1 = np.zeros((self.hidden_size, 1))

        # Layer 2 (Hidden -> Output)
        # W2 shape: (n_output, n_hidden)
        # b2 shape: (n_output, 1)
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b2 = np.zeros((self.output_size, 1))

        print(f"Initialized ANN: Input({input_size}) -> Hidden({hidden_size}) -> Output({output_size})")
        print(f"  W1 shape: {self.W1.shape}, b1 shape: {self.b1.shape}")
        print(f"  W2 shape: {self.W2.shape}, b2 shape: {self.b2.shape}")

        self.cache = {} # To store intermediate values needed for backprop
        self.losses = [] # To track training loss
        self.accuracies = [] # To track training accuracy


    # --- Activation Functions ---
    def _sigmoid(self, Z):
        """Sigmoid activation function."""
        A = 1 / (1 + np.exp(-Z))
        return A

    def _sigmoid_derivative(self, Z):
        """Derivative of the sigmoid function."""
        s = self._sigmoid(Z)
        return s * (1 - s)

    def _tanh(self, Z):
        """Tanh activation function."""
        return np.tanh(Z)

    def _tanh_derivative(self, Z):
        """Derivative of the tanh function."""
        # derivative = 1 - tanh(Z)^2
        t = np.tanh(Z)
        return 1 - np.power(t, 2)

    def _relu(self, Z):
        """ReLU activation function (Optional, Tanh is used by default here)."""
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """Derivative of ReLU."""
        dZ = np.array(Z, copy=True) # Just modifying the array suffices.
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ


    # --- Forward Propagation ---
    def forward_propagation(self, X):
        """
        Performs one forward pass through the network.
        Args:
            X (np.ndarray): Input data. Shape (input_size, number_of_examples).
        Returns:
            A2 (np.ndarray): Output of the sigmoid activation (predictions). Shape (1, number_of_examples).
        """
        # Retrieve weights and biases
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # Layer 1 (Hidden Layer)
        # Linear step: Z = W*X + b
        Z1 = np.dot(W1, X) + b1
        # Activation step (using Tanh here)
        A1 = self._tanh(Z1)
        # A1 = self._relu(Z1) # If I wanted to use ReLU instead

        # Layer 2 (Output Layer)
        # Linear step
        Z2 = np.dot(W2, A1) + b2
        # Activation step (using Sigmoid for binary classification output)
        A2 = self._sigmoid(Z2) # This is Y_hat (predicted probability)

        # Store intermediate values in cache for backpropagation
        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2


    # --- Cost Function ---
    def compute_cost(self, A2, Y, lambd=0):
        """
        Computes the cost (Binary Cross-Entropy Loss).
        Args:
            A2 (np.ndarray): Predictions from forward prop. Shape (1, number_of_examples).
            Y (np.ndarray): True labels. Shape (1, number_of_examples).
            lambd (float): L2 regularization hyperparameter.
        Returns:
            cost (float): The computed cost.
        """
        m = Y.shape[1] # Number of examples

        # Basic Cross-Entropy cost
        # Adding a small epsilon to log arguments to avoid log(0)
        epsilon = 1e-8
        logprobs = np.multiply(np.log(A2 + epsilon), Y) + np.multiply(np.log(1 - A2 + epsilon), (1 - Y))
        cross_entropy_cost = - (1/m) * np.sum(logprobs)

        # Optional L2 Regularization Cost
        l2_regularization_cost = 0
        if lambd > 0:
            # Sum of squares of weights (excluding biases)
            W1_norm = np.sum(np.square(self.W1))
            W2_norm = np.sum(np.square(self.W2))
            l2_regularization_cost = (lambd / (2 * m)) * (W1_norm + W2_norm)

        cost = cross_entropy_cost + l2_regularization_cost
        cost = np.squeeze(cost) # Ensure it's a scalar

        return cost


    # --- Backward Propagation ---
    def backward_propagation(self, X, Y, lambd=0):
        """
        Performs one backward pass to compute gradients.
        Args:
            X (np.ndarray): Input data. Shape (input_size, number_of_examples).
            Y (np.ndarray): True labels. Shape (1, number_of_examples).
            lambd (float): L2 regularization hyperparameter.
        Returns:
            grads (dict): Dictionary containing gradients w.r.t. parameters (dW1, db1, dW2, db2).
        """
        m = X.shape[1] # Number of examples

        # Retrieve weights and cached values
        W1, W2 = self.W1, self.W2
        Z1, A1, Z2, A2 = self.cache["Z1"], self.cache["A1"], self.cache["Z2"], self.cache["A2"]

        # --- Gradient Calculations (Chain Rule) ---

        # Start from the output layer (Layer 2)
        # Derivative of Cost w.r.t A2 (Prediction)
        # dL/dA2 = -(Y/A2 - (1-Y)/(1-A2))
        dA2 = - (np.divide(Y, A2 + 1e-8) - np.divide(1 - Y, 1 - A2 + 1e-8)) # Added epsilon for stability

        # Derivative of Cost w.r.t Z2 (Linear output of Layer 2)
        # dL/dZ2 = dL/dA2 * dA2/dZ2 = dL/dA2 * sigmoid_derivative(Z2)
        dZ2 = dA2 * self._sigmoid_derivative(Z2)

        # Derivative of Cost w.r.t W2
        # dL/dW2 = 1/m * dL/dZ2 * dZ2/dW2 = 1/m * dZ2 @ A1.T
        # Add regularization gradient: (lambda/m) * W2
        dW2 = (1/m) * np.dot(dZ2, A1.T) + (lambd/m) * W2

        # Derivative of Cost w.r.t b2
        # dL/db2 = 1/m * sum(dL/dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Move to the hidden layer (Layer 1)
        # Derivative of Cost w.r.t A1 (Activation of Layer 1)
        # dL/dA1 = dL/dZ2 * dZ2/dA1 = W2.T @ dZ2
        dA1 = np.dot(W2.T, dZ2)

        # Derivative of Cost w.r.t Z1 (Linear output of Layer 1)
        # dL/dZ1 = dL/dA1 * dA1/dZ1 = dL/dA1 * tanh_derivative(Z1)
        dZ1 = dA1 * self._tanh_derivative(Z1)
        # dZ1 = dA1 * self._relu_derivative(Z1) # If using ReLU

        # Derivative of Cost w.r.t W1
        # dL/dW1 = 1/m * dL/dZ1 * dZ1/dW1 = 1/m * dZ1 @ X.T
        # Add regularization gradient: (lambda/m) * W1
        dW1 = (1/m) * np.dot(dZ1, X.T) + (lambd/m) * W1

        # Derivative of Cost w.r.t b1
        # dL/db1 = 1/m * sum(dL/dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Store gradients
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads


    # --- Parameter Update ---
    def update_parameters(self, grads, learning_rate):
        """
        Updates parameters using gradient descent.
        Args:
            grads (dict): Gradients computed from backprop.
            learning_rate (float): The step size for gradient descent.
        """
        # Retrieve gradients
        dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]

        # Update rule: parameter = parameter - learning_rate * gradient
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2


    # --- Training Loop ---
    def train(self, X_train, Y_train, num_epochs, learning_rate, lambd=0, print_cost_every=100):
        """
        Trains the neural network.
        Args:
            X_train (np.ndarray): Training data features. Shape (input_size, num_examples).
            Y_train (np.ndarray): Training data labels. Shape (1, num_examples).
            num_epochs (int): Number of passes through the entire dataset.
            learning_rate (float): Controls the step size for parameter updates.
            lambd (float): L2 regularization strength (lambda). 0 means no regularization.
            print_cost_every (int): How often to print the cost during training.
        """
        print(f"\n--- Starting Training ---")
        print(f"Number of epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"L2 Regularization lambda: {lambd}")

        self.losses = [] # Reset loss history
        self.accuracies = [] # Reset accuracy history

        for i in range(num_epochs):
            # 1. Forward Propagation
            A2 = self.forward_propagation(X_train)

            # 2. Compute Cost
            cost = self.compute_cost(A2, Y_train, lambd)

            # 3. Backward Propagation
            grads = self.backward_propagation(X_train, Y_train, lambd)

            # 4. Update Parameters
            self.update_parameters(grads, learning_rate)

            # --- Record Metrics and Print Progress ---
            if i % print_cost_every == 0 or i == num_epochs - 1:
                # Calculate accuracy on training set for monitoring
                predictions = self.predict(X_train)
                accuracy = np.mean(predictions == Y_train) * 100
                print(f"Epoch {i}/{num_epochs} - Cost: {cost:.6f} - Training Accuracy: {accuracy:.2f}%")
                self.losses.append(cost)
                self.accuracies.append(accuracy)

        print("--- Training Finished ---")


    # --- Prediction ---
    def predict(self, X):
        """
        Makes predictions on new data using the trained parameters.
        Args:
            X (np.ndarray): Input data. Shape (input_size, num_examples).
        Returns:
            predictions (np.ndarray): Binary predictions (0 or 1). Shape (1, num_examples).
        """
        # Perform forward propagation
        A2 = self.forward_propagation(X)

        # Convert probabilities (A2) to binary predictions (0 or 1)
        # Threshold is 0.5
        predictions = (A2 > 0.5).astype(int)

        return predictions
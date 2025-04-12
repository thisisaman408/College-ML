# -*- coding: utf-8 -*-
"""
train_ann.py: Training and Evaluating My From-Scratch ANN

Okay, this is the main script where I'll:
1. Generate some synthetic data suitable for binary classification.
2. Preprocess the data (split into train/test, maybe scale - though Tanh/Sigmoid are less sensitive than some others).
3. Import and instantiate my SimpleANN class from simple_ann.py.
4. Train the network on the training data.
5. Evaluate the trained network on the test data.
6. Visualize the results: loss curve, accuracy curve, and decision boundary.
7. Repeat steps 4-6 with L2 regularization enabled and compare.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import os

# Import my ANN class
from simple_ann import SimpleANN

# --- 1. Setup Environment ---
print("--- Setting up Environment ---")
output_dir = 'ANN_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: '{output_dir}' to save my plots.")
else:
    print(f"Directory '{output_dir}' already exists. Plots will be saved there.")

# Set a random seed for reproducibility of data generation and splitting
np.random.seed(42)


# --- 2. Generate Synthetic Data ---
print("\n--- Generating Synthetic Data ---")
# Let's use make_classification for more control
# I want a reasonably large dataset (e.g., 1500 samples) with 2 features (for easy visualization).
# I'll make it binary classification (n_classes=2).
# n_clusters_per_class=1 makes it simpler, maybe some noise (flip_y)
# n_informative=2 means both features are useful.
X, Y = sklearn.datasets.make_classification(
    n_samples=1500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.05, # Introduce a little noise
    class_sep=0.9, # Make classes not trivially separable
    random_state=42
)

print(f"Generated data shape: X={X.shape}, Y={Y.shape}")
print(f"Number of samples in each class: {np.bincount(Y)}")

# Reshape Y to be a row vector (1, n_samples) as expected by my ANN
Y = Y.reshape(1, -1)
print(f"Reshaped Y shape: {Y.shape}")

# Visualize the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k', s=40)
plt.title('Generated Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plot_path = os.path.join(output_dir, '00_synthetic_data.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved synthetic data plot to: {plot_path}")


# --- 3. Preprocessing ---
print("\n--- Preprocessing Data ---")
# Split into training and testing sets (e.g., 80% train, 20% test)
X_train_orig, X_test_orig, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, Y.T, test_size=0.20, random_state=42, stratify=Y.T
) # Use Y.T for splitting as train_test_split expects (n_samples, n_features)

# Transpose X to match the expected input shape (n_features, n_samples) for my ANN
X_train = X_train_orig.T
X_test = X_test_orig.T
Y_train = Y_train.T # Back to (1, n_samples)
Y_test = Y_test.T   # Back to (1, n_samples)

print(f"Data split complete:")
print(f"  X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"  X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Feature Scaling (Optional but often good practice)
# Although Tanh/Sigmoid are less sensitive than, say, distance-based methods,
# scaling can sometimes help gradient descent converge faster/better.
# Let's try StandardScaler. Fit ONLY on training data.
scaler = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.T).T # Scale expects (n_samples, n_features)
X_test_scaled = scaler.transform(X_test.T).T      # Transform test data using the same scaler

print("Applied StandardScaler to features.")
print(f"  X_train_scaled shape: {X_train_scaled.shape}")
print(f"  X_test_scaled shape: {X_test_scaled.shape}")

# Use scaled data from now on
X_train_final = X_train_scaled
X_test_final = X_test_scaled


# --- Helper function for Plotting Decision Boundary ---
def plot_decision_boundary(model, X, Y, title, filename):
    """Plots the decision boundary of a trained model."""
    plt.figure(figsize=(8, 6))

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.01 # Step size in the mesh

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    # Need to transpose the grid points to match model input format (n_features, n_samples)
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k', s=40)
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Saved decision boundary plot to: {filename}")

# --- Helper function for Plotting Training Curves ---
def plot_training_curves(losses, accuracies, title_prefix, filename):
    """Plots the loss and accuracy curves during training."""
    epochs_recorded = range(0, len(losses) * 100, 100) # Assuming recorded every 100 epochs
    if not epochs_recorded: # Handle case with few epochs
         epochs_recorded = [0]
         if len(losses) > 1:
             epochs_recorded = range(len(losses)) # Adjust if print_cost_every is different

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(epochs_recorded, losses, color=color, marker='o', label='Cost')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Training Accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_recorded, accuracies, color=color, marker='x', linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    # Set accuracy limits potentially
    ax2.set_ylim(min(max(0, min(accuracies)-10), 100), 101)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'{title_prefix} - Training Cost and Accuracy')
    # Add legend - need to get handles from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.savefig(filename)
    plt.close()
    print(f"Saved training curves plot to: {filename}")


# --- 4. Train WITHOUT Regularization ---
print("\n\n--- Training ANN without Regularization ---")
input_dim = X_train_final.shape[0] # Number of features
hidden_nodes = 10 # Let's try 10 hidden neurons
output_dim = 1 # Binary classification

# Create the ANN instance
ann_no_reg = SimpleANN(input_size=input_dim, hidden_size=hidden_nodes, output_size=output_dim)

# Train the model
learning_rate = 0.1
num_epochs = 10000 # Need sufficient epochs for convergence
ann_no_reg.train(X_train_final, Y_train, num_epochs=num_epochs, learning_rate=learning_rate, lambd=0, print_cost_every=1000)

# --- 5. Evaluate WITHOUT Regularization ---
print("\n--- Evaluating ANN without Regularization ---")
# Predictions on test set
Y_pred_test_no_reg = ann_no_reg.predict(X_test_final)
test_accuracy_no_reg = np.mean(Y_pred_test_no_reg == Y_test) * 100
print(f"Test Accuracy (No Regularization): {test_accuracy_no_reg:.2f}%")

# --- 6. Visualize WITHOUT Regularization ---
print("\n--- Visualizing Results (No Regularization) ---")
# Plot training curves
plot_training_curves(
    ann_no_reg.losses, ann_no_reg.accuracies,
    title_prefix='No Regularization',
    filename=os.path.join(output_dir, '01_training_curves_no_reg.png')
)
# Plot decision boundary (using scaled training data for visualization consistency)
plot_decision_boundary(
    ann_no_reg, X_train_final, Y_train,
    title='Decision Boundary (No Regularization - Train Set)',
    filename=os.path.join(output_dir, '02_decision_boundary_no_reg_train.png')
)
# Plot decision boundary on test data
plot_decision_boundary(
    ann_no_reg, X_test_final, Y_test,
    title='Decision Boundary (No Regularization - Test Set)',
    filename=os.path.join(output_dir, '03_decision_boundary_no_reg_test.png')
)


# --- 7. Train WITH L2 Regularization ---
print("\n\n--- Training ANN WITH L2 Regularization ---")
# Let's try a small lambda value first
lambda_l2 = 0.1 # Regularization strength

# Create a new ANN instance (important to start fresh!)
ann_with_reg = SimpleANN(input_size=input_dim, hidden_size=hidden_nodes, output_size=output_dim)

# Train the regularized model
# Might need same or different number of epochs/learning rate - let's start with the same
ann_with_reg.train(X_train_final, Y_train, num_epochs=num_epochs, learning_rate=learning_rate, lambd=lambda_l2, print_cost_every=1000)

# --- 8. Evaluate WITH Regularization ---
print("\n--- Evaluating ANN WITH L2 Regularization ---")
# Predictions on test set
Y_pred_test_with_reg = ann_with_reg.predict(X_test_final)
test_accuracy_with_reg = np.mean(Y_pred_test_with_reg == Y_test) * 100
print(f"Test Accuracy (L2 Regularization, lambda={lambda_l2}): {test_accuracy_with_reg:.2f}%")

# --- 9. Visualize WITH Regularization ---
print("\n--- Visualizing Results (L2 Regularization) ---")
# Plot training curves
plot_training_curves(
    ann_with_reg.losses, ann_with_reg.accuracies,
    title_prefix=f'L2 Regularization (lambda={lambda_l2})',
    filename=os.path.join(output_dir, f'04_training_curves_L2reg_lambda{lambda_l2}.png')
)
# Plot decision boundary (using scaled training data for visualization consistency)
plot_decision_boundary(
    ann_with_reg, X_train_final, Y_train,
    title=f'Decision Boundary (L2 Reg lambda={lambda_l2} - Train Set)',
    filename=os.path.join(output_dir, f'05_decision_boundary_L2reg_lambda{lambda_l2}_train.png')
)
# Plot decision boundary on test data
plot_decision_boundary(
    ann_with_reg, X_test_final, Y_test,
    title=f'Decision Boundary (L2 Reg lambda={lambda_l2} - Test Set)',
    filename=os.path.join(output_dir, f'06_decision_boundary_L2reg_lambda{lambda_l2}_test.png')
)

# --- 10. Comparison Summary ---
print("\n\n--- Comparison Summary ---")
print(f"Test Accuracy WITHOUT Regularization: {test_accuracy_no_reg:.2f}%")
print(f"Test Accuracy WITH L2 Regularization (lambda={lambda_l2}): {test_accuracy_with_reg:.2f}%")
print("\nComparison Notes:")
print("- Check the 'ANN_Outputs' folder for plots.")
print("- Compare the 'Training Curves' plots: Does L2 regularization lead to higher final cost but potentially similar/better accuracy (less overfitting)?")
print("- Compare the 'Decision Boundary' plots: Is the boundary smoother with L2 regularization? Does it seem to generalize better (less sensitive to individual noisy points)?")
if lambda_l2 > 0 and test_accuracy_with_reg > test_accuracy_no_reg:
     print("- In this run, L2 regularization improved test accuracy, suggesting it helped generalization.")
elif lambda_l2 > 0 and test_accuracy_with_reg < test_accuracy_no_reg:
     print("- In this run, L2 regularization slightly hurt test accuracy. The model might have been underfitting or lambda was too high.")
else:
     print("- In this run, L2 regularization had minimal impact on test accuracy. The dataset/model might not have been prone to overfitting.")

print("\n--- Analysis Complete ---")
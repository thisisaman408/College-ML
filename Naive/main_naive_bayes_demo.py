# -*- coding: utf-8 -*-
"""
File: main_naive_bayes_demo.py

Okay, this is my main script to demonstrate Naive Bayes.
I will:
1. Import my own NaiveBayesClassifier from naive_bayes_scratch.py.
2. Generate some synthetic classification data. Since my scratch NB works
   best with categorical features, I'll generate numerical data and then
   *discretize* it into bins.
3. Split the data into training and testing sets.
4. Train my scratch Naive Bayes classifier on the discretized training data.
5. Make predictions on the discretized test data.
6. Evaluate my classifier (accuracy, confusion matrix, classification report).
7. Visualize the confusion matrix and save it.
8. Compare conceptually, and also compare performance against scikit-learn's
   CategoricalNB on the same discretized data.
9. Save all plots into the 'NaiveBayes_Outputs' folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import necessary libraries and my own classifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import CategoricalNB # For comparison

# Import my custom Naive Bayes implementation
try:
    from naive_bayes_scratch import NaiveBayesClassifier
    print("Successfully imported my NaiveBayesClassifier from naive_bayes_scratch.py")
except ImportError:
    print("ERROR: Could not import NaiveBayesClassifier.")
    print("Make sure 'naive_bayes_scratch.py' is in the same directory or Python path.")
    exit() # Can't proceed without the classifier

# --- Setup: Output Directory ---
output_dir = 'NaiveBayes_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: '{output_dir}' to save my plots.")
else:
    print(f"Directory '{output_dir}' already exists. Plots will be saved there.")

# --- 2. Generate and Prepare Data ---
print("\n--- Generating and Preparing Data ---")
n_samples = 500
n_features = 10 # Original number of numerical features
n_classes = 3   # Number of target classes
n_informative = 5 # Number of informative features
random_seed = 42

print(f"Generating synthetic dataset with:")
print(f" - {n_samples} samples")
print(f" - {n_features} numerical features")
print(f" - {n_classes} classes")
print(f" - {n_informative} informative features")
print(f" - Random seed: {random_seed}")

X_num, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=2,
    n_repeated=0,
    n_classes=n_classes,
    n_clusters_per_class=2,
    flip_y=0.02, # Add some noise to labels
    class_sep=0.8, # Make classes not too easy to separate
    random_state=random_seed
)
print("Generated numerical data.")

# Discretize features: Convert numerical features into categorical bins.
# This is necessary for my from-scratch classifier which expects discrete inputs.
n_bins = 5 # How many bins to create for each feature
print(f"Discretizing numerical features into {n_bins} bins using KBinsDiscretizer...")
# Strategy 'uniform' means bins have equal width. 'quantile' means equal number of samples per bin.
# 'kmeans' uses 1D K-Means clustering. Let's use quantile.
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None) # Use subsample=None to avoid warning with newer scikit-learn versions

# Fit on the whole numerical data (or just training data - fitting on whole is common for unsupervised steps)
# Let's fit on the whole data here for simplicity before splitting.
X_cat = discretizer.fit_transform(X_num)
print("Discretization complete. Features are now ordinal encoded bins (0, 1, 2...).")
print(f"Shape of discretized data: {X_cat.shape}")
print(f"Example first 5 rows of discretized data:\n{X_cat[:5]}")

# --- 3. Split Data ---
print("\n--- Splitting Data into Training and Test Sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_cat, y, test_size=0.3, random_state=random_seed, stratify=y # Stratify ensures class proportion is similar in train/test
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 4. Train My Scratch Naive Bayes Classifier ---
print("\n--- Training My From-Scratch Naive Bayes Classifier ---")
# Initialize my classifier with alpha=1.0 (Laplace smoothing)
my_nb_classifier = NaiveBayesClassifier(alpha=1.0)

# Train it on the discretized training data
my_nb_classifier.fit(X_train, y_train)
# The detailed print statements inside the fit method should show the progress.

# --- 5. Make Predictions with My Classifier ---
print("\n--- Making Predictions with My Classifier on Test Data ---")
y_pred_my_nb = my_nb_classifier.predict(X_test)
# The predict method also has print statements.
print(f"Predicted labels for the first 10 test samples: {y_pred_my_nb[:10]}")
print(f"Actual labels for the first 10 test samples:   {y_test[:10]}")

# --- 6. Evaluate My Classifier ---
print("\n--- Evaluating My From-Scratch Naive Bayes Classifier ---")
accuracy_my_nb = accuracy_score(y_test, y_pred_my_nb)
cm_my_nb = confusion_matrix(y_test, y_pred_my_nb)
report_my_nb = classification_report(y_test, y_pred_my_nb)

print(f"Accuracy of my Naive Bayes: {accuracy_my_nb:.4f}")
print("\nClassification Report (My Naive Bayes):\n", report_my_nb)
print("\nConfusion Matrix (My Naive Bayes):")
print(cm_my_nb)

# --- 7. Visualize Confusion Matrix (My Classifier) ---
print("\n--- Visualizing Confusion Matrix (My Classifier) ---")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_my_nb, annot=True, fmt='d', cmap='Blues',
            xticklabels=my_nb_classifier.classes_, yticklabels=my_nb_classifier.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - My Naive Bayes (alpha={my_nb_classifier.alpha})\nAccuracy: {accuracy_my_nb:.3f}")
plot_path_my_nb = os.path.join(output_dir, '01_confusion_matrix_my_NB.png')
plt.savefig(plot_path_my_nb, bbox_inches='tight')
plt.close() # Close the plot
print(f"Saved confusion matrix plot to: {plot_path_my_nb}")
print("The confusion matrix helps me see where my model made mistakes (off-diagonal elements).")

# --- 8. Comparison with Scikit-learn's CategoricalNB ---
print("\n--- Comparing with Scikit-learn's CategoricalNB ---")
print("Now, let's train scikit-learn's CategoricalNB on the *same discretized data* for comparison.")
print("This will help validate if my implementation gives reasonable results.")

# Initialize sklearn's CategoricalNB. It also has an alpha for smoothing.
# Important: sklearn's CategoricalNB expects features to start from 0.
# Our KBinsDiscretizer with encode='ordinal' already does this.
# sklearn's alpha defaults to 1.0, just like mine.
sklearn_nb = CategoricalNB(alpha=1.0)
print("\nTraining scikit-learn's CategoricalNB...")
sklearn_nb.fit(X_train, y_train)
print("Training complete.")

print("\nMaking predictions with scikit-learn's CategoricalNB...")
y_pred_sklearn = sklearn_nb.predict(X_test)

print("\nEvaluating scikit-learn's CategoricalNB...")
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
report_sklearn = classification_report(y_test, y_pred_sklearn)

print(f"Accuracy of sklearn's CategoricalNB: {accuracy_sklearn:.4f}")
print("\nClassification Report (Sklearn CategoricalNB):\n", report_sklearn)
print("\nConfusion Matrix (Sklearn CategoricalNB):")
print(cm_sklearn)

# Visualize Sklearn's Confusion Matrix
print("\nVisualizing Confusion Matrix (Sklearn Classifier)...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens',
            xticklabels=sklearn_nb.classes_, yticklabels=sklearn_nb.classes_) # Use classes_ from sklearn model
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Sklearn CategoricalNB (alpha={sklearn_nb.alpha})\nAccuracy: {accuracy_sklearn:.3f}")
plot_path_sklearn = os.path.join(output_dir, '02_confusion_matrix_sklearn_NB.png')
plt.savefig(plot_path_sklearn, bbox_inches='tight')
plt.close() # Close the plot
print(f"Saved confusion matrix plot to: {plot_path_sklearn}")


# --- 9. Final Comparison Discussion ---
print("\n--- Final Comparison Discussion ---")
print(f"My Scratch NB Accuracy: {accuracy_my_nb:.4f}")
print(f"Sklearn CatNB Accuracy: {accuracy_sklearn:.4f}")

accuracy_diff = abs(accuracy_my_nb - accuracy_sklearn)
print(f"Absolute difference in accuracy: {accuracy_diff:.4f}")

if accuracy_diff < 0.01: # Allowing for small floating point differences
    print("\nConclusion: My from-scratch implementation's performance is very close to scikit-learn's implementation!")
    print("This suggests my understanding and implementation of Naive Bayes with Laplace smoothing are likely correct.")
else:
    print("\nConclusion: There is a noticeable difference between my implementation and scikit-learn's.")
    print("This could be due to subtle differences in handling edge cases, floating point precision, or the exact smoothing application.")
    print("It's worth double-checking my math, especially in the likelihood calculation and handling of unseen values (though less likely with discretization).")

print("\nKey takeaways from implementing Naive Bayes from scratch:")
print(" - Understood the core calculation: P(Class | Features) ~ P(Class) * Product[P(Feature_i | Class)]")
print(" - Implemented prior and likelihood calculations.")
print(" - Saw the importance of Laplace smoothing (alpha > 0) to prevent zero probabilities, especially for feature values not seen with a specific class during training.")
print(" - Used log probabilities to avoid numerical underflow when multiplying many small likelihoods.")
print(" - Realized the need to handle categorical data, hence the discretization step for the generated numerical data.")
print(" - Comparing against a standard library implementation is a great way to validate my own code.")


print("\n\n--- Naive Bayes Demonstration Complete ---")
print(f"All plots saved in the '{output_dir}' folder.")
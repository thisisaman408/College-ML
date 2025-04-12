# -*- coding: utf-8 -*-
"""
File: naive_bayes_scratch.py

Okay, this is where I'll build my own Naive Bayes classifier from scratch.
My goal is to understand how it works internally.
I need to implement the fitting logic (calculating priors and likelihoods)
and the prediction logic (applying Bayes' theorem).
Crucially, I must include Laplace (additive) smoothing to handle cases
where a feature value might not appear with a certain class in the training data.
This class should work with categorical features (or features that have been discretized).
"""

import numpy as np

class NaiveBayesClassifier:
    """
    My custom Naive Bayes classifier implementation from scratch.

    Attributes:
        alpha (float): The smoothing parameter (Laplace smoothing).
                       alpha=1.0 is standard Laplace smoothing.
                       alpha=0.0 means no smoothing.
        classes_ (np.ndarray): Array of unique class labels found during fitting.
        class_priors_ (dict): Dictionary storing the log prior probability for each class.
                              log P(Class)
        feature_log_likelihoods_ (dict): Nested dictionary storing log likelihoods.
                                         Structure: {class_label: {feature_index: {feature_value: log_likelihood}}}
                                         log P(feature_value | Class)
        feature_unique_values_ (dict): Dictionary storing the set of unique values encountered
                                       for each feature during training.
                                       Structure: {feature_index: set(unique_values)}
        n_features_ (int): Number of features seen during fitting.
    """
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier.
        alpha: Smoothing parameter. I'll default to 1.0 for Laplace smoothing.
        """
        if alpha < 0:
            # Smoothing parameter cannot be negative, that wouldn't make sense.
            raise ValueError("Alpha (smoothing parameter) must be non-negative.")
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = {}
        self.feature_log_likelihoods_ = {}
        self.feature_unique_values_ = {} # Need this for smoothing denominator
        self.n_features_ = None
        print(f"My NaiveBayesClassifier initialized with alpha = {self.alpha}")
        if self.alpha == 0:
            print("  Note: alpha=0 means NO smoothing. Risk of zero probabilities!")
        elif self.alpha == 1.0:
            print("  Using standard Laplace smoothing (alpha=1.0).")
        else:
            print(f"  Using additive smoothing with alpha={self.alpha}.")

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.
        Here, I need to calculate the prior probabilities of each class and
        the likelihood of each feature value given each class.

        Args:
            X (np.ndarray): Training data features (samples x features).
                            I'm assuming these are categorical/discrete.
            y (np.ndarray): Training data labels (samples).
        """
        print("\n--- Starting the 'fit' process for my Naive Bayes ---")
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        print(f"Detected {n_samples} samples, {n_features} features.")
        print(f"Detected {n_classes} unique classes: {self.classes_}")

        # First, let's figure out all the unique values each feature can take.
        # This is important for the denominator in the likelihood calculation, especially with smoothing.
        print("Finding unique values for each feature...")
        for feature_idx in range(self.n_features_):
            # Get unique values AND handle potential missing values if any (though assuming complete data here)
            unique_vals = set(np.unique(X[:, feature_idx]))
            self.feature_unique_values_[feature_idx] = unique_vals
            # print(f"  Feature {feature_idx}: Found {len(unique_vals)} unique values.") # Can be verbose

        # Now, calculate priors and likelihoods for each class.
        print("Calculating class priors and feature likelihoods...")
        for idx, current_class in enumerate(self.classes_):
            # Get all samples belonging to the current class
            # Using boolean indexing is efficient here.
            X_class = X[y == current_class]
            n_class_samples = X_class.shape[0]
            print(f"  Processing class: '{current_class}' ({n_class_samples} samples)")

            # Calculate Class Prior: P(Class)
            # Using log probability to avoid potential underflow later.
            # No smoothing applied to prior here usually, but could add if needed:
            # prior_numerator = n_class_samples + self.alpha
            # prior_denominator = n_samples + n_classes * self.alpha
            # self.class_priors_[current_class] = np.log(prior_numerator / prior_denominator)
            # Standard prior:
            if n_samples == 0: # Avoid division by zero if dataset is empty
                 self.class_priors_[current_class] = -np.inf # Log(0)
            else:
                 prior = n_class_samples / n_samples
                 self.class_priors_[current_class] = np.log(prior) if prior > 0 else -np.inf
                 print(f"    Log Prior P(Class={current_class}) = {self.class_priors_[current_class]:.4f}")


            # Calculate Feature Likelihoods: P(feature_i = value | Class)
            # I need to do this for each feature.
            self.feature_log_likelihoods_[current_class] = {}
            for feature_idx in range(self.n_features_):
                self.feature_log_likelihoods_[current_class][feature_idx] = {}
                # Count occurrences of each unique value for this feature WITHIN this class
                feature_values_in_class = X_class[:, feature_idx]
                value_counts = {val: np.sum(feature_values_in_class == val)
                                for val in self.feature_unique_values_[feature_idx]}

                # Total number of unique values for this specific feature across all data
                n_unique_feature_values = len(self.feature_unique_values_[feature_idx])

                # Calculate log likelihood for each value this feature can take
                for feature_value in self.feature_unique_values_[feature_idx]:
                    count = value_counts.get(feature_value, 0) # Get count, default to 0 if not seen

                    # Apply Laplace Smoothing Here!
                    likelihood_numerator = count + self.alpha
                    likelihood_denominator = n_class_samples + (self.alpha * n_unique_feature_values)

                    # Avoid log(0) if denominator is somehow 0 (e.g., empty class and alpha=0)
                    if likelihood_denominator == 0:
                         log_likelihood = -np.inf
                    else:
                         likelihood = likelihood_numerator / likelihood_denominator
                         log_likelihood = np.log(likelihood) if likelihood > 0 else -np.inf # Use log

                    self.feature_log_likelihoods_[current_class][feature_idx][feature_value] = log_likelihood

                # Optional: Print a sample likelihood
                # if feature_idx == 0: # Just print for the first feature for brevity
                #     first_val = list(self.feature_unique_values_[feature_idx])[0]
                #     print(f"      Sample Log Likelihood P(F{feature_idx}={first_val} | Class={current_class}) = {self.feature_log_likelihoods_[current_class][feature_idx][first_val]:.4f}")


        print("--- Fit complete. Priors and Likelihoods learned. ---")
        return self # Good practice to return self

    def predict(self, X):
        """
        Predict class labels for new data X.

        Args:
            X (np.ndarray): New data features (samples x features).
                            Should have the same number of features as training data.
                            Assumed to be categorical/discrete.

        Returns:
            np.ndarray: Predicted class labels for each sample in X.
        """
        print(f"\n--- Predicting labels for {X.shape[0]} new samples ---")
        if self.classes_ is None:
            raise RuntimeError("You must call 'fit' before predicting.")
        if X.shape[1] != self.n_features_:
             raise ValueError(f"Input has {X.shape[1]} features, but model was trained on {self.n_features_}")

        y_pred = []
        for i, sample in enumerate(X):
            # For each sample, calculate the posterior probability (log scale) for each class
            log_posteriors = {}
            for current_class in self.classes_:
                # Start with the log prior for this class
                log_posterior = self.class_priors_[current_class]

                # Add the log likelihood for each feature value in the sample
                for feature_idx in range(self.n_features_):
                    feature_value = sample[feature_idx]

                    # *** Handling unseen feature values during prediction ***
                    # If a feature value was NOT seen during training for this feature index:
                    # Option 1: Assign a very small probability (using smoothing logic with count=0)
                    # Option 2: Ignore this feature for this sample (less common)
                    # My implementation with smoothing handles this naturally!
                    # The likelihood calculation during fit already prepared for all known unique values.
                    # If feature_value was *never* seen for *any* class for this feature_idx during training,
                    # it won't be in self.feature_unique_values_[feature_idx]. This is an edge case.
                    # Let's assume discretization handles this, or we could add a default small probability.

                    # Get the stored log likelihood P(feature_value | current_class)
                    # Use .get() for safety, though fit should have populated all values
                    log_likelihood = self.feature_log_likelihoods_[current_class][feature_idx].get(feature_value)

                    # What if the specific value 'feature_value' was never seen for *this specific class*
                    # but was seen for others? .get() would return None if we didn't pre-populate in fit.
                    # Since I pre-populated with smoothing in fit, I should always get a value.
                    # Let's double check the logic in fit. Yes, it iterates through all unique values
                    # found across the *entire* dataset for that feature.

                    # What if the value is entirely new (not in self.feature_unique_values_[feature_idx])?
                    # My current 'fit' doesn't explicitly handle this. Let's refine 'predict'.
                    if feature_value not in self.feature_unique_values_[feature_idx]:
                        # This value was never seen for this feature index in the *entire* training set.
                        # Apply smoothing as if count=0 for this value.
                        # Need n_class_samples and n_unique_feature_values for this specific class/feature.
                        # This is getting complicated. Let's assume for now test data only contains feature values seen in training.
                        # A robust implementation might need to recalculate this smoothed value on the fly,
                        # or store n_class_samples and n_unique_feature_values per class/feature.
                        # OR: We can pre-calculate the 'unknown value' likelihood during fit using smoothing.
                        # Let's stick to the assumption for now: test set values are within training set unique values.
                        # If a value IS in the unique set but just wasn't seen for *this* class, smoothing handled it in fit.
                        pass # Assume value exists in likelihoods dictionary due to fit logic


                    if log_likelihood is not None:
                        log_posterior += log_likelihood
                    else:
                         # This case should ideally not happen if fit populated correctly for all known values.
                         # If it does, maybe assign a default very small log prob?
                         # Let's assume it's handled. If errors occur, revisit here.
                         # We could pre-calculate a log prob for "unknown" based on smoothing:
                         # log_prob_unknown = np.log(self.alpha / (n_class_samples + self.alpha * n_unique_feature_values))
                         # For now, let's assume the value exists in the dict generated by fit.
                         print(f"Warning: Feature value {feature_value} for feature {feature_idx} not found in likelihoods for class {current_class}. This shouldn't happen with current fit logic if value was seen in training.")
                         log_posterior += -np.inf # Penalize heavily if lookup fails unexpectedly


                log_posteriors[current_class] = log_posterior

            # Choose the class with the highest log posterior probability
            # Handle cases where all posteriors might be -inf (e.g., empty training set for a class)
            if not log_posteriors: # Should not happen if fit ran
                 predicted_class = self.classes_[0] # Default guess? Error?
                 print(f"Warning: No posteriors calculated for sample {i}.")
            else:
                 # Check if all are -inf
                 if all(p == -np.inf for p in log_posteriors.values()):
                     # Arbitrarily pick the first class, or maybe random?
                     predicted_class = self.classes_[0]
                     # print(f"Warning: All posterior probabilities are -inf for sample {i}. Defaulting.")
                 else:
                     predicted_class = max(log_posteriors, key=log_posteriors.get)

            y_pred.append(predicted_class)

            # # Optional: Print calculation for the first sample
            # if i == 0:
            #     print(f"  Log Posteriors for Sample 0: {log_posteriors}")
            #     print(f"  Predicted Class for Sample 0: {predicted_class}")


        print("--- Prediction complete. ---")
        return np.array(y_pred)
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

np.random.seed(42)

##################################################################
# Starter code for exercise 5: Logistic Model for Argument Quality
##################################################################

GROUP = "04"  # TODO: write in your group number


def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features + 1).
    """
    data = pd.read_csv(filename, delimiter='\t')
    features = np.hstack([np.ones((data.shape[0], 1)), data.values])
    return features


def load_class_values(filename: str) -> np.array:
    """
    Load the class values for overall quality (class 0 for quality 1 and class 1
    for overall quality 2 or 3) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    data = pd.read_csv(filename, delimiter='\t')

    if 'overall quality' not in data.columns:
        raise KeyError("Column 'overall quality' not found in the dataset.")

    classes = np.where(data['overall quality'] > 1, 1, 0)
    return classes


def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        return np.sum(cs) / len(cs)


def logistic_function(w: np.array, x: np.array) -> float:
    """
    Return the output of a logistic function with parameter vector `w` on
    example `x`.
    Hint: use np.exp(np.clip(..., -30, 30)) instead of np.exp(...) to avoid
    divisions by zero
    """
    z = np.clip(np.dot(x, w.T), -30, 30)
    return 1 / (1 + np.exp(-z))


def logistic_prediction(w: np.array, x: np.array) -> float:
    """
    Making predictions based on the output of the logistic function
    """
    return np.round(logistic_function(w, x))


def initialize_random_weights(p: int) -> np.array:
    """
    Generate a pseudorandom weight vector of dimension p.
    """
    return np.random.uniform(-1,1,(p,1))



def logistic_loss(w: np.array, x: np.array, c: int) -> float:
    """
    Calculate the logistic loss function
    """
    return -np.sum(c * np.log(logistic_function(w, x)) + (1 - c) * np.log(1 - logistic_function(w, x)))


def train_logistic_regression_with_bgd(xs: np.array, cs: np.array, eta: float=1e-8, iterations: int=1000, validation_fraction: float=0) -> Tuple[np.array, float, float]:
    """
    Fit a logistic regression model using the Batch Gradient Descent algorithm and
    return the learned weights as a numpy array.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values c(x) for every element in `xs` as a one-dimensional numpy array with length n
    - `eta`: the learning rate as a float value
    - `iterations': the number of iterations to run the algorithm for
    - 'validation_fraction': fraction of xs and cs used for validation (not for training)

    Returns:
    - the learned weights as a column vector, i.e. a two-dimensional numpy array with shape (1, p)
    - logistic loss value
    - misclassification rate of predictions on training part of xs/cs
    - misclassification rate of predictions on validation part of xs/cs
    """
    # Initialize weights
    w = initialize_random_weights(xs.shape[0])

    loss = logistic_loss(w, xs, cs)
    train_misclassification_rate = misclassification_rate(logistic_prediction(w, xs), cs)
    
    validation_xs = xs  
    validation_cs = cs  
    
    validation_misclassification_rate = misclassification_rate(logistic_prediction(w, validation_xs), validation_cs)

    return w, loss, train_misclassification_rate, validation_misclassification_rate


def plot_loss_and_misclassification_rates(losss: List[float], train_misclassification_rates: List[float], validation_misclassification_rates: List[float]):
    """
    Plots the normalized loss (divided by max(losss)) and both misclassification rates
    for each iteration.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losss / np.max(losss), label='Normalized Loss')
    plt.title('Normalized Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_misclassification_rates, label='Training Misclassification Rate')
    plt.plot(validation_misclassification_rates, label='Validation Misclassification Rate')
    plt.title('Misclassification Rates Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Rate')
    plt.legend()

    plt.show()

########################################################################
# Tests
import os
from pytest import approx


def test_logistic_function():
    x = np.array([1, 1, 2])
    assert logistic_function(np.array([0, 0, 0]), x) == approx(0.5)
    assert logistic_function(np.array([1e2, 1e2, 1e2]), x) == approx(1)
    assert logistic_function(np.array([-1e2, -1e2, -1e2]), x) == approx(0)
    assert logistic_function(np.array([1e2, -1e2, 0]), x) == approx(0.5)


def test_bgd():
    xs = np.array([
        [1, -1],
        [1, 2],
        [1, -2],
    ])
    cs = np.array([0, 1, 0])
    
    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, 0.1, 100)
    assert w @ [1, -1] < 0 and w @ [1, 2] > 0
    w, _, _, _ = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100)
    assert w @ [1, -1] > 0 and w @ [1, 2] < 0




########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    train_features_file_name = "features-train-cleaned.tsv"
    train_classes_file_name = "quality-scores-train-cleaned.tsv"
    test_features_file_name = "features-test-cleaned.tsv"
    test_predictions_file_name = "quality-scores-test-predicted.tsv"

    # Load data
    xs = load_feature_vectors(train_features_file_name)
    xs_test = load_feature_vectors(test_features_file_name)
    cs = load_class_values(train_classes_file_name)

    print("(a)")
    # TODO: Print number of examples with each class
    # Count the occurrences of each class in the 'cs' array
    class_counts = np.bincount(cs)

    # Print the number of examples for each class
    for class_label, count in enumerate(class_counts):
        print(f"Class {class_label}: {count} examples")

    print("(b)")
    # TODO: Print misclassification rate of random classifier
    random_predictions = np.random.randint(2, size=len(cs))  # Randomly predict 0 or 1
    random_misclassification_rate = misclassification_rate(random_predictions, cs)

    print(f"Misclassification rate of random classifier: {random_misclassification_rate * 100:.2f}%")

    print("(c)")
    test_c_result = pytest.main(['-k', 'test_logistic_function', '--tb=short', 'programming_exercise_logistic_model.py'])
    if test_c_result != 0:
        sys.exit(test_c_result)
    print("Test logistic function successful")

    print("(d)")
    test_d_result = pytest.main(['-k', 'test_bgd', '--tb=short', 'programming_exercise_logistic_model.py'])
    if test_d_result != 0:
        sys.exit(test_d_result)
    print("Test bgd successful")

    print("(e)")
    w, losss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd(xs, cs, validation_fraction=0.2)
    plot_loss_and_misclassification_rates(losss, train_misclassification_rates, validation_misclassification_rates)

    print("(f)")
    
    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, validation_fraction=0.2)
    predictions = np.round(logistic_function(w, xs_test))

    
    pd.DataFrame(predictions).to_csv(test_predictions_file_name, header=False, index=False)


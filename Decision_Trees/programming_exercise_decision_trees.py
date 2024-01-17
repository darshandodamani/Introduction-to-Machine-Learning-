#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple
########################################################################
# Starter code for exercise 6: Argument quality prediction with CART decision trees
########################################################################
GROUP = '04' # 04 Your group number here

def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features, 1).
    """
    # features = pd.read_csv(filename, sep='\t', usecols=["#id", "chars_count"])
    features = pd.read_csv(filename, sep='\t')
    features = features.select_dtypes(include=np.int64) # keep only numerical values
    feature_names = features.columns
    xs = features.loc[:, feature_names].values 
    xs = xs.reshape(xs.shape[0], xs.shape[-1], 1)
    xs = np.concatenate([np.ones([xs.shape[0], 1, 1]), xs], axis=1)
    return xs

# From last exercise sheet
def load_class_values(filename: str) -> np.array:
    """
    Load the class values for overall quality (class 0 for quality 1 and class 1
    for overall quality 2 or 3) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    #return np.ravel((pd.read_csv(filename, sep='\t', usecols=["overall_quality"]).to_numpy() > 1) * 1)
    # Load the DataFrame from the file
    df = pd.read_csv(filename, sep='\t')

    # Print the DataFrame columns for debugging
    print("Columns in the loaded DataFrame:", df.columns)

    # Assuming 'overall_quality' is a column in the file
    if 'conclusion_reasons_char_count_ratio' in df.columns:
        return np.ravel((df['conclusion_reasons_char_count_ratio'].to_numpy() > 1) * 1)
    else:
        raise ValueError("Column 'conclusion_reasons_char_count_ratio' not found in the file.")




def most_common_class(cs: np.array):
    """Return the most common class value in the given array

    Arguments:
    - cs: a 1-dimensional array of length n, containing of the class values c(x) for
          every element x of a dataset D
    """
    # TODO: Your code here
    unique_classes, counts = np.unique(cs, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common_class_value = unique_classes[most_common_index]
    return most_common_class_value

def gini_impurity(cs: np.array) -> float:
    """Compute the Gini index for a set of examples represented by the list of
    class values

    Arguments:
    - cs: a 1-dimensional array of length n, containing of the class values c(x) for
          every element x of a dataset D
    """
    # TODO: Your code here
    total_samples = len(cs)
    if total_samples == 0:
        return 0.0

    # Calculate the probability of each class in the set
    class_probabilities = np.array([np.sum(cs == c) / total_samples for c in np.unique(cs)])

    # Calculate the Gini index using the formula: Gini(D) = 1 - sum(p_i^2)
    gini_index = 1.0 - np.sum(class_probabilities**2)

    return gini_index

def gini_impurity_reduction(impurity_D: float, cs_l: np.array, cs_r: np.array) -> float:
    """Compute the Gini impurity reduction of a binary split.

    Arguments:
    - impurity_D: the Gini impurity of the entire document D set to be split
    - cs_l: an array with the class values of the examples in the left split
    - cs_r: an array with the class values of the examples in the right split
    """
    size_D = len(cs_l) + len(cs_r)
    # TODO: Your code here
    size_l = len(cs_l)
    size_r = len(cs_r)
    size_D = size_l + size_r

    # Calculate the Gini impurity for the left and right splits
    impurity_l = gini_impurity(cs_l)
    impurity_r = gini_impurity(cs_r)

    # Calculate the weighted average Gini impurity after the split
    weighted_impurity = (size_l / size_D) * impurity_l + (size_r / size_D) * impurity_r

    # Calculate the Gini impurity reduction using the formula: Impurity(D) - Impurity(D|split)
    impurity_reduction = impurity_D - weighted_impurity

    return impurity_reduction

def possible_thresholds(xs: np.array, feature: int) -> np.array:
    """Compute all possible thresholds for splitting the example set xs along
    the given feature. Pick thresholds as the mid-point between all pairs of
    distinct, consecutive values in ascending order.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - feature: an integer with 0 <= a < p, giving the feature to be used for splitting xs
    """
    # Extract the feature values for the specified feature
    feature_values = xs[:, feature, 0]

    # Sort the feature values in ascending order
    sorted_values = np.sort(feature_values)

    # Calculate mid-points between consecutive values
    thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

    # Ensure that the resulting array has at most two elements
    thresholds = thresholds[:2]

    return thresholds

# def find_split_indexes(xs: np.array, feature: int, threshold: float) -> Tuple[np.array, np.array]:
#     """Split the given dataset using the provided feature and threshold.

#     Arguments:
#     - xs: an array of shape (n, p, 1)
#     - feature: an integer with 0 <= a < p, giving the feature to be used for splitting xs
#     - threshold: the threshold to be used for splitting (xs, cs) along the given feature

#     Returns:
#     - left: a 1-dimensional integer array, length <= n
#     - right: a 1-dimensional integer array, length <= n
#     """
#     # This function is provided for you.
#     smaller = (xs[:, feature, :] < threshold).flatten()
#     bigger = ~smaller  # element-wise negation

#     idx = np.arange(xs.shape[0])

#     return idx[smaller], idx[bigger]

def find_split_indexes(xs: np.array, feature: int, threshold: float) -> Tuple[np.array, np.array]:
    """Find the indices of samples that are on the left and right sides of a split.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - feature: an integer with 0 <= feature < p, giving the feature to split xs
    - threshold: the threshold to split xs along the feature

    Returns:
    - left_indexes: an array of indices for the samples on the left side of the split
    - right_indexes: an array of indices for the samples on the right side of the split
    """
    n, p, _ = xs.shape
    assert 0 <= feature < p

    left_indexes = np.where(xs[:, feature, 0] <= threshold)[0]
    right_indexes = np.where(xs[:, feature, 0] > threshold)[0]

    return left_indexes, right_indexes

def find_best_split(xs: np.array, cs: np.array) -> Tuple[int, float]:
    """
    Find the best split point for the dataset (xs, cs) from among the given
    possible feature indexes, as determined by the Gini index.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - cs: a 1-dimensional array of length n

    Returns:
    - the feature index of the best split
    - the threshold value of the best split
    """
    # hints to start
    a_best = None
    threshold_best = None
    gini_reduction_best = 0
    gini_all = gini_impurity(cs) # impurity of the example set D
    features = np.arange(xs.shape[1]) # features available for splitting
    for a_i in features:
        for threshold in possible_thresholds(xs, a_i):
            # TODO: Your code here
            # Split the dataset based on the current feature and threshold
            left_indexes, right_indexes = find_split_indexes(xs, a_i, threshold)
            
            # Extract class values for the left and right splits
            cs_left = cs[left_indexes]
            cs_right = cs[right_indexes]

            # Calculate Gini impurity reduction for the current split
            gini_reduction = gini_impurity_reduction(gini_all, cs_left, cs_right)

            # Update the best split if the current split has higher impurity reduction
            if gini_reduction > gini_reduction_best:
                gini_reduction_best = gini_reduction
                a_best = a_i
                threshold_best = threshold

    return a_best, threshold_best

def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    # if len(cs) == 0:
    #     return float('nan')
    # else:
    #     return 1 - (np.sum(np.equal(cs, ys)) / len(cs))
    if len(cs) == 0:
        return float('nan')
    else:
        hits = [cs[i][ys[i]] for i in range(len(ys))]
        return 1 - (sum(hits) / len(ys))

class CARTNode:
    """Class representing a node in a CART decision tree.
    """
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def set_split(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def set_label(self, label):
        self.label = label

    def classify(self, x):
        if self.feature is None:
            return self.label

        v = x[self.feature]

        if v < self.threshold:
            return self.left.classify(x)
        else:
            return self.right.classify(x)


class CARTModel:
    """Trivial model interface class for the CART decision tree.
    """
    def __init__(self, max_depth=None):
        self._t = None  # root of the decision tree
        self._max_depth = max_depth

    def fit(self, xs: np.array, cs: np.array):
        self._t = id3_cart(xs, cs, max_depth=self._max_depth)

    def predict(self, x):
        return self._t.classify(x)


def id3_cart(xs: np.array, cs: np.array, max_depth: Optional[int] = 10) -> CARTNode:
    """Construct a CART decision tree with the modified ID3 algorithm.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - cs: a 1-dimensional array of length n
    - max_depth: limit to the size of the tree that is constructed; unlimited if None

    Returns:
    - the root node of the constructed decision tree
    """
    if max_depth is not None and (max_depth == 0 or len(np.unique(cs)) == 1):
        # Create a leaf node if max_depth is reached or all examples have the same class
        leaf = CARTNode()
        leaf.set_label(most_common_class(cs))
        return leaf

    # Find the best split
    best_feature, best_threshold = find_best_split(xs, cs)

    if best_feature is None:
        # No suitable split found, create a leaf node
        leaf = CARTNode()
        leaf.set_label(most_common_class(cs))
        return leaf

    # Split the dataset
    left_indexes, right_indexes = find_split_indexes(xs, best_feature, best_threshold)

    # Recursive call to build left and right subtrees
    left_subtree = id3_cart(xs[left_indexes], cs[left_indexes], None if max_depth is None else max_depth - 1)
    right_subtree = id3_cart(xs[right_indexes], cs[right_indexes], None if max_depth is None else max_depth - 1)

    # Create an internal node with the best split
    node = CARTNode()
    node.set_split(best_feature, best_threshold, left_subtree, right_subtree)

    return node



# Replace these with the actual file names
training_classes_file_name = "quality-scores-train-cleaned.tsv"
training_features_file_name = "features-train-cleaned.tsv"
test_features_file_name = "features-test-cleaned.tsv"


def train_and_predict(training_features_file_name: str,
                      training_classes_file_name: str,
                      test_features_file_name: str) -> np.array:
    """Train a model on the given training dataset, and predict the class values
    for the given testing dataset.

    Return an array with the predicted class values, in the same order as the
    examples in the testing dataset.
    """
    # Load training data
    train_xs = load_feature_vectors(training_features_file_name)
    
    # Load class values from the correct file
    train_cs = load_class_values(training_classes_file_name)

    # Load test data
    test_xs = load_feature_vectors(test_features_file_name)

    # Initialize and train the model
    model = CARTModel(max_depth=None)  
    model.fit(train_xs, train_cs)

    # Predict on the training set
    train_predictions = np.array([model.predict(x) for x in train_xs])

    # Calculate and print misclassification rate on the training set
    train_misclassification_rate = misclassification_rate(train_cs, train_predictions)
    print(f"Misclassification Rate on Training Set: {train_misclassification_rate:.2%}")

    # Predict on the test set
    test_predictions = np.array([model.predict(x) for x in test_xs])

    return test_predictions

########################################################################
# Tests
import os
from pytest import approx

def test_most_common_class():
    cs = np.array(['red', 'green', 'green', 'blue', 'green'])
    assert most_common_class(cs) == 'green', \
        "Identify the correct most common class"

def test_gini_impurity():
    # should work with two classes
    cs = np.array(['a', 'a', 'b', 'a'])
    assert gini_impurity(cs) == approx(2*0.75*0.25), \
        "Compute the correct Gini index for a two-class dataset"

    # should also work with more classes
    cs = np.array(['a', 'b', 'c', 'b', 'a'])
    assert gini_impurity(cs) == approx(1 - (0.4**2 + 0.4**2 + 0.2**2)), \
        "Compute the correct Gini index for a three-class dataset"

def test_gini_impurity_reduction():
    # cs = np.array(['a', 'a', 'b', 'a'])
    i_D = 0.375

    assert gini_impurity_reduction(i_D, np.array(['a', 'a']), np.array(['b', 'a'])) == approx(0.125), \
        "Compute the correct gini reduction for the first test split"

    assert gini_impurity_reduction(i_D, np.array(['a', 'a', 'a']), np.array(['b'])) == approx(0.375), \
        "Compute the correct gini reduction for the second test split"


def test_possible_thresholds():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    # first feature allows two possible split points
    thresholds = possible_thresholds(xs, 0)
    print(thresholds)  # Add this line to print the actual output
    assert thresholds == approx(np.array([0.25, 0.75])), \
        "Find all possible thresholds for the first feature."

def test_find_split_indexes():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    l, r = find_split_indexes(xs, 0, 0.75)
    assert all(l == np.array([1, 2])) and all(r == np.array([0, 3]))

    l, r = find_split_indexes(xs, 0, 0.25)
    assert all(l == np.array([2])) and all(r == np.array([0, 1, 3]))

def test_find_best_split():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    cs = np.array(['a', 'a', 'c', 'a'])
    a, t = find_best_split(xs, cs)
    assert a == 0, "Choose the best feature."
    assert t == 0.25, "Choose the best threshold."

def test_cart_model():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    cs = np.array(['a', 'a', 'b', 'b'])
    tree = CARTModel()
    tree.fit(xs, cs)
    preds = [tree.predict(x) for x in xs]

    assert all(cs == preds), \
        "On a dataset without label noise, reach zero training error."


if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    train_features_file_name = sys.argv[1]
    train_classes_file_name = sys.argv[2]
    test_features_file_name = sys.argv[3]

    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict function...")
    predictions = train_and_predict(train_features_file_name, train_classes_file_name, test_features_file_name)
    print("Writing predictions to file...")
    pd.DataFrame(predictions).to_csv(f"argument-clarity-predictions-mlp-group-{GROUP}.tsv", header=False, index=False)

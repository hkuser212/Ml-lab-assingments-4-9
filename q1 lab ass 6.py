import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris['data']
y = iris['target']
class_labels = iris['target_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def calculate_mean_variance(X, y):
    classes = np.unique(y)
    means = {}
    variances = {}

    for cls in classes:
        X_cls = X[y == cls]
        means[cls] = np.mean(X_cls, axis=0)
        variances[cls] = np.var(X_cls, axis=0)

    return means, variances


means, variances = calculate_mean_variance(X_train, y_train)


def gaussian_probability(x, mean, var):
    eps = 1e-6  # To avoid division by zero
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponent = np.exp(- ((x - mean) ** 2) / (2 * var + eps))
    return coeff * exponent


def calculate_class_probabilities(X, means, variances):
    class_probs = {}
    for cls in means:
        class_probs[cls] = np.sum(np.log(gaussian_probability(X, means[cls], variances[cls])), axis=1)

    return class_probs


def predict(X, means, variances):
    class_probs = calculate_class_probabilities(X, means, variances)
    predictions = []

    for i in range(X.shape[0]):
        class_prob = {cls: class_probs[cls][i] for cls in class_probs}
        predictions.append(max(class_prob, key=class_prob.get))

    return np.array(predictions)


y_train_pred = predict(X_train, means, variances)
y_test_pred = predict(X_test, means, variances)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('Classification Report (Test):')
print(classification_report(y_test, y_test_pred, target_names=class_labels))
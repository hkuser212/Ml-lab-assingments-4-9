import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 31)}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print(f'Best K value: {grid_search.best_params_["n_neighbors"]}')
print(f'Best cross-validated accuracy: {grid_search.best_score_:.4f}')

best_knn = grid_search.best_estimator_
y_test_pred = best_knn.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test set accuracy with best K: {test_accuracy:.4f}')
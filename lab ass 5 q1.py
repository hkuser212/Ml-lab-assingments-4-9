import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
iris = datasets.load_iris()

# Create a DataFrame for easier handling of the data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Iris Dataset")
plt.show()

X=iris.data
y=iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(multi_class="ovr",max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

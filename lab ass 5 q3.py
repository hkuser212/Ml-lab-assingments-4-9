import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Correct way to load the dataset (no header in the file)
df = pd.read_csv("BankNote_Authentication.csv")

# Assign column names manually (if there are no headers)
df.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

# Check the first few rows of the dataset to ensure it's loaded correctly
print(df.head())

# Ensure that the target variable (Class) is of integer type (0 or 1)
df['Class'] = df['Class'].astype(int)

# Split data into input features (X) and the target variable (y)
X = df[['Variance', 'Skewness', 'Curtosis', 'Entropy']]
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the Decision Tree Classifier (CART algorithm using Gini index)
cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)

# Train the model
cart_model.fit(X_train, y_train)

# Predict the test set results
y_pred = cart_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(cart_model, feature_names=['Variance', 'Skewness', 'Curtosis', 'Entropy'],
               class_names=['Fake', 'Authentic'], filled=True, rounded=True)
plt.show()

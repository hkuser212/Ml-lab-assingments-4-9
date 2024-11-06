import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('weather.csv')

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df.drop('Play', axis=1))  # Drop the target column for encoding
y = df['Play']  # Target variable remains as it is


X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.3, random_state=42)


dt_model = DecisionTreeClassifier()


dt_model.fit(X_train, y_train)


y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
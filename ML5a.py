import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
df = pd.read_csv(r"C:\Users\namiy\Downloads\train_u6lujuX_CVtuZ9i (1).csv")

print(df.head())
print(df['Loan_Status'].value_counts())

# Handling Missing Values
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Education'].fillna(df['Education'].mode()[0], inplace=True)
df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)

# Features and Target
X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Property_Area']]
y = df['Loan_Status']

# Convert Categorical to Numeric
X = pd.get_dummies(X, columns=['Education', 'Property_Area'], drop_first=True)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
deep_tree = DecisionTreeClassifier(random_state=42)
deep_tree.fit(X_train, y_train)

shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
shallow_tree.fit(X_train, y_train)

# Predictions
y_pred_deep = deep_tree.predict(X_test)
y_pred_shallow = shallow_tree.predict(X_test)

# Evaluation
print("Deep Tree Accuracy:", accuracy_score(y_test, y_pred_deep))
print("Shallow Tree Accuracy:", accuracy_score(y_test, y_pred_shallow))

print("\nDeep Tree Report:\n", classification_report(y_test, y_pred_deep))
print("\nShallow Tree Report:\n", classification_report(y_test, y_pred_shallow))

# Overfitting Check
train_acc_deep = accuracy_score(y_train, deep_tree.predict(X_train))
test_acc_deep = accuracy_score(y_test, y_pred_deep)

print("Deep Tree Train Accuracy:", train_acc_deep)
print("Deep Tree Test Accuracy:", test_acc_deep)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_deep)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Deep Tree)")
plt.show()

# Tree Structure
plt.figure(figsize=(12,8))
plot_tree(shallow_tree, feature_names=X.columns,
          class_names=encoder.classes_, filled=True)
plt.title("Decision Tree Structure (Shallow Tree)")
plt.show()

# Feature Importance
importances = deep_tree.feature_importances_
indices = np.argsort(importances)

plt.figure()
plt.barh(X.columns[indices], importances[indices])
plt.title("Feature Importance")
plt.show()

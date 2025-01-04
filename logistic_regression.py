# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load a sample dataset
# We'll use the Iris dataset for simplicity
from sklearn.datasets import load_iris
iris = load_iris()

# Extracting features and labels
X = iris.data  # Features
y = (iris.target == 2).astype(int)  # We classify whether the flower is of class 2 (e.g., Virginica)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Set a high max_iter for convergence
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Optional: Show coefficients of the model
print("\nModel Coefficients:\n", model.coef_)
print("Model Intercept:", model.intercept_)

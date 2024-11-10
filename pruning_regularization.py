# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:09:37 2024

@author: Iaina
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
FILE_PATH = r"C:\ALY6040_DATAMINING\Assignment2\mushrooms.xlsx"
data = pd.read_excel(FILE_PATH)

# Preprocess data
data = pd.get_dummies(data, drop_first=True)  # One-hot encode categorical variables
X = data.drop('class_p', axis=1)  # Features
y = data['class_p']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initial Decision Tree without pruning
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Initial Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

# Cost Complexity Pruning (finding optimal ccp_alpha)
path = clf.cost_complexity_pruning_path(X_train, y_train)  # Get ccp_alpha values
ccp_alphas = path.ccp_alphas  # Extract alpha values

# Train a decision tree for each alpha and store accuracy
train_acc = []
test_acc = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))

# Plot training and testing accuracy vs. alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_acc, label="Training Accuracy", marker='o')
plt.plot(ccp_alphas, test_acc, label="Testing Accuracy", marker='o')
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Alpha for Pruned Decision Trees")
plt.legend()
plt.show()

# Choosing the best alpha based on highest test accuracy
optimal_alpha = ccp_alphas[test_acc.index(max(test_acc))]
print("Optimal alpha:", optimal_alpha)

# Train final pruned tree with optimal alpha
pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
pruned_clf.fit(X_train, y_train)
y_pruned_pred = pruned_clf.predict(X_test)

# Evaluate pruned tree
print("\nPruned Decision Tree Accuracy:", accuracy_score(y_test, y_pruned_pred))
print("Pruned Decision Tree Classification Report:\n", classification_report(y_test, y_pruned_pred))

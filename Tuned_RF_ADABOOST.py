import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# File path (adjust this path if necessary)
FILE_PATH = r"C:\ALY6040_DATAMINING\Assignment2\mushrooms.xlsx"

# Step 1: Load the data
def load_data(filepath):
    """Load the mushroom dataset from the specified Excel file path."""
    data = pd.read_excel(filepath)
    print("Data loaded successfully.")
    return data

# Step 2: Preprocess the data
def preprocess_data(data):
    """Preprocess data by applying one-hot encoding to categorical variables."""
    data_encoded = pd.get_dummies(data)
    print("Data preprocessing completed (one-hot encoding applied).")
    return data_encoded

# Step 3: Split the data into features and target
def split_data(data):
    """Split the data into training and testing sets."""
    X = data.drop(['class_e', 'class_p'], axis=1)  # Features
    y = data['class_p']  # Target variable: class_p (1 if poisonous, 0 if edible)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Step 4: Tune the Decision Tree using cross-validation with GridSearchCV
def tune_decision_tree(X_train, y_train):
    """Tune Decision Tree hyperparameters using GridSearchCV with cross-validation."""
    param_grid = {
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    print("Best Decision Tree Parameters:", grid_search.best_params_)
    return best_dt

# Step 5: Train ensemble models using the best Decision Tree parameters
def train_ensemble_models(best_dt, X_train, y_train):
    """Train Random Forest and AdaBoost classifiers with tuned parameters."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=best_dt.get_params()['max_depth'], random_state=42)
    ada = AdaBoostClassifier(estimator=best_dt, n_estimators=50, random_state=42)

    rf.fit(X_train, y_train)
    ada.fit(X_train, y_train)
    
    print("Ensemble models training completed.")
    return rf, ada

# Step 6: Evaluate the models
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate the model on test data and print the accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nEvaluating {model_name}:")
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

# Step 7: Visualize the Decision Tree
def visualize_tree(model, feature_names):
    """Plot the Decision Tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Edible", "Poisonous"], filled=True)
    plt.show()

# Main function to execute all steps
def main():
    # Load and preprocess data
    data = load_data(FILE_PATH)
    data_encoded = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_encoded)

    # Tune Decision Tree with cross-validation
    best_dt = tune_decision_tree(X_train, y_train)
    
    # Train ensemble models with the best Decision Tree parameters
    rf, ada = train_ensemble_models(best_dt, X_train, y_train)

    # Evaluate all models
    evaluate_model(best_dt, X_test, y_test, "Tuned Decision Tree")
    evaluate_model(rf, X_test, y_test, "Random Forest")
    evaluate_model(ada, X_test, y_test, "AdaBoost")

    # Visualize the best Decision Tree
    visualize_tree(best_dt, feature_names=X_train.columns)

# Run the main function
if __name__ == "__main__":
    main()


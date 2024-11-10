# Mushroom Classification Models

This repository contains a series of machine learning models developed to classify mushrooms as edible or poisonous based on their attributes. The project explores various model optimization techniques, including hyperparameter tuning, ensemble methods (Random Forest, AdaBoost), and regularization techniques, to enhance the accuracy and generalizability of the model.

## Project Overview

The mushroom dataset used in this project is relatively simple, with clear distinctions between edible and poisonous mushrooms based on certain attributes. Initial models achieved near-perfect accuracy, which indicated the straightforward nature of the dataset. However, the project follows a structured approach to demonstrate key concepts from machine learning, including regularization, pruning, and ensemble methods.

### Files in This Repository

- **OriginalModel.py**: Contains the initial decision tree model implementation.
- **Tuned_RF_ADABOOST.py**: Applies hyperparameter tuning and ensemble methods (Random Forest and AdaBoost) for improved accuracy.
- **pruning_regularization.py**: Implements pruning and regularization techniques to avoid overfitting and enhance model generalization.
- **mushrooms.xlsx**: The dataset used for model training and evaluation.
- **Decision Tree Mushrooms.txt**: Text description of the decision tree structure used in the model.
- **ADecision-TheoreticGeneralizationofOn....pdf**: Reference material related to decision theory.

### Project Details

1. **Dataset**: The mushroom dataset contains attributes that classify mushrooms as either edible or poisonous. The data is structured and relatively simple, making it ideal for demonstrating classification techniques.
2. **Initial Model**: A decision tree model was first implemented, achieving high accuracy on the dataset.
3. **Ensemble Methods**: Random Forest and AdaBoost were applied to improve model robustness. These methods reduce variance and increase model performance by aggregating multiple weak learners.
4. **Pruning and Regularization**: Pruning was performed to prevent overfitting and enhance model interpretability. Regularization parameters (`min_samples_leaf`, `min_samples_split`, etc.) were also explored to further optimize the model.

### Usage

To run the models:
1. Clone the repository:
   ```bash
   git clone https://github.com/IainAmosMelchizedek/Mushroom_Classification_Models.git

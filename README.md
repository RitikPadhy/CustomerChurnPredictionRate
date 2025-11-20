# Customer Churn Rate Prediction

## Overview
This project predicts whether a customer will leave a service. Using historical bank customer data, the model calculates customer churn rates and identifies the likelihood of customer exit.

---

## Part 1: Data Preprocessing
- **Dataset**: Import the dataset using pandas and explore using `head()`, `shape()`, `info()`, and `describe()`.
- **Data Exploration**: Identify categorical (`object`) and numerical (`int64`, `float64`) columns.
- **Handling Missing Data**: Check for null values and handle them appropriately.
- **Categorical Encoding**: Encode categorical columns using `pd.get_dummies()` with `drop_first=True`.
- **Visualization**: Plot the number of customers who exited and visualize correlation with heatmaps.
- **Train-Test Split**: Use `train_test_split` with 80% training and 20% testing data.
- **Feature Scaling**: Standardize features using `StandardScaler`.

---

## Part 2: Building the Model
Different machine learning models were tested:

1. **Logistic Regression**  
   - Predicts probability of customer exit based on input features.  
   - Evaluated using accuracy, F1-score, precision, recall, and confusion matrix.  

2. **Random Forest Classifier**  
   - Ensemble model building decision trees on random samples.  
   - Provides higher accuracy compared to Logistic Regression.  

3. **XGBoost Classifier**  
   - Gradient-boosted decision tree implementation.  
   - Outperformed both Logistic Regression and Random Forest.  
   - Selected as the final model due to highest accuracy.

---

## Part 3: Hyperparameter Tuning
- **Goal**: Improve XGBoostClassifier performance by tuning hyperparameters.  
- **Hyperparameters Tuned**: `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `colsample_bytree`.  
- **Method**: `RandomizedSearchCV` with 5 iterations, 5-fold cross-validation, and `n_jobs=-1`.  
- **Outcome**: Best parameters obtained from `randomized_search.best_estimator_` and `randomized_search.best_params_`.

---

## Part 4: Final Model
- Train the XGBoostClassifier using best hyperparameters.  
- Evaluate model using confusion matrix and cross-validation scores.  

---

## Part 5: Predicting a Single Observation
- Example observation with 10 features.  
- Standardize using `sc.transform()` based on training data.  
- Use the final model to predict whether the customer will exit.

---

## Usage
1. Load dataset in a Python environment (e.g., Jupyter Notebook or Google Colab).  
2. Perform data preprocessing, encoding, and scaling.  
3. Train and evaluate Logistic Regression, Random Forest, and XGBoost models.  
4. Perform hyperparameter tuning for XGBoost to optimize performance.  
5. Use the final model to predict customer churn for single or multiple observations.

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

MODELS = {
    "classification": [
        LogisticRegression(),
        RandomForestClassifier(),
        SVC(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        MLPClassifier()
    ],
    "regression": [
        Ridge(),
        Lasso(),
        RandomForestRegressor(),
        SVR(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        MLPRegressor()
    ]
}


def select_model(task, X_preprocessed, y):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42)

    # Initialize variables to store the best model and its performance
    best_model = None
    best_performance = np.inf

    # Iterate through the candidate models for the given task
    for model in MODELS[task]:
        # Train the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate the model's performance
        if task == "classification":
            performance = 1 - accuracy_score(y_val, y_pred)
        else:  # task == "regression"
            performance = mean_squared_error(y_val, y_pred)

        # Update the best model and its performance if necessary
        if performance < best_performance:
            best_model = model
            best_performance = performance

    return best_model

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

HYPERPARAMETER_DISTRIBUTIONS = {
    'LogisticRegression': {
        'C': uniform(0.001, 10),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'RandomForestClassifier': {
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'SVC': {
        'C': uniform(0.1, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': randint(2, 5),
        'shrinking': [True, False],
        'probability': [True, False]
    },
    'KNeighborsClassifier': {
        'n_neighbors': randint(1, 20),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'DecisionTreeClassifier': {
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(randint(10, 100).rvs(),) for _ in range(10)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': uniform(0.0001, 0.1)
    },
    'Ridge': {
        'alpha': uniform(0.1, 10),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'Lasso': {
        'alpha': uniform(0.1, 10)
    },
    'RandomForestRegressor': {
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'SVR': {
        'C': uniform(0.1, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': randint(2, 5),
        'shrinking': [True, False]
    },
    'KNeighborsRegressor': {
        'n_neighbors': randint(1, 20),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'DecisionTreeRegressor': {
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'MLPRegressor': {
        'hidden_layer_sizes': [(randint(10, 100).rvs(),) for _ in range(10)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': uniform(0.0001, 0.1),
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
}


def tune_hyperparameters(model, X_preprocessed, y):
    # Obtain the hyperparameter distribution for the given model
    model_class_name = model.__class__.__name__
    param_distribution = HYPERPARAMETER_DISTRIBUTIONS.get(model_class_name)

    if param_distribution:
        # Perform randomized search with cross-validation
        search = RandomizedSearchCV(
            model, param_distribution, n_iter=50, cv=5, n_jobs=-1, random_state=42)
        search.fit(X_preprocessed, y)

        # Update the model with optimized hyperparameters
        model = search.best_estimator_
    else:
        # Train the model with default hyperparameters if no distribution is available
        model.fit(X_preprocessed, y)

    return model

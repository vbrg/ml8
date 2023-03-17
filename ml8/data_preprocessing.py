import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(X, categorical_features):
    # Impute missing values
    X = X.fillna(X.mean(numeric_only=True))
    X[categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])

    # Identify numerical features
    numerical_features = X.columns.difference(categorical_features)

    # Define column transformers for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed

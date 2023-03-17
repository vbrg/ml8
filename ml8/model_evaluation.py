import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

def evaluate_model(task, model, X_preprocessed, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate and print the model's performance metrics
    if task == "classification":
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    else:  # task == "regression"
        print("Mean Squared Error:")
        print(mean_squared_error(y_test, y_pred))

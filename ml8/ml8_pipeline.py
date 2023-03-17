import pandas as pd
from data_preprocessing import preprocess_data
from model_selection import select_model
from hyperparameter_tuning import tune_hyperparameters
from model_evaluation import evaluate_model


def run_automl(dataset, target_column, task):
    # Read the dataset
    df = pd.read_csv(dataset)

    # Preprocess the data
    X_preprocessed, y = preprocess_data(df, target_column)

    # Select the best model
    model = select_model(task, X_preprocessed, y)

    # Tune the model's hyperparameters
    model = tune_hyperparameters(model, X_preprocessed, y)

    # Evaluate the model
    evaluate_model(task, model, X_preprocessed, y)

    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python automl_pipeline.py dataset.csv target_column task")
        sys.exit(1)

    dataset = sys.argv[1]
    target_column = sys.argv[2]
    task = sys.argv[3]

    run_automl(dataset, target_column, task)

# ML8: An Automated Machine Learning (AutoML) Platform

ML8 is an open-source AutoML platform designed to simplify the process of building and deploying machine learning models. By automating data preprocessing, model selection, hyperparameter tuning, and model evaluation, ML8 enables users to quickly develop high-quality models tailored to their specific needs.

## Features

- Support for a variety of machine learning tasks, including classification, regression, and clustering
- Automated data preprocessing, including missing value imputation, feature scaling, and encoding categorical variables
- Model selection from a wide range of algorithms, including those from Scikit-learn, TensorFlow, and PyTorch
- Hyperparameter tuning using techniques like grid search, random search, and Bayesian optimization
- Model evaluation using relevant metrics (e.g., accuracy, F1 score, mean squared error) and selection of the best model for the given task
- User-friendly interface or API for easy configuration and usage

## Installation

You can install ML8 using pip: `pip install git+https://github.com/vbrg/ml8.git`


## Usage

After installing ML8, you can use it in your Python projects:

```python
from ml8 import AutoML

# Load your dataset
# ...

# Initialize the AutoML platform
automl = AutoML(task="classification")

# Train and evaluate models
best_model = automl.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)
````

For more detailed examples, check the examples/ directory in the repository.

## Contributing
We welcome contributions to the ML8 project! If you'd like to contribute, please read our contributing guidelines and submit pull requests or report issues on GitHub.

## License
ML8 is released under the [MIT License](https://chat.openai.com/chat/LICENSE).
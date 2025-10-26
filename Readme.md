# PROJECT 5.1 - HOUSE PRICE PREDICTION
This Project contains the code, data, and analysis developed to predict the house price based on some input information. Our goal is study about `How to build the Linear Regression and apply Ridge/ Lasso Regularization`.


## Overview
In this project, we will build a Text Classification program that involves classifying an abstract of a publication (scientific article) into different topics.
The program will be built on the Python platform and use popular machine learning libraries such as scikit-learn, numpy, etc.
Accordingly, the general Input/Output of the program will include:
* **Input**: An abstract of the house information in market survey.
* **Output**: House price in some area with its information.

* **Data Exploration & Preprocessing**:


* **Model Selection & Training**:
Implementing `LRRegularizationVectorized` that are fine-tuned over key hyperparameters. This is self-built model

* **Model Evaluation & Forecast Generation**:
Evaluating performance using metrics such as score, mse and cost. The final model produces reliable forecasts.

## Repository Structure

```
.
├── LICENSE (not applicable)
├── README.md
├── datasets
│   ├── train-house-prices-advanced-regression-techniques.csv](dataset/train-house-prices-advanced-regression-techniques.csv
│   ├── train.csv (after preprocessing)
│   └── test.csv (after preprocessing)
├── notebooks
│   ├── 01_data_eda.ipynb
│   └── 02_model_train.ipynb
├── requirements.txt
└── utils
    ├── data_explorer.py
    └── model_built.py

```

## Installation
1. Clone the Repository:

```
git clone https://github.com/isenginer/AIVN_Project5.1_EnsembleLearning.git
```

2. Install Dependencies (Optional):

This section is for the library reference only. If your IDE has sufficient library, this section is not required to apply.
```
pip install -r requirements.txt
```

## Usage
### Interactive Analysis
Open the Jupyter notebooks in the notebooks directory to explore the data, visualize key insights, and review the model development process:

```

```

### Running the Scripts
Use the following command to run:

```
python main.py
```

### Methodology
Our approach can be summarized as follows:
* **Data Preprocessing**:
We standardize features and create new ones to capture temporal patterns (e.g., sine/cosine transformations of the hour) and aggregate multi-site weather data.

* **Modeling**:
The model algorithm is employed due to its ability to model non-linear relationships and its built-in regularization. Model hyperparameters (number of trees, tree depth, learning rate, etc.) are fine-tuned using a time-series aware cross-validation strategy.

* **Validation**:
Model performance is evaluated using metrics such as MAE, MSE, and MAPE, ensuring the forecasts are both accurate and robust.

## License
The project is public and no license required.
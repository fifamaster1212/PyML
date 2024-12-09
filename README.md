
# PyML | Domain-Specific Language for Machine Learning

PyML is a domain-specific language (DSL) designed to simplify common machine learning workflows. Built on top of Python, pandas, scikit-learn, and matplotlib, PyML aims to provide data scientists and analysts with an intuitive language for tasks such as loading data, preprocessing, training models, evaluating performance, and visualizing results. We use a [sample dataset of housing prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) from Kaggle in the examples below.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Language Grammar](#language-grammar)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Feature Engineering](#feature-engineering)
  - [Plotting](#plotting)
  - [Column Descriptions and Types](#column-descriptions-and-types)
  - [Examples for Functions with Parameters](#examples-for-functions-with-parameters)
- [Testing Suite](#testing-suite)
- [Potential Future Additions](#potential-future-additions)

## Introduction

Machine learning pipelines often require a series of repetitive steps: loading datasets, handling missing values, encoding categorical features, splitting data into training and testing sets, selecting models, training, evaluating performance, and generating predictions. PyML was created to streamline these processes through a concise, human-readable DSL.

PyML leverages Python’s scientific stack under the hood, enabling you to harness the power of `pandas`, `numpy`, `scikit-learn`, and `matplotlib` with straightforward commands. This allows both beginners and experienced practitioners to quickly set up and run ML experiments without the boilerplate of traditional Python scripts.

## Features

- **Data Loading**: Easily load CSV datasets into memory.
- **Preprocessing**: Handle missing values, filter data by conditions, and normalize numeric columns.
- **Feature Selection**: Define target and features directly from the DSL.
- **Model Creation**: Build linear models (Ridge, Lasso) and Logistic Regression models with simple commands.
- **Training and Evaluation**: Train models and evaluate performance using MSE and R² scores.
- **Visualization**: Plot data relationships directly for quick insights.
- **Describing Columns**: Inspect column statistics and types for better data understanding.
- **Prediction**: Generate predictions and save them to CSV files.

## Language Grammar

PyML’s grammar is command-driven. Each line represents one operation. Most commands follow the pattern:

```plaintext
load "filename.csv"                 # Load a dataset
target "ColumnName"                 # Set target column
features ["col1", "col2", ...]      # Set feature columns
missing fill 0                      # Fill missing values with 0
split ratio=0.8 shuffle             # Split data into train/test sets
model ridge alpha=1.0               # Create a ridge regression model
train                               # Train the current model
r2 train                            # Calculate R² on train set
r2 test                             # Calculate R² on test set
describe "ColumnName"               # Describe a single column
describe ["col1", "col2"]           # Describe multiple columns
type "ColumnName"                   # Show the data type of a single column
type ["col1", "col2"]               # Show data types of multiple columns
normalize ["col1", "col2"] method=zscore   # Normalize specified columns
plot x="colX" y="colY"              # Plot one column against another
predict save="predictions.csv"      # Predict on test set and save to CSV
feature "NewFeature" = "col1 + col2"  # Add a derived feature
```

## Installation

**Prerequisites**:

- Python 3.x
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

**Installation Steps**:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/fifamaster1212/PyML.git
   ```

2. **Install Dependencies**:

   ```bash
   cd pyml
   pip install pandas numpy scikit-learn matplotlib
   ```

## Usage

You can run PyML commands interactively using the provided REPL (`main()` function) or by passing a script file containing PyML commands line-by-line.

**Interactive REPL**:
```bash
python3 PyMLinterpreter.py
```

When the REPL starts, type commands one at a time. For example:

```plaintext
>>> load "data.csv"
>>> target "SalePrice"
>>> features ["LotArea", "OverallQual", "YearBuilt"]
>>> split ratio=0.8 shuffle
>>> model ridge alpha=1.0
>>> train
>>> r2 test
>>> predict save="predictions.csv"
>>> exit
```

## Examples

### Data Loading and Preprocessing

```plaintext
load "housing_data.csv"
missing fill 0
filter_data "LotArea > 5000 and OverallQual >= 7"
normalize ["LotArea", "GrLivArea"] method=minmax
describe ["LotArea", "GrLivArea"]
type ["LotArea", "OverallQual"]
```

**Explanation**:
- Loads a housing dataset.
- Fills missing values with 0.
- Filters rows based on given conditions.
- Normalizes numeric features.
- Describes columns and checks their data types.

### Model Training and Evaluation

```plaintext
load "housing_data.csv"
target "SalePrice"
features ["LotArea", "OverallQual", "YearBuilt", "GrLivArea"]
split ratio=0.8 shuffle
model ridge alpha=1.0
train
r2 train
r2 test
error
predict save="predictions.csv"
```

**Explanation**:
- Sets the target and features.
- Splits the data into training and testing sets.
- Creates a Ridge model, trains it, and evaluates it using R² and MSE.
- Makes predictions on the test set and saves them.

### Feature Engineering

```plaintext
load "housing_data.csv"
feature "TotalArea" = "LotArea + GrLivArea + TotalBsmtSF"
describe "TotalArea"
```

**Explanation**:
- Adds a new feature (`TotalArea`) calculated from existing columns.
- Describes the new feature to confirm it was created correctly.

### Plotting

```plaintext
load "housing_data.csv"
plot x="OverallQual" y="SalePrice"
```

**Explanation**:
- Loads the data and plots the relationship between overall quality and sale price.

### Column Descriptions and Types

```plaintext
load "housing_data.csv"
describe ["LotArea", "SalePrice"]
type ["OverallQual", "YearBuilt"]
```

**Explanation**:
- Provides statistical descriptions for specified columns.
- Prints the data types of the specified columns.

### Examples for Functions with Parameters

#### Handling Missing Values

```plaintext
missing fill 0
missing drop
missing mean
```

**Explanation**:
- `missing fill 0`: Replaces all missing values with `0`.
- `missing drop`: Removes rows with missing values.
- `missing mean`: Replaces missing values with the mean of the respective column.

#### Splitting Data

```plaintext
split ratio=0.8 shuffle
split ratio=0.7 noshuffle
```

**Explanation**:
- `split ratio=0.8 shuffle`: Splits the data into 80% training and 20% testing with shuffling.
- `split ratio=0.7 noshuffle`: Splits the data into 70% training and 30% testing without shuffling.

#### Normalizing Data

```plaintext
normalize ["LotArea", "GrLivArea"] method=minmax
normalize ["TotalBsmtSF"] method=zscore
```

**Explanation**:
- `normalize ["LotArea", "GrLivArea"] method=minmax`: Scales specified columns to a 0–1 range.
- `normalize ["TotalBsmtSF"] method=zscore`: Scales the specified column using z-score normalization.

---

## Testing Suite

PyML includes a `unittest`-based testing suite that verifies functionality for loading data, setting targets, selecting features, training models, evaluating performance, and more.

**Running Tests**:

```bash
python3 -m unittest discover -s tests
```

If all tests pass, you can be confident that PyML’s core functionality is working as intended.

## Potential Future Additions

**Extended Model Support**:
- Add more models (e.g., decision trees, random forests, gradient boosting).

**Enhanced Visualization**:
- Include histograms, scatter plots with regression lines, and correlation heatmaps.

**Advanced Preprocessing**:
- Include one-hot encoding, label encoding, and scaling options beyond min-max and z-score.

**Time Series and Deep Learning**:
- Add support for time-series analyses and incorporate deep learning frameworks for more advanced tasks.

**Improved Error Handling and Logging**:
- Provide more descriptive error messages and logging for easier debugging.

---

PyML aims to simplify common ML tasks so you can focus on analysis rather than boilerplate code. With its DSL-like syntax and integration with popular Python libraries, PyML makes it easy to go from dataset to model in just a few lines of code.

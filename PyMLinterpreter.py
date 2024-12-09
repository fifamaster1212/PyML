import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import ast
import re

class PyMLInterpreter:
    def __init__(self):
        self.data = None
        self.target = None
        self.features = None
        self.model = None
        self.train_data = None
        self.test_data = None

    def load(self, filepath):
        """Load a dataset from a CSV file."""
        self.data = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return self.data

    def summary(self):
        """Print a summary of the dataset."""
        if self.data is not None:
            print(self.data.describe())
        else:
            print("No data loaded.")

    def preview(self, rows=5):
        """Preview the first few rows of the dataset."""
        if self.data is not None:
            print(self.data.head(rows))
        else:
            print("No data loaded.")

    def set_target(self, target_column):
        """Set the target column."""
        self.target = target_column
        print(f"Target set to '{self.target}'.")

    def set_features(self, features):
        """Set the feature columns."""
        self.features = features
        print(f"Features set to {self.features}.")

    def split(self, ratio=0.8, shuffle=True):
        """Split the dataset into training and testing sets."""
        if self.data is not None and self.target and self.features:
            self.train_data, self.test_data = train_test_split(
                self.data, test_size=(1 - ratio), shuffle=shuffle
            )
            print(f"Data split: {ratio*100:.2f}% train, {(1-ratio)*100:.2f}% test.")
        else:
            print("Data, target, or features not set.")

    def create_model(self, model_type="ridge", alpha=1.0, regularizer=None):
        """Create a regression or classification model."""
        if model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            self.model = Lasso(alpha=alpha)
        elif model_type == "logistic":
            if regularizer == "l1":
                self.model = LogisticRegression(penalty="l1", solver="liblinear", C=1/alpha)
            else:
                self.model = LogisticRegression(penalty="l2", solver="lbfgs", C=1/alpha)
        print(f"{model_type.capitalize()} model created with alpha={alpha}.")

    def train(self):
        """Train the model on the training set."""
        if self.train_data is not None and self.model:
            X_train = self.train_data[self.features]
            y_train = self.train_data[self.target]
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")
        else:
            print("Model or training data not available.")

    def summary(self):
        """Print model summary."""
        if self.model:
            if hasattr(self.model, "coef_"):
                print("Model Coefficients:", self.model.coef_)
            if hasattr(self.model, "intercept_"):
                print("Model Intercept:", self.model.intercept_)
        else:
            print("No model available.")

    def error(self):
        """Calculate the error on the test set."""
        if self.test_data is not None and self.model:
            X_test = self.test_data[self.features]
            y_test = self.test_data[self.target]
            predictions = self.model.predict(X_test)
            if isinstance(self.model, Ridge) or isinstance(self.model, Lasso):
                error = mean_squared_error(y_test, predictions)
                print(f"Mean Squared Error: {error}")
            else:
                print("Error metric not supported for this model type.")
        else:
            print("Test data or model not available.")

    def predict(self, filepath, save=False):
        """Make predictions and save them to a file."""
        if self.test_data is not None and self.model:
            X_test = self.test_data[self.features]
            predictions = self.model.predict(X_test)
            output = pd.DataFrame(predictions, columns=["Predictions"])
            if save:
                output.to_csv(filepath, index=False)
                print(f"Predictions saved to {filepath}.")
        else:
            print("Test data or model not available.")

    def plot(self, x_values, y_values):
        """Plot x_values against y_values."""
        if self.data is not None and x_values in self.data.columns and y_values in self.data.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[x_values], self.data[y_values], alpha=0.7)
            plt.xlabel(x_values)
            plt.ylabel(y_values)
            plt.title(f"{y_values} vs {x_values}")
            plt.grid(True)
            plt.show()
        else:
            print("Columns not found in the data.")

    def describe_columns(self, columns):
        """Print summary statistics for specified columns."""
        if self.data is not None:
            if all(col in self.data.columns for col in columns):
                print(self.data[columns].describe())
            else:
                missing = [col for col in columns if col not in self.data.columns]
                print(f"Columns not found: {missing}")
        else:
            print("No data loaded.")

    def handle_missing(self, method, value=None):
        """Handle missing values in the dataset."""
        if self.data is not None:
            if method == "drop":
                self.data = self.data.dropna()
                print("Missing values dropped.")
            elif method == "fill":
                if value is None:
                    print("Specify a value to fill missing data.")
                else:
                    self.data = self.data.fillna(value)
                    print(f"Missing values filled with {value}.")
            elif method == "mean":
                self.data = self.data.fillna(self.data.mean())
                print("Missing values filled with column means.")
            elif method == "median":
                self.data = self.data.fillna(self.data.median())
                print("Missing values filled with column medians.")
            elif method == "forward_fill":
                self.data = self.data.ffill()
                print("Forward fill applied.")
            elif method == "backward_fill":
                self.data = self.data.bfill()
                print("Backward fill applied.")
            else:
                print("Invalid method. Use 'drop', 'fill', 'mean', 'median', 'forward_fill', or 'backward_fill'.")
        else:
            print("No data loaded.")

    def filter_data(self, condition):
        """Filter rows based on a condition."""
        if self.data is not None:
            try:
                self.data = self.data.query(condition)
                print(f"Rows filtered with condition: {condition}")
            except Exception as e:
                print(f"Error in filtering rows: {e}")
        else:
            print("No data loaded.")

    def normalize(self, columns, method="min-max"):
        """Normalize specified columns using Min-Max or Z-score scaling."""
        if self.data is not None:
            if method == "min-max":
                for col in columns:
                    if col in self.data.columns:
                        col_min = self.data[col].min()
                        col_max = self.data[col].max()
                        if col_max != col_min:
                            self.data[col] = (self.data[col] - col_min) / (col_max - col_min)
            elif method == "z-score":
                for col in columns:
                    if col in self.data.columns:
                        mean_val = self.data[col].mean()
                        std_val = self.data[col].std()
                        if std_val != 0:
                            self.data[col] = (self.data[col] - mean_val) / std_val
        else:
            print("No data loaded.")

    def add_feature(self, name, formula):
        """Add a new feature column based on a formula."""
        if self.data is not None:
            try:
                self.data[name] = self.data.eval(formula)
                print(f"Feature '{name}' added using formula: {formula}")
            except Exception as e:
                print(f"Error in adding feature: {e}")
        else:
            print("No data loaded.")

    def calculate_r2(self, dataset="test"):
        """Calculate R² for the train or test dataset."""
        if self.target is None:
            print("Error: Target variable is not set.")
            return

        if self.features is None or not self.features:
            print("Error: Features are not set.")
            return

        if dataset not in ["train", "test"]:
            print("Error: Invalid dataset. Use 'train' or 'test'.")
            return

        if dataset == "train" and self.train_data is not None:
            X = self.train_data[self.features]
            y_true = self.train_data[self.target]
        elif dataset == "test" and self.test_data is not None:
            X = self.test_data[self.features]
            y_true = self.test_data[self.target]
        else:
            print(f"Error: {dataset.capitalize()} data is not available.")
            return

        if self.model:
            y_pred = self.model.predict(X)
            r2 = r2_score(y_true, y_pred)
            print(f"R² ({dataset} set): {r2:.4f}")
            return r2
        else:
            print("Error: No trained model available.")

class PyMLParser:
    def __init__(self, interpreter: PyMLInterpreter):
        self.interpreter = interpreter

    def parse_line(self, line):
        line = line.strip()
        if not line or line.startswith("#"):
            return  # ignore empty or commented lines

        # Identify the command by its first keyword
        tokens = line.split(maxsplit=1)
        if len(tokens) == 0:
            return
        cmd = tokens[0]
        args = tokens[1] if len(tokens) > 1 else ""

        # Dispatch to the appropriate handler based on the command
        if cmd == "load":
            self._parse_load(args)
        elif cmd == "target":
            self._parse_target(args)
        elif cmd == "features":
            self._parse_features(args)
        elif cmd == "missing":
            self._parse_missing(args)
        elif cmd == "split":
            self._parse_split(args)
        elif cmd == "model":
            self._parse_model(args)
        elif cmd == "train":
            self.interpreter.train()
        elif cmd == "r2":
            self._parse_r2(args)
        elif cmd == "normalize":
            self._parse_normalize(args)
        elif cmd == "feature":
            self._parse_feature(args)
        elif cmd == "plot":
            self._parse_plot(args)
        elif cmd == "predict":
            self._parse_predict(args)
        elif cmd == "describe":
            self._parse_describe(args)
        else:
            print(f"Unknown command: {cmd}")

    def _parse_load(self, args):
        # load "filename.csv"
        # Extract filename between quotes
        match = re.match(r'^"([^"]+)"$', args.strip())
        if match:
            filename = match.group(1)
            self.interpreter.load(filename)
        else:
            print("Error: load command expects a quoted filename.")

    def _parse_target(self, args):
        # target "SalePrice"
        match = re.match(r'^"([^"]+)"$', args.strip())
        if match:
            target = match.group(1)
            self.interpreter.set_target(target)
        else:
            print("Error: target command expects a quoted target name.")

    def _parse_features(self, args):
        # features ["LotArea", "OverallQual", ...]
        args = args.strip()
        try:
            # Use literal_eval to parse the Python-like list
            features = ast.literal_eval(args)
            if isinstance(features, list):
                self.interpreter.set_features(features)
            else:
                print("Error: features command expects a list.")
        except Exception as e:
            print(f"Error parsing features: {e}")

    def _parse_missing(self, args):
        # missing fill 0
        # missing drop
        # missing mean
        parts = args.split()
        if len(parts) == 1:
            # e.g. missing drop/mean/median/forward_fill/backward_fill
            method = parts[0]
            self.interpreter.handle_missing(method=method)
        elif len(parts) == 2:
            # e.g. missing fill 0
            method, val = parts
            # Try convert val to int/float if numeric
            val = self._convert_value(val)
            self.interpreter.handle_missing(method=method, value=val)
        else:
            print("Error: invalid missing command format.")

    def _parse_split(self, args):
        # split ratio=0.8 shuffle
        # split ratio=0.7 noshuffle
        ratio_match = re.search(r'ratio=([\d.]+)', args)
        shuffle = True
        if ratio_match:
            ratio = float(ratio_match.group(1))
        else:
            print("Error: split command requires ratio=<float>.")
            return

        if "noshuffle" in args:
            shuffle = False

        self.interpreter.split(ratio=ratio, shuffle=shuffle)

    def _parse_model(self, args):
        # model ridge alpha=1.0
        # model lasso alpha=0.5
        # model logistic l1 alpha=0.1
        parts = args.split()
        model_type = parts[0]
        regularizer = None
        alpha = 1.0
        for p in parts[1:]:
            if p == "l1":
                regularizer = "l1"
            elif p.startswith("alpha="):
                alpha = float(p.split('=')[1])
        self.interpreter.create_model(model_type=model_type, alpha=alpha, regularizer=regularizer)

    def _parse_r2(self, args):
        # r2 train or r2 test
        dataset = args.strip()
        if dataset in ["train", "test"]:
            self.interpreter.calculate_r2(dataset=dataset)
        else:
            print("Error: r2 command expects 'train' or 'test'.")

    def _parse_normalize(self, args):
        # normalize ["TotalArea", "GrLivArea"] method=zscore
        # split args into two parts: the array and the method
        method_match = re.search(r'method=(\S+)', args)
        if method_match:
            method = method_match.group(1)
            method = method.lower()
            # Extract the array part by removing the method part
            array_part = args.split('method=')[0].strip()
            try:
                columns = ast.literal_eval(array_part)
                self.interpreter.normalize(columns, method=method if method != "minmax" else "min-max")
            except Exception as e:
                print(f"Error parsing normalize command: {e}")
        else:
            print("Error: normalize command expects method=<method>.")

    def _parse_feature(self, args):
        # feature "TotalArea" = "LotArea + GrLivArea + TotalBsmtSF"
        # Extract name and formula
        match = re.match(r'^"([^"]+)"\s*=\s*"([^"]+)"$', args.strip())
        if match:
            name = match.group(1)
            formula = match.group(2)
            self.interpreter.add_feature(name, formula)
        else:
            print("Error: feature command should be in format: feature \"Name\" = \"formula\"")

    def _parse_plot(self, args):
        # plot x="OverallQual" y="SalePrice"
        x_match = re.search(r'x="([^"]+)"', args)
        y_match = re.search(r'y="([^"]+)"', args)
        if x_match and y_match:
            x_val = x_match.group(1)
            y_val = y_match.group(1)
            self.interpreter.plot(x_values=x_val, y_values=y_val)
        else:
            print("Error: plot command requires x=\"...\" y=\"...\"")

    def _parse_predict(self, args):
        # predict save="predictions.csv"
        save_match = re.search(r'save="([^"]+)"', args)
        if save_match:
            filename = save_match.group(1)
            self.interpreter.predict(filepath=filename, save=True)
        else:
            print("Error: predict command requires save=\"filename\"")

    def _convert_value(self, val_str):
        # Try int
        if val_str.isdigit():
            return int(val_str)
        # Try float
        try:
            return float(val_str)
        except ValueError:
            return val_str
        
    def _parse_describe(self, args):
        args = args.strip()
        # If args starts with '[', treat it as an array of strings
        if args.startswith('['):
            try:
                columns = ast.literal_eval(args)
                if isinstance(columns, list):
                    self.interpreter.describe_columns(columns)
                else:
                    print("Error: describe expects a list of strings or a single string.")
            except Exception as e:
                print(f"Error parsing describe command: {e}")
        else:
            # Single column scenario
            # If user wrote: describe col (no quotes), ensure col is a string
            # For consistency with previous commands, we can require quotes around single columns
            # or allow raw words. If you want raw words:
            # Just treat the raw arg as a column name:
            column = args.strip('"')  # remove quotes if any
            self.interpreter.describe_columns([column])

    def parse_script(self, script):
        for line in script.split('\n'):
            self.parse_line(line)

def main():
    interpreter = PyMLInterpreter()
    parser = PyMLParser(interpreter)
    print("Welcome to the PyML REPL. Type 'exit' or 'quit' to end the session.")

    while True:
        # Read a command from the user
        line = input(">>> ").strip()

        # Check for exit conditions
        if line.lower() in ["exit", "quit"]:
            print("Exiting PyML REPL.")
            break

        # Parse and execute the command
        parser.parse_line(line)

if __name__ == "__main__":
    main()

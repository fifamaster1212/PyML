import unittest
from io import StringIO
from contextlib import redirect_stdout
import os
from PyMLinterpreter import PyMLInterpreter, PyMLParser

class TestDSLParser(unittest.TestCase):

    def setUp(self):
        self.interpreter = PyMLInterpreter()
        self.parser = PyMLParser(self.interpreter)

    def _run_dsl(self, dsl_script):
        with StringIO() as buf, redirect_stdout(buf):
            self.parser.parse_script(dsl_script)

    def test_load(self):
        dsl_script = 'load "data.csv"'
        self._run_dsl(dsl_script)
        self.assertIsNotNone(self.interpreter.data, "Data should be loaded into the interpreter.")

    def test_target(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('target "SalePrice"')
        self.assertEqual(self.interpreter.target, "SalePrice", "Target should be set to 'SalePrice'.")

    def test_features(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('target "SalePrice"')
        self._run_dsl('features ["LotArea", "OverallQual", "YearBuilt", "GarageArea", "GrLivArea"]')
        self.assertEqual(self.interpreter.features, ["LotArea", "OverallQual", "YearBuilt", "GarageArea", "GrLivArea"],
                         "Features should be set as specified.")

    def test_missing(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('missing fill 0')
        # Since this is mocked, we can't assert state changes unless interpreter modifies its data.
        # For now, we just confirm no exceptions.

    def test_split(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('target "SalePrice"')
        self._run_dsl('features ["LotArea"]')
        self._run_dsl('split ratio=0.8 shuffle')
        self.assertIsNotNone(self.interpreter.train_data, "Train data should be set after splitting.")
        self.assertIsNotNone(self.interpreter.test_data, "Test data should be set after splitting.")

    def test_model(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('target "SalePrice"')
        self._run_dsl('features ["LotArea"]')
        self._run_dsl('model ridge alpha=1.0')
        self.assertIsNotNone(self.interpreter.model, "A model should be created and stored in the interpreter.")

    def test_train(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('target "SalePrice"')
        self._run_dsl('features ["LotArea"]')
        self._run_dsl('model ridge alpha=1.0')
        self._run_dsl('train')
        # just ensure no errors were raised.

    def test_r2(self):
        # Set up the environment for training
        self._run_dsl('load "data.csv"')
        self._run_dsl('target "SalePrice"')
        self._run_dsl('features ["LotArea"]')
        self._run_dsl('model ridge alpha=1.0')
        self._run_dsl('train')
        self._run_dsl('r2 train')
        self._run_dsl('r2 test')
        
    def test_feature(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('feature "TotalArea" = "LotArea + GrLivArea + TotalBsmtSF"')
        # Since it's mocked, just ensure no exceptions.

    def test_normalize(self):
        self._run_dsl('load "data.csv"')
        # Deep copy to ensure we have a baseline snapshot
        initial_sum = self.interpreter.data['GrLivArea'].sum()
        self._run_dsl('normalize ["GrLivArea"] method=z-score')
        # sum the data to ensure it's been normalized
        data_sum = self.interpreter.data['GrLivArea'].sum()
        self.assertNotEqual(initial_sum, data_sum, "Data should be normalized.")

    def test_plot(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('plot x="OverallQual" y="SalePrice"')
        # Just ensure no errors. Plotting doesn't change internal state.

    def test_predict(self):
        self._run_dsl('load "data.csv"')
        self._run_dsl('features ["LotArea"]')
        self._run_dsl('target "SalePrice"')
        self._run_dsl('model ridge alpha=1.0')
        self._run_dsl('split ratio=0.8')
        self._run_dsl('train')
        # if predictions.csv exists, delete it
        if 'predictions.csv' in os.listdir():
            os.remove('predictions.csv')
        self._run_dsl('predict save="predictions.csv"')
        # assert that predictions.csv was created
        self.assertTrue('predictions.csv' in os.listdir(), "Predictions should be saved to predictions.csv.")

if __name__ == '__main__':
    unittest.main()

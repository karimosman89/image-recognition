import unittest
import numpy as np
from model import create_model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = create_model()

    def test_model_compilation(self):
        self.assertEqual(self.model.loss, 'sparse_categorical_crossentropy')

if __name__ == '__main__':
    unittest.main()

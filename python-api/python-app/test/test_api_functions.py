import unittest
import os
import numpy as np
from io import BytesIO
import json

from pydantic import ValidationError
import tensorflow as tf

from api_functions import ApiFunctions
from main import NumpyBody


tf.get_logger().setLevel('ERROR')


class testApiFuncitonsClass(unittest.TestCase):

    def setUp(self):
        self.api = ApiFunctions()

    def tearDown(self):
        self.api = ApiFunctions()

    def test_get_api_version(self):
        self.assertEqual(self.api.get_api_version(), 0.1)

    def test_get_possible_extensions(self):
        self.assertEqual(self.api.get_possible_extensions(),
                         ['csv', 'npy', 'pkl', 'json'])

    def test_read_file(self):
        ''' All the files are converted to numpy arrays and are checked by reading in the numpy array (not converting each file to check) '''
        npy_file = None
        # Numpy File
        with open(os.path.join(os.getcwd(), 'test/train_item.npy'), "rb") as npy_file:
            npy_file = npy_file.read()
            self.assertEqual(self.api.read_file(
                npy_file, 'npy').all(), np.load(BytesIO(npy_file)).all())

        # CSV
        with open(os.path.join(os.getcwd(), 'test/train_item.csv'), "rb") as csv_file:
            self.assertEqual(self.api.read_file(
                csv_file.read(), 'csv').all(), np.load(BytesIO(npy_file)).all())

        # DF
        with open(os.path.join(os.getcwd(), 'test/train_item.pkl'), "rb") as df_file:
            self.assertEqual(self.api.read_file(
                df_file.read(), 'pkl').all(), np.load(BytesIO(npy_file)).all())

    def test_process_numpy(self):
        file_ext = 'npy'
        with open(os.path.join(os.getcwd(), f'test/train_item.{file_ext}'), 'rb') as npy:
            np_array = self.api.read_file(npy.read(), file_ext)
        self.assertEqual(self.api.process_numpy(np_array, file_ext)[0].shape,
                         (1, 60, 31, 1))
        self.assertEqual(self.api.process_numpy(np_array, file_ext)[1].shape,
                         (1, 9))

    def test_predict_trafic(self):
        file_ext = 'npy'
        # with open(os.path.join(os.getcwd(), f'test/test_item.{file_ext}'), "rb") as test_data:
        #     prediction = self.api.read_file(test_data.read(), file_ext)
        with open(os.path.join(os.getcwd(), f'test/train_item.{file_ext}'), "rb") as train_data:
            array_to_process = self.api.read_file(train_data.read(), file_ext)

        input_conv, input_val = self.api.process_numpy(
            array_to_process, file_ext)
        print(self.api.predict_trafic(input_conv, input_val).all())
        self.assertIsNotNone(self.api.predict_trafic(
            input_conv, input_val).all())

    def test_numpy_body(self):
        with open(os.path.join(os.getcwd(), 'test/test_item.csv'), 'rb') as test_data:
            test_data = self.api.read_file(test_data.read(), 'csv')

        test_data = json.dumps(test_data.tolist())
        parsed_array = NumpyBody(Array=test_data)

        self.assertIsInstance(parsed_array.Array, np.ndarray)

    def test_numy_body_raises(self):
        self.assertRaises(ValidationError, NumpyBody, Array='invalid')


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import json

from fastapi.testclient import TestClient
from fastapi import File, UploadFile
import numpy as np
from main import app

client = TestClient(app)

class TestApi(unittest.TestCase):
    def test_predict_file(self):
        with open(os.path.join(os.getcwd(), 'test/train_item.npy'), 'rb') as f:
            resp = client.post('/predict/file', files={"files": ("test_item.npy", f, "data/nyp")})
            
        self.assertEqual(resp.status_code, 200)
        predict = json.loads(resp.text)
        self.assertIsNotNone(predict['predictions'])

#    def test_predict_nparray(self):
        # Unable to test at this time: see https://github.com/tiangolo/fastapi/issues/459

        # ndarray = np.fromfile(os.path.join(os.getcwd(), 'test/test_item.npy'))
        # json_ndarray = json.dumps(ndarray.tolist())
        # resp = client.post('/predict/', json={'Array': json_ndarray})
        # print(resp.text)
        # self.assertEqual(resp.status_code, 200)
        # predict = json.loads(resp.text)
        # self.assertIsNotNone(predict['predictions'])



if __name__ == '__main__':
    unittest.main()
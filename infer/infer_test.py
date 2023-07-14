import unittest
import os
import random
import sys
from infer_new import infer
from parameterized import parameterized
from flask import Flask, request
from flask.testing import FlaskClient
from io import BytesIO
from werkzeug.datastructures import FileStorage
import requests

app = Flask(__name__)
test_data_path = "../testdatapool"

def select_one_jpg(number):
    path = os.path.join(test_data_path, str(number))
    jpg_files = [file for file in os.listdir(path) if file.endswith('.jpg')]
    if jpg_files:
        random_file = random.choice(jpg_files)
        return os.path.join(path, random_file)
    else:
        return None
    
class TestInfer(unittest.TestCase):

    # paraeterized decorator로 하나의 함수로 가능한 여러 테스트를 한 번의 정의로 가능합니다.
    @parameterized.expand([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_infer(self, number):
        _, result = infer(select_one_jpg(number))
        self.assertEqual(result, number)



if __name__ == "__main__":
    success_threshold = 8
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInfer)
    result = unittest.TextTestRunner().run(suite)
    if len(result.errors) > 0:
        print("Error occured")
        exit(1)
    fail_count = len(result.failures)
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    print(success_count, "/", result.testsRun)
    if success_count >= success_threshold:
        print("Success")
        exit(0)
    else:
        print("Failure")
        exit(1)
"""用临时端口验证真实模型 HTTP 预测和输入错误处理。"""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from pathlib import Path
import pandas as pd
from src.predictor import KPIPredictor
from src.server import make_handler

class APITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = KPIPredictor.load('models/kpi_ai.joblib')
        cls.payload = json.loads(Path('examples/congestion.json').read_text())
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
            cls.predictor, pd.read_csv('data/kpi_dataset.csv'), '<h1>demo</h1>'))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_health(self):
        with urlopen(self.base + '/health') as response:
            self.assertEqual(json.load(response)['status'], 'ok')

    def test_predict(self):
        request = Request(self.base + '/api/predict', json.dumps(self.payload).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request) as response:
            self.assertEqual(json.load(response)['root_cause'], 'congestion')

    def test_invalid_limit(self):
        for value in ['-1', '0', '201', 'abc']:
            with self.subTest(value=value), self.assertRaises(HTTPError) as caught:
                urlopen(self.base + '/api/anomalies?limit=' + value)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()

    def test_invalid_features(self):
        for payload in [{}, [], dict(self.payload, sinr_db='bad'),
                        dict(self.payload, sinr_db=float('inf'))]:
            with self.subTest(kind=type(payload).__name__), self.assertRaises(ValueError):
                self.predictor.predict_one(payload)

    def test_invalid_json(self):
        request = Request(self.base + '/api/predict', b'not-json',
                          {'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()

if __name__ == '__main__':
    unittest.main()

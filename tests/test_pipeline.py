import tempfile
import unittest
from pathlib import Path

from src.generate_data import build_dataset
from src.model import FEATURES, train
from src.predictor import KPIPredictor


class PipelineTest(unittest.TestCase):
    def test_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = build_dataset(rows=2400, cells=6, seed=9)
            data_path = tmp / "data.csv"
            model_path = tmp / "model.joblib"
            data.to_csv(data_path, index=False)
            metrics = train(str(data_path), str(model_path), str(tmp / "reports"), seed=9)
            self.assertGreater(metrics["anomaly_f1"], 0.90)
            predictor = KPIPredictor.load(str(model_path))
            result = predictor.predict_one(data.iloc[0][FEATURES].to_dict())
            self.assertIn("root_cause", result)
            self.assertIn("advice", result)


if __name__ == "__main__":
    unittest.main()

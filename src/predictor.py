"""统一推理入口：校验特征、分类、辅助打分并映射静态排障建议。"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd


CAUSE_CN = {
    "normal": "正常", "congestion": "容量拥塞", "weak_coverage": "弱覆盖",
    "interference": "无线干扰", "transmission": "传输链路异常",
}

ADVICE = {
    "normal": "持续监控，无需处置。",
    "congestion": "核查忙时PRB与用户数，评估载波扩容、负载均衡或参数优化。",
    "weak_coverage": "结合路测检查覆盖盲区、天线方位角/下倾角及邻区配置。",
    "interference": "排查同频干扰与PCI冲突，复核频点、功率和邻区参数。",
    "transmission": "检查回传链路丢包、时延、端口告警及设备可用性。",
}


@dataclass
class KPIPredictor:
    """封装训练产物，保证训练和推理采用相同的特征顺序。"""

    bundle: dict

    @classmethod
    def load(cls, path: str) -> "KPIPredictor":
        """仅加载可信模型文件；不要加载用户上传的未知 joblib 文件。"""
        return cls(joblib.load(path))

    def predict_frame(self, frame: pd.DataFrame) -> list[dict]:
        features = self.bundle["features"]
        # 请求允许携带额外元数据，但必须包含所有模型特征。
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError("缺少特征：" + ", ".join(missing))
        if frame.empty:
            raise ValueError("至少需要一条记录")
        try:
            X = frame[features].apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("特征必须为数值") from exc
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("特征不能包含 NaN 或无穷值")
        classes = self.bundle["classifier"].classes_
        probabilities = self.bundle["classifier"].predict_proba(X)
        causes = classes[np.argmax(probabilities, axis=1)]
        # 取反后越大越异常；大于零对应模型默认异常阈值，不是概率。
        scores = -self.bundle["detector"].decision_function(self.bundle["scaler"].transform(X))
        results = []
        for cause, probs, score in zip(causes, probabilities, scores):
            results.append({
                "is_anomaly": bool(cause != "normal"),
                "root_cause": str(cause),
                "root_cause_cn": CAUSE_CN[str(cause)],
                # 随机森林预测概率的最大值，未经概率校准。
                "confidence": round(float(np.max(probs)), 4),
                "unsupervised_anomaly_score": round(float(score), 4),
                "advice": ADVICE[str(cause)],
            })
        return results

    def predict_one(self, payload: dict) -> dict:
        """单条预测供 HTTP 接口和演示使用。"""
        if not isinstance(payload, dict):
            raise ValueError("请求体必须为 JSON 对象")
        return self.predict_frame(pd.DataFrame([payload]))[0]

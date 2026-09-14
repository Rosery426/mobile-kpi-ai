"""离线训练与评估：随机森林负责分类，孤立森林只提供辅助异常分数。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "hour", "peak_hour", "active_users", "rsrp_dbm", "sinr_db", "prb_util_pct",
    "dl_throughput_mbps", "latency_ms", "packet_loss_pct", "handover_success_pct",
    "availability_pct",
]


def train(data_path: str, model_path: str, report_dir: str, seed: int = 2027) -> dict:
    """训练模型并保存模型包、指标和混淆矩阵。

    采用同一生成分布的分层随机留出集，不能据此推断真实网络泛化能力。
    """
    df = pd.read_csv(data_path)
    # 显式白名单排除 root_cause/is_anomaly 标签及小区标识，避免直接标签泄漏。
    X = df[FEATURES]
    y = df["root_cause"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # 只在训练集拟合预处理；测试集不参与 fit。
    scaler = StandardScaler().fit(X_train)
    normal_scaled = scaler.transform(X_train[y_train == "normal"])
    detector = IsolationForest(n_estimators=220, contamination=0.035, random_state=seed, n_jobs=-1)
    detector.fit(normal_scaled)

    # 主分类器直接使用原始特征；树模型无需标准化。
    classifier = RandomForestClassifier(
        n_estimators=260, max_depth=14, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    # 下列二分类指标来自 RF 是否预测 normal，不是 Isolation Forest 的指标。
    anomaly_true = (y_test != "normal").astype(int)
    anomaly_pred = (pred != "normal").astype(int)
    metrics = {
        "evaluation_source": "RandomForest predictions against noisy synthetic labels",
        "seed": seed,
        "train_rows": int(len(X_train)),
        "rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "anomaly_precision": round(float(precision_score(anomaly_true, anomaly_pred)), 4),
        "anomaly_recall": round(float(recall_score(anomaly_true, anomaly_pred)), 4),
        "anomaly_f1": round(float(f1_score(anomaly_true, anomaly_pred)), 4),
        "root_cause_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "root_cause_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    bundle = {"features": FEATURES, "scaler": scaler, "detector": detector, "classifier": classifier}
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    # joblib 包含可执行的反序列化内容，只应加载自己训练或可信来源的文件。
    joblib.dump(bundle, model_file)

    report = Path(report_dir)
    report.mkdir(parents=True, exist_ok=True)
    (report / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues", xticks_rotation=25, colorbar=False)
    ax.set_title("Root-cause classification — held-out test set")
    fig.tight_layout()
    fig.savefig(report / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kpi_dataset.csv")
    parser.add_argument("--model", default="models/kpi_ai.joblib")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model, args.reports), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

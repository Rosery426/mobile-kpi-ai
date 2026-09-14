"""本地演示 HTTP 服务：模型启动时加载一次，默认仅监听回环地址。"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.predictor import KPIPredictor


def make_handler(predictor: KPIPredictor, data: pd.DataFrame, dashboard: str):
    """闭包共享只读数据和模型；看板统计的是仿真标签，不是模型预测。"""
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json({"status": "ok", "model": "kpi-ai-v1"})
            elif parsed.path == "/api/summary":
                counts = data["root_cause"].value_counts().to_dict()
                self.send_json({"rows": len(data), "cells": data["cell_id"].nunique(), "root_cause_counts": counts})
            elif parsed.path == "/api/anomalies":
                try:
                    limit = int(parse_qs(parsed.query).get("limit", [20])[0])
                    if not 1 <= limit <= 200:
                        raise ValueError()
                except ValueError:
                    self.send_json({"error": "limit 必须为 1 至 200 的整数"}, 400)
                    return
                sample = data[data["is_anomaly"] == 1].tail(limit)
                self.send_json(sample.to_dict(orient="records"))
            elif parsed.path in ("/", "/index.html"):
                body = dashboard.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/predict":
                self.send_json({"error": "not found"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    self.send_json({"error": "请求体须为 1 至 65536 字节"}, 400)
                    return
                payload = json.loads(self.rfile.read(size))
                self.send_json(predictor.predict_one(payload))
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/kpi_ai.joblib")
    parser.add_argument("--data", default="data/kpi_dataset.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    predictor = KPIPredictor.load(args.model)
    data = pd.read_csv(args.data)
    dashboard = Path("web/index.html").read_text(encoding="utf-8")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(predictor, data, dashboard))
    print(f"dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

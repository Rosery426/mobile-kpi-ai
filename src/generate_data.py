"""生成可复现的教学仿真 KPI；不使用运营商真实数据。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CAUSES = ["normal", "congestion", "weak_coverage", "interference", "transmission"]


def build_dataset(rows: int = 18000, cells: int = 24, seed: int = 2027) -> pd.DataFrame:
    """生成 rows 条记录，随机分配至 cells 个小区。

    全局时间戳每次递增 5 分钟，并非每个小区连续采样。
    seed 固定随机抽样；标签噪声模拟误标，不代表真实故障分布。
    """
    if rows <= 0 or cells <= 0:
        raise ValueError("rows 和 cells 必须为正整数")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-08-01", periods=rows, freq="5min")
    cell_ids = np.array([f"ZZ-CELL-{i:03d}" for i in range(1, cells + 1)])
    cell = rng.choice(cell_ids, rows)
    hour = timestamps.hour.to_numpy()
    peak = (((hour >= 11) & (hour <= 13)) | ((hour >= 18) & (hour <= 22))).astype(int)
    active_users = np.clip(rng.normal(65 + peak * 55, 18, rows), 5, 210)
    prb = np.clip(rng.normal(34 + peak * 26 + active_users * 0.10, 8, rows), 5, 98)
    rsrp = np.clip(rng.normal(-91, 6, rows), -120, -68)
    sinr = np.clip(rng.normal(17, 4.5, rows), -5, 30)
    latency = np.clip(rng.normal(22 + peak * 3, 5, rows), 5, 80)
    packet_loss = np.clip(rng.lognormal(-2.6, 0.45, rows), 0, 5)
    handover = np.clip(rng.normal(98.2, 0.8, rows), 88, 100)
    availability = np.clip(rng.normal(99.93, 0.05, rows), 98.8, 100)
    throughput = np.clip(155 - prb * 0.72 + sinr * 2.2 + rng.normal(0, 12, rows), 2, 240)

    # 先生成正常指标，再依据故障类型覆写相关指标。
    cause = rng.choice(CAUSES, rows, p=[0.66, 0.12, 0.09, 0.07, 0.06])
    for name in CAUSES[1:]:
        idx = cause == name
        count = idx.sum()
        if name == "congestion":
            prb[idx] = np.clip(rng.normal(94, 3, count), 80, 100)
            active_users[idx] = np.clip(rng.normal(175, 18, count), 110, 240)
            throughput[idx] = np.clip(rng.normal(24, 9, count), 1, 60)
            latency[idx] = np.clip(rng.normal(74, 18, count), 35, 180)
        elif name == "weak_coverage":
            rsrp[idx] = np.clip(rng.normal(-116, 3.5, count), -125, -103)
            sinr[idx] = np.clip(rng.normal(2, 3, count), -8, 9)
            throughput[idx] = np.clip(rng.normal(18, 8, count), 1, 50)
            handover[idx] = np.clip(rng.normal(91, 2.5, count), 80, 97)
        elif name == "interference":
            sinr[idx] = np.clip(rng.normal(-1, 3, count), -10, 7)
            rsrp[idx] = np.clip(rng.normal(-91, 5, count), -108, -75)
            packet_loss[idx] = np.clip(rng.normal(3.8, 1.1, count), 1.2, 9)
            throughput[idx] = np.clip(rng.normal(27, 10, count), 1, 65)
        elif name == "transmission":
            latency[idx] = np.clip(rng.normal(145, 28, count), 75, 260)
            packet_loss[idx] = np.clip(rng.normal(6.5, 1.7, count), 2.5, 15)
            availability[idx] = np.clip(rng.normal(98.7, 0.45, count), 96.5, 99.7)
            throughput[idx] = np.clip(rng.normal(13, 6, count), 0.5, 38)

    # 以 3% 的概率更换标签，指标保持不变；评估对象因此是有噪声的标签。
    noisy = rng.random(rows) < 0.03
    for i in np.flatnonzero(noisy):
        alternatives = [item for item in CAUSES if item != cause[i]]
        cause[i] = rng.choice(alternatives)

    return pd.DataFrame({
        "timestamp": timestamps.astype(str), "cell_id": cell, "hour": hour,
        "peak_hour": peak, "active_users": active_users.round(0).astype(int),
        "rsrp_dbm": rsrp.round(2), "sinr_db": sinr.round(2),
        "prb_util_pct": prb.round(2), "dl_throughput_mbps": throughput.round(2),
        "latency_ms": latency.round(2), "packet_loss_pct": packet_loss.round(3),
        "handover_success_pct": handover.round(2), "availability_pct": availability.round(3),
        "root_cause": cause, "is_anomaly": (cause != "normal").astype(int),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=18000)
    parser.add_argument("--cells", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default="data/kpi_dataset.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset(args.rows, args.cells, args.seed)
    df.to_csv(output, index=False)
    print(f"wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()

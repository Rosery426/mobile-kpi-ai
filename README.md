# 移动网络 KPI 智能运维原型

Python + scikit-learn 构建的可复现教学项目，覆盖仿真数据生成、五分类模型训练、辅助异常打分、REST API 和标签统计看板。

**所有数据均为仿真；没有使用真实运营商数据或大语言模型。** 看板展示标签统计，单条模型预测通过 POST /api/predict 调用。排障建议为静态映射。

## 快速开始

需要 Python 3.12，建议在虚拟环境中运行：

```bash
python -m pip install -r requirements.txt
python -m src.generate_data
python -m src.model
python -m unittest discover -s tests -v
python -m src.server --port 8080
```

在浏览器打开 http://127.0.0.1:8080。请始终从项目根目录执行命令。

## 模型与评估

- Random Forest：11 个输入特征，正常与四类故障分类。
- Isolation Forest：正常标签训练样本上的辅助分数，不参与最终分类。
- 18,000 条记录，24 个仿真小区；全局时间步长 5 分钟，非逐小区连续采样。
- 分层随机划分为 13,500 条训练和 4,500 条测试记录。
- 随机森林异常 F1 96.43%，五分类准确率 96.58%，Macro-F1 94.43%。

指标仅针对带人为标签噪声的同分布仿真数据；不能证明真实网络泛化能力。完整解释见 [项目详细说明](docs/项目详细说明.md)，机器可读结果见 [metrics.json](reports/metrics.json)。

## 文件导航

| 路径 | 说明 |
|---|---|
| src/generate_data.py | 仿真生成与标签噪声 |
| src/model.py | 训练、评估、产物导出 |
| src/predictor.py | 输入校验及推理 |
| src/server.py | 本地 HTTP 服务 |
| web/index.html | 标签统计看板 |
| tests/ | 自动测试 |
| examples/congestion.json | 预测请求样例 |
| docs/项目详细说明.md | 技术、参数、训练、接口、运行与面试说明 |
| data/、models/、reports/ | 可复现数据、可信预训练模型及报告 |

只加载可信模型文件。服务为本地演示原型，不包含生产环境的鉴权、限流或高可用配置。

# 真实 VLM Fallback 切片评测报告

> 数据集：data/vision_fallback_cases.jsonl，样本数：26，图纸为脚本合成（无真实企业图纸）。

- 用例整体成功率：88.5%
- metadata 成功率：93.3%（14/15）
- table 成功率：66.7%（4/6）
- relations 成功率：100.0%（5/5）

## Validator 集成结果（真实 FAN-MULTI-01 抽取结果 + stub OCR）
- {'label': 'matching_ocr_ledger_hit', 'device_id': 'A16', 'ocr_hint': 'A16', 'status': 'confirmed', 'warnings': []}
- {'label': 'mismatched_ocr_ledger_hit', 'device_id': 'A17', 'ocr_hint': 'A1?', 'status': 'suspected', 'warnings': ["OCR 读取为 'A1?'，VLM 读取为 'A17' 且与台账匹配，倾向采用 VLM 结果，OCR 原始值已保留"]}
- {'label': 'no_ocr_no_ledger', 'device_id': 'A18', 'ocr_hint': None, 'status': 'suspected', 'warnings': ['设备编号 A18 台账中未找到匹配，仅有 VLM 单一来源']}

## 重复调用稳定性（FAN-A13-02.png，重复 3 次，不走缓存）
- 每次结果完全一致：True

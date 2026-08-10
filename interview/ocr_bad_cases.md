# OCR Pipeline Bad Case 报告

> 自动收集自 `outputs/ocr_eval_report.json`（真实 A/B/C/D/E 实验结果）与实现阶段发现并修复的代码 bug。归因（failure_type/根因/修复方案）由人工逐条标注，不做自动分类——见脚本 `src/vision/bad_case_report.py` 顶部注释：这一步的判断没有可靠的自动化信号来源，硬做就是伪造。

## 一、真实评测中发现的 Bad Case

### OCR05（FAN-A22-01-heavyblur.png，blur）

- **各实验得分**：{'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0}
- **归因类型**：`model_error`
- **根因**：重度模糊（约1/7降采样再放大）下，真实 VLM 在这次跑批中同样对全部字段返回 null（诚实），但本次会话更早时候用同一张图做过更细的重复实验，观察到在类似退化程度下模型有时会编造一个格式正常、看起来合理但实际错误的图号（例如把这张图读成了本仓库最早示例图纸的图号 FAN-A12-01），而不是承认读不出来。两种失败方式（沉默 vs 编造）在这个退化区间都真实出现过，说明'不确定返回 null'这条规则的可靠性会随图像质量退化程度非线性地滑坡，不能只靠 Prompt 里写一句话就完全信任。
- **修复方案**：在 Validator/QualityJudge 层对图像质量本身做独立于模型自我报告的客观判断（例如本轮已实现的sharpness/contrast 指标），质量低于某个硬阈值时直接标记为不可信或转人工复核，不能把'模型会不会编造'这件事完全交给模型自己判断。
- **状态**：known_limitation

### OCR10（FAN-A24-01.png，blur）

- **各实验得分**：{'A': 0.0, 'B': 0.0, 'C': 0.75, 'D': 0.0, 'E': 0.75}
- **归因类型**：`preprocess_error`
- **根因**：原图（FAN-A24-01，模糊+水印复合退化）上真实 VLM 调用能读出 3/4 关键字段（图号、电机编号、控制柜型号），唯独断路器编号被读错（QF-24→CF-24，Q/C 形近误读）。经过真实 cv2 预处理（去噪+Otsu二值化）后，VLM 对同一设备返回了全部字段为 null——不是编造，是诚实报告读不出来，但信息量从 3/4 降到 0/4。根因是 Otsu 全局二值化无法区分'真实文字'和'半透明水印噪点'，两者灰度接近时二值化会把水印当前景保留、同时把本来还能辨认的浅色正文一并抹掉。
- **修复方案**：不应对模糊+水印复合退化的图像默认执行全局二值化；PreprocessDecision 应该在检测到水印类退化（目前的 ImageQualityMetrics 没有专门的水印检测信号）时跳过 binarize，或改用自适应阈值而非全局 Otsu。这是这一轮找到的一个尚未修复的真实产品缺陷，不是评测脚本或标注的问题。
- **状态**：known_limitation

## 二、实现阶段发现并修复的代码 Bug

### IMPL01（FAN-A13-02.png (title-block ROI)，complex_table）

- **归因类型**：`evaluation_bug`
- **根因**：table_structure.py 最初默认'给了 OCR 结果就把 cv2 检测到的第一行网格当表头'，但标题栏这种'标签:数值'两列表格根本没有表头行——第一行数据被静默丢进一个没人读取的 headers 字段，永远拿不回来。写 test_cell_text_is_populated_when_ocr_result_is_supplied 这条测试时被抓到。
- **修复方案**：加了 has_header_row 参数，默认 False，调用方明确知道这是不是真的有表头的表才由它决定，不再靠几何形状猜。
- **状态**：fixed_and_passing（已有回归测试锁定）

### IMPL02（FAN-A28-01-borderless.png，complex_table）

- **归因类型**：`evaluation_bug`
- **根因**：detect_table_structure 把每张合成图纸自带的最外层页面边框（2条横线+2条竖线，所有图都有）误判成了一个 1x1 的'表格'，无框表格测试因此返回 confidence=1.0 和一行编造的空数据，而不是诚实的'没有检测到表格'。
- **修复方案**：要求至少一个方向上有内部分隔线（n_rows>1 或 n_cols>1），单纯的外边框不再被当成表格。
- **状态**：fixed_and_passing（已有回归测试锁定）

### IMPL03（FAN-A13-02.png，normal）

- **归因类型**：`schema_error`
- **根因**：真实 VLM 在处理标题栏这类混合了数字参数和纯标识符字段（如电机编号 M-13）的表格时，偶尔把非数字字符串塞进了 DeviceParameter.value（类型是 Optional[float]），导致 Pydantic 校验直接抛异常，整次抽取失败退出，没有任何降级处理。这是本轮第一次真实批量跑 extract_table 就撞见的问题，不是构造出来的。
- **修复方案**：抽取后、校验前对每个 parameter 的 value 做一次可解析性检查，解析不了就置 None，raw_text 原样保留，不让一个字段的类型不匹配拖垮整次抽取。
- **状态**：fixed_and_passing（已有回归测试锁定）

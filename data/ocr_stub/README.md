# OCR Stub 数据

这个目录下的 `*.json` 文件是 `src/vision/ocr_engine.py::MockOCREngine` 用的手写 OCR 结果，**不是任何真实 OCR 引擎的输出**。

原因：这台开发机上 `pytesseract` 缺少底层 `tesseract.exe` 二进制，`paddleocr` 能真实下载模型权重但推理阶段会抛出 Paddle 内部框架错误（详见 `src/vision/ocr_engine.py::PaddleOCREngine` 的类注释）。真实 OCR 推理在这台机器上跑不通，因此用手写 stub 顶上，让 Validator/QualityJudge 等下游逻辑能被测试到。

每个文件按 `<图片文件名去掉扩展名>.json` 命名，对应 `data/drawings/` 下的同名图片。**内容故意包含了真实 OCR 常见的误读模式**（形近字符：`1`/`I`/`l`，`0`/`O`；数字被印章/水印遮挡后的残缺或空白；低置信度但语义正确的读数），不是全部照抄图片上的正确文字——如果 stub 总是"标准答案"，那么任何依赖它的 Validator 交叉校验测试都只是在自证，没有意义。

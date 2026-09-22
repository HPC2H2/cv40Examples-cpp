# 代码阅读指南

## 从一个例子开始

先在 [完整对照表](EXAMPLES.md) 找到教材例号，运行 `build\cv40.exe --run 17.6 --headless`，再打开对应的 `src/ch17_sudoku.cpp`。生成的图片按处理顺序编号，能与函数里的 `context.show()` 一一对应。

`src/registry.cpp` 保存原 Python 路径、例号、章节和资源需求；`src/main.cpp` 只负责命令行解析、选择例子和汇总结果；`src/runtime.cpp` 负责读写文件、显示结果和处理帧。具体算法集中在各章文件中。

## 本轮整理的算法

| 内容 | 入口与阅读顺序 |
|---|---|
| 基础分类器 | `ch15_classifiers.cpp`：构造训练数据 → 训练 → 预测 |
| KNN 数字/字母识别 | `ch16_knn.cpp`：划分训练集和测试集 → 每行一个样本 → KNN → 准确率 |
| 数独图像识别 | `ch17_sudoku.cpp`：`locateDigits` → `recognizeBoard` → `solveSudoku` → `drawSolution` |
| 数独约束求解 | `sudoku_solver.cpp`：初始化行/列/宫约束 → 选择候选最少的格 → 尝试填数 → 失败回退 |
| HOG + SVM | `ch18_svm.cpp`：去倾斜 → 4 个区域的梯度直方图 → SVM |
| DNN 输入预处理 | `ch23_blobs.cpp`：逐项演示缩放、尺寸、均值、通道交换和裁剪 |
| DNN 应用 | `ch24_dnn.cpp`：章节入口只分派，各模型各自完成预处理、推理和结果解析 |
| MCC 指纹示例 | `ch14_mcc.cpp`：按注释的 6 个阶段阅读；`computeLocalDescriptors` 单独计算空间描述子 |

第 17 章求解器不依赖 OCR。可以给 `solveSudoku` 传入任意 9×9、`CV_32S` 类型的盘面：`0` 为空格，`1～9` 为线索。有解则原地填好，无解或线索冲突则返回 `false`，并保留输入盘面。矩阵形状或类型不符合约定时抛出异常。

## 关键数据约定

- OpenCV 彩色图按 **BGR** 排列；`blobFromImage` 的 `swapRB` 由模型决定，不能统一设为 `true`。
- 分类器通常接收二维 `CV_32F` 矩阵：每行一个样本，每列一个特征。`include/learning.hpp` 统一展平操作和准确率输出。
- DNN 的四维输出常用 **NCHW**，即批次、通道、行、列；SSD 每条检测结果有 7 个值，具体布局写在解析代码旁。
- `Mat` 赋值和 ROI 默认共享内存；需要独立图像时用 `clone()`。MCC 局部滤波特别需要克隆 ROI，才能与原 NumPy 示例的边界处理一致。
- NumPy `uint8` 加法会回绕；OpenCV `add` 会饱和。本工程用 `wrapAdd` 明确区分两种行为。

## 可重复验证

`build.cmd` 编译并执行 CTest。`verify.cmd` 在新的结果目录中运行基础入口、可用模型、离线回放及错误路径检查，避免旧图片让失败的运行看似成功。它保存逐项日志和 JSON 报告；失败返回非零退出码，缺模型明确记为 `SKIP`。

摄像头流程可用 `--frames DIR` 输入一组按文件名排序的图片。例如：

```powershell
.\build\cv40.exe --run 28.1 --models models --headless --frames my-frames --max-frames 10
```

程序使用与摄像头相同的逐帧处理函数。回放可以验证已有图像上的处理逻辑，但不能验证摄像头驱动、实时帧率，以及疲劳检测的完整时间序列行为。

`scripts/compare_python.py` 是可选的独立对照工具：执行原 Python 示例、关闭其窗口和图片写入，将数组结果与 C++ 输出逐像素比较，超出容差就失败。详见 [验证记录](VALIDATION.md)。C++ 构建和运行不依赖这个脚本或 Python。

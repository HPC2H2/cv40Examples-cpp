# 计算机视觉 40 例：C++17 版

根据李立宗《计算机视觉40例从入门到深度学习（OpenCV-Python）》配套 [Python 仓库](https://github.com/ZhangXinNan/cv40examples) 改写，原 Python 文件保持不变。本工程不调用 Python 解释器。

采用 **C++17**，显式设置 `CMAKE_CXX_STANDARD 17`、`CMAKE_CXX_STANDARD_REQUIRED ON`，关闭编译器语言扩展。使用 `std::filesystem` 管理素材路径、标准容器和 OpenCV 矩阵代替 NumPy，OpenCV 窗口/绘图代替 Matplotlib，欧氏距离直接计算，数独使用原生回溯求解。

## 构建

- Windows：Visual Studio 的“使用 C++ 的桌面开发”组件、CMake ≥ 3.24、Ninja。
- 固定依赖：OpenCV **4.13.0** + 同版本 opencv_contrib、dlib **20.0.0**。
- 默认首次配置自动从官方 GitHub 下载并编译依赖，需要联网；不修改系统 Python 环境。已在本机准备的源码缓存位于 `.deps/`。
- 运行采用 CPU，不需要 CUDA。编译时建议内存 16 GB 以上。

在 PowerShell 进入本目录后执行：

```powershell
cd E:\cv40Examples-cpp
.\build.cmd
```

`.cmd` 不受 PowerShell 的 `.ps1` 执行策略限制。初次编译依赖耗时较长，后续为增量编译。

也可在已经配置编译器环境的终端手动构建：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure
```

已有完整依赖时，可以通过 `-DCV40_FETCH_DEPS=OFF`、`-DOpenCV_DIR=...`、`-Ddlib_DIR=...` 指定。OpenCV 必须包括 `face`、`ximgproc` 贡献模块。

## 运行

```powershell
.\build\cv40.exe --list
.\build\cv40.exe --run 4.1
.\build\cv40.exe --run 13.3 --headless
.\build\cv40.exe --run 17.6 --headless
.\build\cv40.exe --run 14.0 --headless
.\build\cv40.exe --run 27.1
.\build\cv40.exe --run 24.3 --models models
.\build\cv40.exe --run 27.4 --models models
.\build\cv40.exe --all-safe
```

`--run` 后面是教材例号。[完整对照表](docs/EXAMPLES.md) 覆盖原仓库 184 个文件，参考版本也有独立 ID。源文件按章节组织，共用步骤提取为函数；数独求解与图像识别分离，各 DNN 应用有独立函数。建议从 [代码阅读指南](docs/READING_GUIDE.md) 开始。

默认打开 OpenCV 窗口，静态示例按任意键关闭；摄像头示例按 Esc 退出。`--headless` 不打开窗口，将中间结果保存到当前目录的 `output/<例号>/`，适合逐步查看算法效果。原示例的 `imwrite` 输出也统一写入此处，不覆盖素材。

常用选项：

| 选项 | 含义 |
|---|---|
| `--data DIR` | 指定素材根目录，内部是 `3OpenCV` 等章节文件夹；默认本工程 `data/` |
| `--models DIR` | 优先从此目录查找外部模型；保留模型表中的相对路径 |
| `--output DIR` | 指定结果目录 |
| `--input 85,2` | 提供基础章节的数值输入；无窗口批量运行使用预设输入 |
| `--seed 42` | 固定 OpenCV 随机种子，便于复查 |
| `--camera 0` | 选择摄像头 |
| `--frames DIR` | 按文件名排序读取图片，离线运行摄像头处理流程；与 `--camera` 互斥 |
| `--max-frames 30` | 限定摄像头帧数；无窗口摄像头运行必须指定 |
| `--all-safe` | 批量运行无需摄像头、无需外部权重的示例，失败返回非零状态 |
| `--self-test` | 检查整数运算、水印、哈希、轮廓、数独约束与回溯恢复、中文路径及回放行为 |

`data/` 复制了原仓库现有素材；教材 PDF 和原 Python 源码没有复制进来。图片使用二进制文件读取 + `imdecode`，支持 Windows 中文素材路径。

## 模型与验证范围

深度学习权重与 Python/C++ 语言无关；C++ 可以加载同一份 `.caffemodel`、`.weights`、`.pb`、`.t7`。本机已在 `models/` 额外准备 MobileNet-SSD、dlib 68 点关键点和 dlib CNN 检测模型，使用时传 `--models models`。其他缺失模型详见 [模型清单](docs/MODELS.md)。替换成 ONNX 模型通常要同时调整预处理、输出解析与类别表，不能只改扩展名。

OpenCV HOG 行人检测与 dlib HOG 人脸检测自带检测器；Haar 级联 XML 已随素材提供；第 26 章 LBPH/EigenFaces/FisherFaces 使用现有图片现场训练。这些例子不需要另下载深度学习权重。

本机 Release 构建和 CTest 自检通过，164 个基础入口 + 7 个模型入口实际运行通过；6 个摄像头入口还通过了离线图片回放检查。摄像头硬件与实时行为未验证，另有 5 个静态深度学习入口仍缺权重。实际检查结果及与原 Python 的数值对照见 [验证记录](docs/VALIDATION.md)。

重新验证：

```powershell
.\build.cmd
.\verify.cmd
```

`verify.cmd` 不需要 Python，使用本次 PowerShell 进程的执行策略设置，不修改系统策略。每次创建新的 `output/verification-日期-编号/`，保存逐项日志、图片和 `report.json`；最近一次结果索引为 `output/latest-verification.json`。缺模型会明确记录为 `SKIP`，检查失败则返回非零退出码。

## 有意保留及修正

- NumPy 的 `uint8` 加法回绕与 `cv::add` 的饱和行为分别实现，保留教材对比。
- 图像搜索按目录实际类别标记，修正用“文件位置除以 10”推算标签的脆弱做法。
- 输入为空、找不到轮廓/模型、零面积除法、摄像头读取失败均提供检查。
- 车牌模板包含“台”，不再遗漏字典最后一项；参考版本保留首位汉字、次位字母约束。
- 数独求解检查初始冲突，避免把错误识别出的盘面当成合法解。
- 第 28.4 例对每个检测到的人脸裁剪后再分类；原程序给每张脸都传整帧。
- 换脸的颜色除法增加零值保护；变换输出预先清零，避免未初始化边界。
- MCC 附带的 `sample_1_2.npz` 已转换为两个 CSV，保留双精度数值；运行不依赖 NumPy 或 pickle。CSV 分别为 35×4 的细节点与 35×208 的特征。
- MCC 的第二张指纹 `sample_1_2.png` 不在上游素材中，匹配仍使用真实 CSV 特征，右侧明确显示细节点示意图。可补充原图片后显示纹理。
- MCC 的局部滤波使用独立 ROI，避免 C++ `Mat` 在边缘读取母图像素，与 Python/NumPy 的处理范围保持一致。
- OpenCV 4.13 的 HOG 并行实现分别汇总检测框和权重，MeanShift 结果可能随线程调度变化。本例在检测时临时使用单线程，之后恢复线程设置；验证脚本还检查重复运行的一致性。
- 兼容上游旧格式 `opencv-haar-classifier` XML。MobileNet-SSD 额外下载的权重配套官方 `deploy.prototxt`，不与原素材中另一版配置混用。
- 绘图使用 OpenCV，文字标注以 ASCII 为主；中文识别结果在 UTF-8 控制台输出。

## 来源

原示例和素材归各自权利人；本次是学习用的语言移植，没有为原资源重新授予许可证。请在再分发前核对上游授权。MCC 参考程序源自 Raffaele Cappelli 在 WSB 2021 的课程，教材作者将其整理为 Python 文件，详见 [来源说明](docs/ATTRIBUTION.md)。

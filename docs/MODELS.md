# 外部模型

权重文件保存训练结果，不存在通过把 Python 换为 C++ 就能免除权重的替代方式。本工程使用 OpenCV DNN 和 dlib 的原生 C++ API。以下模型不在原仓库中；本次已额外下载其中 3 个模型用于验证，其他模型未补齐时程序报出明确的缺失路径。

本机 `models/` 已准备：

- `MobileNetSSD_deploy.caffemodel` 和配套 `MobileNetSSD_deploy.prototxt.txt`，可运行 24.3。
- `shape_predictor_68_face_landmarks.dat`，可运行 27.2–27.4、28.1–28.3。
- `mmod_human_face_detector.dat`，可运行 27.5。

使用示例：`build\cv40.exe --run 24.3 --models models --headless`。下载来源与校验值见 [下载记录](MODEL_DOWNLOADS.md)。模型文件在 `.gitignore` 中单独忽略。

**MobileNet-SSD 必须使用成对的配置和权重**：本次下载的 `mobilenet_iter_73000.caffemodel` 对应模型作者的 `deploy.prototxt`，含 BatchNorm 层。上游教材附带的配置是另一种部署结构，不能直接搭配此次下载的权重。`--models models` 会同时选用已准备好的配套文件。

把文件放到对应例子的素材目录，或使用 `--models E:\cv40-models`。后者在该目录下按下表“相对路径”查找。例如 `24.1` 对应 `E:\cv40-models\model\bvlc_googlenet.caffemodel`；`27.2` 对应 `E:\cv40-models\shape_predictor_68_face_landmarks.dat`。

| 例号 | 上游缺少的模型相对路径 | 用途 / 读取器 |
|---|---|---|
| 24.1 | `model/bvlc_googlenet.caffemodel` | GoogLeNet 分类 / Caffe |
| 24.2 | `yolov3.weights` | YOLOv3 检测 / Darknet |
| 24.3 | `MobileNetSSD_deploy.caffemodel` | MobileNet SSD 检测 / Caffe |
| 24.4 | `fcn8s-heavy-pascal.caffemodel` | FCN-8s 语义分割 / Caffe |
| 24.5 | `dnn/frozen_inference_graph.pb` | Mask R-CNN Inception v2 COCO 2018-01-28 / TensorFlow |
| 24.6 | `model/eccv16/starry_night.t7` | 风格迁移 / Torch7 |
| 24.7 | `graph_opt.pb` | OpenPose MobileNet 人体姿态 / TensorFlow |
| 27.2–27.4、28.1–28.3 | `shape_predictor_68_face_landmarks.dat` | 68 点人脸关键点 / dlib |
| 27.5 | `mmod_human_face_detector.dat` | CNN 人脸检测 / dlib |
| 28.4 | `model/opencv_face_detector_uint8.pb` | 人脸检测 / TensorFlow |
| 28.4 | `model/age_net.caffemodel`、`model/gender_net.caffemodel` | 年龄与性别示例 / Caffe |

教材配置 `.prototxt` / `.cfg` / `.pbtxt` 和类别文本已复制到 `data/` 中。使用外部权重时必须核对相应版本；配套的新配置放入 `--models` 指定目录，可以覆盖原配置而不改动素材。

若只想运行不额外下载模型的示例，可使用 `--all-safe`。其中已有 HOG 行人检测、Haar/dlib HOG 人脸检测、KNN/SVM 数字识别、LBPH/EigenFaces/FisherFaces 等；这些是独立的教材算法，不代替第 24 章的深度学习实验。

换用 ONNX 的步骤是：确定任务及训练类别，使用匹配模型的 `readNetFromONNX`，核对颜色顺序、输入尺寸、归一化、张量布局和输出结构，最后用已知图片验证。分类模型不能直接替换检测或分割模型，dlib 的 68 点模型也不能直接换成任意 ONNX 文件。

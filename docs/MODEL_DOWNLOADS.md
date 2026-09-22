# 本次额外下载的模型

下载日期：2026-09-22。文件位于本机 `models/`，无需放入原 Python 仓库。模型是数据文件，Python/C++ 可共用。

## dlib

来源：[dlib 作者的模型仓库](https://github.com/davisking/dlib-models)。下载 `.bz2` 后解压得到 `.dat`。

- [68 点关键点模型](https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2)
- [CNN 人脸检测模型](https://raw.githubusercontent.com/davisking/dlib-models/master/mmod_human_face_detector.dat.bz2)

## MobileNet-SSD

来源：[模型作者的 Caffe 实现](https://github.com/chuanqi305/MobileNet-SSD)。必须同时使用这里的配置和权重。

- [mobilenet_iter_73000.caffemodel](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel)，本机命名为 `MobileNetSSD_deploy.caffemodel`。
- [deploy.prototxt](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt)，本机命名为 `MobileNetSSD_deploy.prototxt.txt`。

## 本机文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `mmod_human_face_detector.dat` | `be467b1a76f482693de3b0f6a1ff91d092319be71523d4d4b0628f6a53fcb87a` |
| `shape_predictor_68_face_landmarks.dat` | `fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f` |
| `MobileNetSSD_deploy.caffemodel` | `52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7` |
| `MobileNetSSD_deploy.prototxt.txt` | `2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608` |

下载链接指向上游分支；重新下载时可用上述校验值确认是否与本次验证使用的文件一致。模型来源及各自使用条件见上游说明。

# 来源与改写说明

- 教材：李立宗，《计算机视觉40例从入门到深度学习（OpenCV-Python）》，电子工业出版社。
- 配套代码：<https://github.com/ZhangXinNan/cv40examples>。原文件署名李立宗，邮箱 `lilizong@gmail.com`。本项目按该代码的章节和例号移植为 C++17。
- MCC 参考程序：Raffaele Cappelli，WSB 2021，Hands on Fingerprint Recognition with OpenCV and Python。上游 Python 文件及教材 14.2.4 节给出的来源：<https://www.comp.hkbu.edu.hk/wsb2021/lecturer_details.php?lect_id=2>。
- C++ CNN 人脸网络类型与 dlib 官方 `examples/dnn_mmod_face_detection_ex.cpp` 一致，以兼容原 `.dat` 模型。
- `data/` 中图片、标签、级联 XML 等来自配套仓库；新增两个 MCC CSV 是原 `.npz` 数组的数值导出。原始 `.npz` 保留供核对。
- OpenCV、opencv_contrib、dlib 的源代码和许可证位于各自 `.deps/*-src/` 中；构建配置下载官方版本。不改变这些依赖的授权条款。

参考过用户提供的教材第 14 章 MCC 与 SIFT 的算法说明。教材 PDF 没有收录进本工程。

#pragma once
#include "cv40.hpp"
namespace cv40 {
inline std::vector<uchar> bytes(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f), "Cannot read model: " + p.u8string());
    return std::vector<uchar>((std::istreambuf_iterator<char>(f)), {});
}
inline dnn::Net caffe(Context &c, const std::string &config, const std::string &weights) {
    auto model = bytes(c.file(weights)), proto = bytes(c.file(config));
    return dnn::readNetFromCaffe(proto, model);
}
inline dnn::Net tensorflow(Context &c, const std::string &weights, const std::string &config = "") {
    auto model = bytes(c.file(weights));
    auto proto = config.empty() ? std::vector<uchar>{} : bytes(c.file(config));
    return dnn::readNetFromTensorflow(model, proto);
}
inline std::vector<std::string> lines(Context &c, const std::string &name) {
    std::ifstream f(c.file(name));
    std::vector<std::string> out;
    std::string s;
    while (std::getline(f, s)) {
        if (!s.empty() && s.back() == '\r') {
            s.pop_back();
        }
        if (!s.empty()) {
            out.push_back(s);
        }
    }
    return out;
}
inline std::string className(const std::vector<std::string> &names, int i) {
    return i >= 0 && i < int(names.size()) ? names[i] : "class_" + std::to_string(i);
}
inline int argmax(const Mat &m) {
    Mat flat = m.reshape(1, 1);
    Point index;
    minMaxLoc(flat, nullptr, nullptr, nullptr, &index);
    return index.x;
}
inline Rect detectionBox(const float *p, Size size) {
    int x1 = int(p[0] * size.width), y1 = int(p[1] * size.height), x2 = int(p[2] * size.width),
        y2 = int(p[3] * size.height);
    return Rect(Point(std::min(x1, x2), std::min(y1, y2)), Point(std::max(x1, x2), std::max(y1, y2))) &
           Rect({}, size);
}
} // namespace cv40

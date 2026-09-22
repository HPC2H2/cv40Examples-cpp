#pragma once
#include "cv40.hpp"

namespace cv40 {
// OpenCV 的分类器每行接收一个样本，特征统一使用 CV_32F。
inline Mat flattenToFloat(const Mat &image) {
    Mat features;
    image.reshape(1, 1).convertTo(features, CV_32F);
    return features;
}
inline void printAccuracy(const Mat &results, const Mat &labels) {
    Mat target;
    labels.convertTo(target, results.type());
    require(results.size() == target.size() && !results.empty(), "Invalid evaluation data");
    std::cout << "accuracy=" << 100. * countNonZero(results == target) / results.total() << "% ("
              << results.total() << " samples)\n";
}
} // namespace cv40

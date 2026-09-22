// C++17 adaptation of examples by 李立宗 (计算机视觉40例).
#pragma once
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cv40 {
namespace fs = std::filesystem;
using namespace cv;
using Contour = std::vector<Point>;
using Contours = std::vector<Contour>;
struct Example {
    const char *id;
    const char *source;
    int chapter;
    int number;
    int variant;
    bool camera;
    bool model;
};
extern const std::vector<Example> examples;
struct Context {
    fs::path root;   // 素材根目录，内部按教材章节分文件夹。
    fs::path models; // 外部权重目录；查找时优先于素材目录。
    fs::path output; // 所有生成文件的根目录，避免覆盖原素材。
    fs::path frames; // 可选：按文件名排序的离线帧，用于复现摄像头处理流程。
    const Example *example = nullptr;
    bool headless = false;
    int camera = 0;
    int maxFrames = 0; // 0 表示不主动限制；无窗口摄像头运行必须设置上限。
    int imageNumber = 0;
    std::vector<double> input;
    std::size_t inputIndex = 0;
    fs::path chapterDir() const;
    fs::path file(const std::string &name) const;
    Mat read(const std::string &name, int mode = IMREAD_COLOR) const;
    void show(const std::string &name, const Mat &image);
    void save(const std::string &name, const Mat &image) const;
    double ask(const std::string &prompt, double fallback);
    void video(const std::function<void(Mat &, int)> &process);
};
Mat readImage(const fs::path &path, int mode = IMREAD_COLOR);
void writeImage(const fs::path &path, const Mat &image);
std::vector<fs::path> imageFiles(const fs::path &dir);
Mat gray(const Mat &image);
Mat binary(const Mat &image, double threshold = 127, int type = THRESH_BINARY);
Contours contours(const Mat &image, int mode = RETR_EXTERNAL);
Contour largest(const Contours &all);
Mat masked(const Mat &image, const Mat &mask);
Mat wrapAdd(const Mat &a, const Mat &b);
Mat hashImage(const Mat &image);
int hamming(const Mat &a, const Mat &b);
Mat montage(const std::vector<Mat> &images, int width = 240);
Mat perspective(const Mat &image, std::vector<Point2f> points);
void label(Mat &image, const std::string &text, Point at = {10, 30}, Scalar color = {0, 0, 255},
           double scale = .7);
void require(bool condition, const std::string &message);
void run(Context &c);
void selfTest();
bool solveSudoku(Mat &board);
void chapter2(Context &, int, int);
void chapter3(Context &, int, int);
void chapter4(Context &, int, int);
void chapter5(Context &, int, int);
void chapter6(Context &, int, int);
void chapter7(Context &, int, int);
void chapter8(Context &, int, int);
void chapter9(Context &, int, int);
void chapter10(Context &, int, int);
void chapter11(Context &, int, int);
void chapter12(Context &, int, int);
void chapter13(Context &, int, int);
void chapter14(Context &, int, int);
void chapter15(Context &, int, int);
void chapter16(Context &, int, int);
void chapter17(Context &, int, int);
void chapter18(Context &, int, int);
void chapter19(Context &, int, int);
void chapter20(Context &, int, int);
void chapter23(Context &, int, int);
void chapter24(Context &, int, int);
void chapter25(Context &, int, int);
void chapter26(Context &, int, int);
void chapter27(Context &, int, int);
void chapter28(Context &, int, int);
} // namespace cv40

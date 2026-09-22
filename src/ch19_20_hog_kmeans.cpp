#include "cv40.hpp"
namespace cv40 {
namespace {
class ScopedSerialHog {
  public:
    ScopedSerialHog() : previousThreads_(getNumThreads()) {
        // OpenCV 4.13 HOGInvoker 分别加锁追加框和权重；并行时两者可能错位。
        // 此示例用串行检测保持对应关系，离开作用域后恢复其他算法的线程设置。
        setNumThreads(1);
    }
    ~ScopedSerialHog() {
        setNumThreads(previousThreads_);
    }

  private:
    int previousThreads_;
};
} // namespace

static std::vector<Rect> exampleNms(std::vector<Rect> boxes, double threshold) {
    std::sort(boxes.begin(), boxes.end(), [](auto a, auto b) { return a.br().y < b.br().y; });
    std::vector<Rect> result;
    while (!boxes.empty()) {
        Rect current = boxes.back();
        boxes.pop_back();
        result.push_back(current);
        boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                                   [&](Rect other) {
                                       int w = std::max(0, std::min(current.br().x, other.br().x) -
                                                               std::max(current.x, other.x) + 1),
                                           h = std::max(0, std::min(current.br().y, other.br().y) -
                                                               std::max(current.y, other.y) + 1);
                                       return other.area() > 0 && double(w * h) / other.area() > threshold;
                                   }),
                    boxes.end());
    }
    return result;
}
void chapter19(Context &c, int n, int variant) {
    Mat a = c.read(n == 0 ? "nms.jpg" : n == 3 ? "backPadding2.jpg" : "back.jpg");
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    auto detect = [&](Size stride, Size padding, double scale, bool meanshift, const std::string &name) {
        Mat r = a.clone();
        std::vector<Rect> boxes;
        std::vector<double> weights;
        int64 start = getTickCount();
        {
            ScopedSerialHog serialDetection;
            hog.detectMultiScale(a, boxes, weights, 0, stride, padding, scale, 2, meanshift);
        }
        std::cout << name << ": boxes=" << boxes.size()
                  << " seconds=" << double(getTickCount() - start) / getTickFrequency() << '\n';
        for (auto b : boxes) {
            std::cout << name << ": rect=" << b.x << ',' << b.y << ',' << b.width << ',' << b.height << '\n';
            rectangle(r, b, {0, 0, 255}, 2);
        }
        c.show(name, r);
        return boxes;
    };
    if (n == 0 && variant == 0) {
        auto boxes = detect({4, 4}, {8, 8}, 1.05, false, "original");
        auto kept = exampleNms(boxes, .5);
        for (auto b : kept) {
            rectangle(a, b, {0, 0, 255}, 2);
        }
        c.show("NMS", a);
    } else if (n == 0) {
        detect({}, {}, 1.05, false, "without_meanshift");
        detect({}, {}, 1.05, true, "meanshift");
    } else if (n == 1) {
        detect({}, {}, 1.05, false, "result");
    } else if (n == 2) {
        for (int s : {4, 12, 24}) {
            detect({s, s}, {}, 1.05, false, "stride" + std::to_string(s));
        }
    } else if (n == 3) {
        for (int p : {0, 8}) {
            detect({16, 16}, {p, p}, 1.05, false, "padding" + std::to_string(p));
        }
    } else if (n == 4) {
        for (double s : {1.01, 1.05, 1.3}) {
            detect({}, {}, s, false, "scale" + std::to_string(s));
        }
    } else if (n == 5) {
        detect({}, {}, 1.01, false, "without_meanshift");
        detect({}, {}, 1.01, true, "meanshift");
    } else {
        detect({8, 8}, {2, 2}, 1.03, true, "result");
    }
}
void chapter20(Context &c, int n, int) {
    Mat data, a;
    if (n == 1) {
        data = Mat(50, 2, CV_32F);
        for (int y = 0; y < 50; ++y) {
            for (int x = 0; x < 2; ++x) {
                data.at<float>(y, x) = float(theRNG().uniform(0, 100));
            }
        }
    } else {
        a = c.read("cat.jpg");
        a.reshape(1, int(a.total())).convertTo(data, CV_32F);
    }
    Mat labels, centers;
    double distance =
        kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 10, 1.), 10,
               KMEANS_RANDOM_CENTERS, centers);
    std::cout << "compactness=" << distance << "\ncenters=" << centers << '\n';
    if (n == 1) {
        Mat plot(540, 540, CV_8UC3, Scalar::all(255));
        rectangle(plot, {20, 20, 500, 500}, Scalar::all(0));
        for (int i = 0; i < data.rows; ++i) {
            Point p(20 + int(data.at<float>(i, 0) * 5), 520 - int(data.at<float>(i, 1) * 5));
            if (labels.at<int>(i) == 0) {
                rectangle(plot, {p.x - 3, p.y - 3, 7, 7}, {0, 180, 0}, FILLED);
            } else {
                circle(plot, p, 4, {0, 0, 255}, FILLED);
            }
        }
        for (int i = 0; i < 2; ++i) {
            Point p(20 + int(centers.at<float>(i, 0) * 5), 520 - int(centers.at<float>(i, 1) * 5));
            drawMarker(plot, p, {255, 0, 0}, MARKER_CROSS, 20, 3);
        }
        c.show("clusters", plot);
    } else {
        Mat result(a.size(), a.type());
        for (int i = 0; i < int(a.total()); ++i) {
            int k = labels.at<int>(i);
            auto &pixel = result.at<Vec3b>(i / a.cols, i % a.cols);
            for (int j = 0; j < 3; ++j) {
                pixel[j] = static_cast<uchar>(centers.at<float>(k, j));
            }
        }
        c.show("original", a);
        c.show("art", result);
    }
}
} // namespace cv40

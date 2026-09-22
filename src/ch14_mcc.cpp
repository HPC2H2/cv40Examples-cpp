// Native translation of the supplied Cappelli WSB 2021 MCC teaching example.
// This is the example's spatial descriptor, not a full production MCC system.
#include "cv40.hpp"
namespace cv40 {
struct Minutia {
    int x, y;
    bool termination;
    double angle;
};
static Mat drawMinutiae(const Mat &image, const std::vector<Minutia> &points, bool directions) {
    Mat out;
    cvtColor(image, out, COLOR_GRAY2BGR);
    for (auto p : points) {
        Scalar color = p.termination ? Scalar(255, 0, 0) : Scalar(0, 0, 255);
        Point pt(p.x, p.y);
        if (directions) {
            circle(out, pt, 3, color, 1, LINE_AA);
            line(out, pt, pt + Point(cvRound(std::cos(p.angle) * 7), -cvRound(std::sin(p.angle) * 7)), color,
                 1, LINE_AA);
        } else {
            drawMarker(out, pt, color, MARKER_CROSS, 8);
        }
    }
    return out;
}
static Mat csvMatrix(const fs::path &path) {
    std::ifstream f(path);
    require(bool(f), "Cannot open MCC reference: " + path.u8string());
    Mat out;
    std::string line;
    while (std::getline(f, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        std::vector<double> values;
        double x;
        while (ss >> x) {
            values.push_back(x);
        }
        if (values.empty()) {
            continue;
        }
        Mat row(1, int(values.size()), CV_64F, values.data());
        if (!out.empty()) {
            require(out.cols == row.cols, "Ragged MCC CSV");
        }
        out.push_back(row.clone());
    }
    return out;
}
static Mat computeLocalDescriptors(const std::vector<Minutia> &minutiae) {
    // 半径 70 的圆内采样 16×16 网格中的 208 个点；坐标随细节点方向旋转。
    std::vector<Point2d> cells;
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            Point2d p((x - 8) * 8.75 + 4.375, (y - 8) * 8.75 + 4.375);
            if (p.dot(p) <= 70 * 70) {
                cells.push_back(p);
            }
        }
    }
    Mat structures(int(minutiae.size()), int(cells.size()), CV_64F);
    for (std::size_t i = 0; i < minutiae.size(); ++i) {
        auto p = minutiae[i];
        double cosine = std::cos(p.angle), sine = std::sin(p.angle);
        for (std::size_t j = 0; j < cells.size(); ++j) {
            auto cell = cells[j];
            Point2d q(cosine * cell.x + sine * cell.y + p.x, -sine * cell.x + cosine * cell.y + p.y);
            double contribution = 0;
            for (std::size_t k = 0; k < minutiae.size(); ++k) {
                if (k != i) {
                    Point2d delta = q - Point2d(minutiae[k].x, minutiae[k].y);
                    contribution += std::exp(-.5 * delta.dot(delta) / 49) / (std::sqrt(2 * CV_PI) * 7);
                }
            }
            structures.at<double>(int(i), int(j)) = 1 / (1 + std::exp(-400 * (contribution - .01)));
        }
    }
    return structures;
}

void mccDemo(Context &context) {
    // 1. 梯度能量分割指纹前景，并估计局部脊线方向。
    Mat fingerprint = context.read("sample_1_1.png", IMREAD_GRAYSCALE), gx, gy, gradientSquaredX,
        gradientSquaredY, gradientMagnitude, localMagnitude, mask;
    context.show("original", fingerprint);
    Sobel(fingerprint, gx, CV_32F, 1, 0);
    Sobel(fingerprint, gy, CV_32F, 0, 1);
    gradientSquaredX = gx.mul(gx);
    gradientSquaredY = gy.mul(gy);
    magnitude(gx, gy, gradientMagnitude);
    boxFilter(gradientMagnitude, localMagnitude, -1, {25, 25}, {-1, -1}, false);
    double max;
    minMaxLoc(localMagnitude, nullptr, &max);
    threshold(localMagnitude, mask, max * .2, 255, THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    context.show("mask", mask);
    Mat xx, yy, xy, ridgeAngle, strength, numerator, denominator;
    boxFilter(gradientSquaredX, xx, -1, {23, 23}, {-1, -1}, false);
    boxFilter(gradientSquaredY, yy, -1, {23, 23}, {-1, -1}, false);
    boxFilter(gx.mul(gy), xy, -1, {23, 23}, {-1, -1}, false);
    phase(xx - yy, -2 * xy, ridgeAngle);
    ridgeAngle = (ridgeAngle + CV_PI) / 2;
    denominator = xx + yy;
    magnitude(xx - yy, 2 * xy, numerator);
    divide(numerator, denominator, strength);
    strength.setTo(0, denominator == 0);
    Mat orientation;
    cvtColor(fingerprint, orientation, COLOR_GRAY2BGR);
    for (int y = 0; y < fingerprint.rows; y += 16) {
        for (int x = 0; x < fingerprint.cols; x += 16) {
            if (mask.at<uchar>(y, x)) {
                double t = ridgeAngle.at<float>(y, x), s = strength.at<float>(y, x) * 8;
                Point d(cvRound(std::cos(t) * s), -cvRound(std::sin(t) * s)), p(x + 1, y + 1);
                line(orientation, p - d, p + d, {255, 0, 0}, 1, LINE_AA);
            }
        }
    }
    context.show("orientations", orientation);
    // 2. 从教材指定区域的灰度投影峰值间距估计脊线周期。
    require(fingerprint.cols >= 130 && fingerprint.rows >= 90,
            "Fingerprint too small for ridge frequency ROI");
    // A NumPy ROI passed to cv2 is an isolated image. Clone the Mat ROI so blur
    // does not sample neighboring pixels from the parent fingerprint.
    Mat region = fingerprint(Rect(80, 10, 50, 80)).clone(), smoothed, signature;
    blur(region, smoothed, {5, 5});
    reduce(smoothed, signature, 1, REDUCE_SUM, CV_64F);
    std::vector<int> peaks;
    for (int i = 1; i < signature.rows - 1; ++i) {
        if (signature.at<double>(i) > signature.at<double>(i - 1) &&
            signature.at<double>(i) >= signature.at<double>(i + 1)) {
            peaks.push_back(i);
        }
    }
    require(peaks.size() >= 2, "Cannot estimate fingerprint ridge period");
    double ridgePeriod = double(peaks.back() - peaks.front()) / (peaks.size() - 1);
    std::cout << "ridge period=" << ridgePeriod << '\n';
    Mat chart(320, 640, CV_8UC3, Scalar::all(255));
    double top;
    minMaxLoc(signature, nullptr, &top);
    for (int i = 1; i < signature.rows; ++i) {
        line(chart, {(i - 1) * 8, 300 - int(signature.at<double>(i - 1) / top * 280)},
             {i * 8, 300 - int(signature.at<double>(i) / top * 280)}, {255, 0, 0}, 2);
    }
    for (int i : peaks) {
        line(chart, {i * 8, 0}, {i * 8, 319}, {180, 180, 180});
    }
    context.show("ridge_signature", chart);
    context.show("frequency_region", region);
    std::vector<Mat> filtered;
    int kernelSize = cvRound(ridgePeriod * 2 + 1);
    if (kernelSize % 2 == 0) {
        ++kernelSize;
    }
    for (int i = 0; i < 8; ++i) {
        Mat k = getGaborKernel({kernelSize, kernelSize}, 1.5 / std::sqrt(6 * std::log(10.)) * ridgePeriod,
                               CV_PI / 2 - i * CV_PI / 8, ridgePeriod, 1, 0, CV_64F);
        k /= sum(k)[0];
        k -= mean(k)[0];
        Mat f;
        filter2D(255 - fingerprint, f, CV_32F, k);
        filtered.push_back(f);
        Mat vis;
        normalize(k, vis, 0, 1, NORM_MINMAX);
        context.show("gabor" + std::to_string(i), vis);
    }
    // 3. 按局部方向选择 Gabor 滤波响应，增强脊线。
    Mat enhanced(fingerprint.size(), CV_8U);
    for (int y = 0; y < fingerprint.rows; ++y) {
        for (int x = 0; x < fingerprint.cols; ++x) {
            int i = cvRound(std::fmod(ridgeAngle.at<float>(y, x), CV_PI) / CV_PI * 8) % 8;
            enhanced.at<uchar>(y, x) =
                mask.at<uchar>(y, x) &
                static_cast<uchar>(std::clamp(filtered[i].at<float>(y, x), 0.f, 255.f));
        }
    }
    context.show("enhanced", enhanced);
    // 4. 二值化、细化；8 邻域的 0→1 跳变数区分端点（1）与分叉点（3）。
    Mat ridges = binary(enhanced, 32), skeleton;
    ximgproc::thinning(ridges, skeleton, ximgproc::THINNING_GUOHALL);
    context.show("ridges", ridges);
    context.show("skeleton", skeleton);
    const std::array<Point, 8> steps{{{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}}};
    Mat codes = Mat::zeros(fingerprint.size(), CV_8U), crossingNumber = Mat::zeros(fingerprint.size(), CV_8U);
    std::array<int, 256> crossings{};
    for (int code = 0; code < 256; ++code) {
        for (int d = 0; d < 8; ++d) {
            crossings[code] += !(code & (1 << d)) && (code & (1 << ((d + 1) % 8)));
        }
    }
    for (int y = 1; y < fingerprint.rows - 1; ++y) {
        for (int x = 1; x < fingerprint.cols - 1; ++x) {
            int code = 0;
            for (int d = 0; d < 8; ++d) {
                if (skeleton.at<uchar>(y + steps[d].y, x + steps[d].x)) {
                    code |= 1 << d;
                }
            }
            codes.at<uchar>(y, x) = uchar(code);
            if (skeleton.at<uchar>(y, x)) {
                crossingNumber.at<uchar>(y, x) = uchar(crossings[code]);
            }
        }
    }
    Mat padded, dist;
    copyMakeBorder(mask, padded, 1, 1, 1, 1, BORDER_CONSTANT);
    distanceTransform(padded, dist, DIST_C, 3);
    dist = dist(Rect(1, 1, fingerprint.cols, fingerprint.rows));
    std::vector<Minutia> all, interior, valid;
    for (int y = 1; y < fingerprint.rows - 1; ++y) {
        for (int x = 1; x < fingerprint.cols - 1; ++x) {
            if (crossingNumber.at<uchar>(y, x) == 1 || crossingNumber.at<uchar>(y, x) == 3) {
                Minutia p{x, y, crossingNumber.at<uchar>(y, x) == 1, 0};
                all.push_back(p);
                if (dist.at<float>(y, x) > 10) {
                    interior.push_back(p);
                }
            }
        }
    }
    context.show("minutiae", drawMinutiae(fingerprint, all, false));
    context.show("interior", drawMinutiae(skeleton, interior, false));
    // 5. 沿骨架追踪，剔除短伪脊线并估计细节点方向。
    auto nextDirections = [&](Point p, int previous) {
        std::vector<int> directions;
        if (p.x <= 0 || p.y <= 0 || p.x >= fingerprint.cols - 1 || p.y >= fingerprint.rows - 1) {
            return directions;
        }
        int code = codes.at<uchar>(p);
        for (int d = 0; d < 8; ++d) {
            if (code & (1 << d)) {
                directions.push_back(d);
            }
        }
        if (previous != 8) {
            std::stable_sort(directions.begin(), directions.end(), [&](int a, int b) {
                return 4 - std::abs(std::abs(a - previous) - 4) < 4 - std::abs(std::abs(b - previous) - 4);
            });
            if (!directions.empty() && directions.back() == (previous + 4) % 8) {
                directions.pop_back();
            }
        }
        return directions;
    };
    auto follow = [&](Point origin, int direction) -> std::optional<double> {
        Point p = origin;
        double length = 0;
        while (length < 20) {
            auto directions = nextDirections(p, direction);
            if (directions.empty()) {
                break;
            }
            bool stop = false;
            for (int d : directions) {
                Point q = p + steps[d];
                if (q.x < 0 || q.y < 0 || q.x >= fingerprint.cols || q.y >= fingerprint.rows ||
                    crossingNumber.at<uchar>(q) != 2) {
                    stop = true;
                    break;
                }
            }
            if (stop) {
                break;
            }
            direction = directions[0];
            p += steps[direction];
            length += direction % 2 == 0 ? std::sqrt(2.) : 1.;
        }
        if (length < 10) {
            return {};
        }
        return std::atan2(double(origin.y - p.y), double(p.x - origin.x));
    };
    for (auto p : interior) {
        std::optional<double> d;
        if (p.termination) {
            d = follow({p.x, p.y}, 8);
        } else {
            auto directions = nextDirections({p.x, p.y}, 8);
            if (directions.size() == 3) {
                std::array<std::optional<double>, 3> angles;
                for (int i = 0; i < 3; ++i) {
                    angles[i] = follow(Point(p.x, p.y) + steps[directions[i]], directions[i]);
                }
                if (angles[0] && angles[1] && angles[2]) {
                    double best = DBL_MAX;
                    for (int i = 0; i < 3; ++i) {
                        double a = *angles[i], b = *angles[(i + 1) % 3],
                               difference = CV_PI - std::abs(std::abs(a - b) - CV_PI);
                        if (difference < best) {
                            best = difference;
                            d = std::atan2((std::sin(a) + std::sin(b)) / 2, (std::cos(a) + std::cos(b)) / 2);
                        }
                    }
                }
            }
        }
        if (d) {
            p.angle = *d;
            valid.push_back(p);
        }
    }
    require(!valid.empty(), "No valid fingerprint minutiae");
    context.show("directed_minutiae", drawMinutiae(fingerprint, valid, true));
    // 6. 计算局部空间描述子，与附带的参考描述子进行距离匹配。
    Mat structures = computeLocalDescriptors(valid);
    Mat reference = csvMatrix(context.file("sample_1_2_arr_1.csv")),
        points = csvMatrix(context.file("sample_1_2_arr_0.csv"));
    Mat other(fingerprint.size(), CV_8UC1, Scalar(255));
    auto referenceImage = context.chapterDir() / "sample_1_2.png";
    if (!context.models.empty() && fs::is_regular_file(context.models / "sample_1_2.png")) {
        referenceImage = context.models / "sample_1_2.png";
    }
    bool hasReferenceImage = fs::is_regular_file(referenceImage);
    if (hasReferenceImage) {
        other = readImage(referenceImage, IMREAD_GRAYSCALE);
    } else {
        std::cout << "Reference image absent; plotting the supplied reference minutiae.\n";
    }
    require(reference.cols == structures.cols && reference.rows == points.rows,
            "Invalid MCC reference dimensions");
    struct Pair {
        double distance;
        int a, b;
    };
    std::vector<Pair> pairs;
    for (int i = 0; i < structures.rows; ++i) {
        for (int j = 0; j < reference.rows; ++j) {
            double denom = norm(structures.row(i)) + norm(reference.row(j));
            pairs.push_back({denom ? norm(structures.row(i) - reference.row(j)) / denom : 0, i, j});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](auto a, auto b) { return a.distance < b.distance; });
    require(pairs.size() >= 5, "Too few MCC pairs");
    double score = 0;
    Mat result(std::max(fingerprint.rows, other.rows), fingerprint.cols + other.cols, CV_8UC3,
               Scalar::all(255));
    drawMinutiae(fingerprint, valid, true).copyTo(result(Rect(0, 0, fingerprint.cols, fingerprint.rows)));
    std::vector<Minutia> referenceMinutiae;
    for (int i = 0; i < points.rows; ++i) {
        referenceMinutiae.push_back({int(points.at<double>(i, 0)), int(points.at<double>(i, 1)),
                                     points.at<double>(i, 2) != 0, points.at<double>(i, 3)});
    }
    Mat colored = drawMinutiae(other, referenceMinutiae, true);
    if (!hasReferenceImage) {
        label(colored, "Reference minutiae only", {5, 18}, Scalar(0, 0, 0), .4);
    }
    colored.copyTo(result(Rect(fingerprint.cols, 0, other.cols, other.rows)));
    for (int i = 0; i < 5; ++i) {
        auto p = pairs[i];
        score += p.distance;
        line(result, {valid[p.a].x, valid[p.a].y},
             {fingerprint.cols + int(points.at<double>(p.b, 0)), int(points.at<double>(p.b, 1))}, {0, 0, 255},
             1, LINE_AA);
    }
    std::cout << "minutiae=" << valid.size() << " descriptor=" << structures.size()
              << " comparison score=" << 1 - score / 5 << '\n';
    context.show("matched_pairs", result);
}
} // namespace cv40

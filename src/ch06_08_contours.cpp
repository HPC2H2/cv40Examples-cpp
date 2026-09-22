#include "cv40.hpp"
namespace cv40 {
void chapter6(Context &c, int n, int) {
    if (n == 4) {
        for (int type : {MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE}) {
            std::cout << getStructuringElement(type, {5, 5}) << '\n';
        }
        return;
    }
    if (n == 6) {
        std::vector<std::string> a{u8"刘能", u8"广坤", u8"赵四", u8"一水", u8"小萌"},
            b{"Python", "OpenCV", u8"计算机视觉", u8"机器学习", u8"深度学习"};
        for (int i = 0; i < 5; ++i) {
            std::cout << i << ' ' << a[i] << ' ' << b[i] << '\n';
        }
        return;
    }
    Mat a = c.read(n == 1   ? "cat3.jpg"
                   : n == 5 ? "kernel.bmp"
                   : n == 7 ? "lena.bmp"
                   : n == 8 ? "count.jpg"
                            : "opencv.png"),
        r;
    c.show("original", a);
    if (n == 5) {
        for (int type : {MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE}) {
            dilate(a, r, getStructuringElement(type, {59, 59}));
            c.show("dilate" + std::to_string(type), r);
        }
        return;
    }
    if (n == 7) {
        threshold(a, r, 127, 255, THRESH_BINARY);
        c.show("threshold", r);
        return;
    }
    r = binary(a, n == 8 ? 150 : 127, n == 8 ? THRESH_BINARY_INV : THRESH_BINARY);
    if (n == 8) {
        auto k = getStructuringElement(MORPH_ELLIPSE, {5, 5});
        erode(r, r, k, {-1, -1}, 4);
        dilate(r, r, k, {-1, -1}, 3);
        GaussianBlur(r, r, {3, 3}, 0);
        c.show("gaussian", r);
    }
    auto all = contours(r, n == 8 ? RETR_TREE : RETR_LIST);
    require(!all.empty(), "No object contours");
    Contours selected;
    for (std::size_t i = 0; i < all.size(); ++i) {
        double area = contourArea(all[i]);
        if (n == 2) {
            std::cout << i << " area=" << area << '\n';
        }
        if (n == 1 && i > 0) {
            break;
        }
        if (n == 3 && area <= 1000 || n == 8 && area <= 30) {
            continue;
        }
        selected.push_back(all[i]);
    }
    drawContours(a, selected, -1, n == 8 ? Scalar(0, 255, 0) : Scalar(0, 0, 255),
                 n == 3   ? 8
                 : n == 8 ? 1
                          : 3);
    if (n == 1 || n == 8) {
        for (std::size_t i = 0; i < selected.size(); ++i) {
            auto m = moments(selected[i]);
            if (m.m00 != 0) {
                label(a, n == 1 ? "cat" : std::to_string(i), {int(m.m10 / m.m00), int(m.m01 / m.m00)});
            }
        }
    }
    std::cout << "objects=" << selected.size() << '\n';
    c.show("result", a);
}
void chapter7(Context &c, int n, int) {
    if (n == 1) {
        for (const std::string name : {"opening.bmp", "opening2.bmp"}) {
            Mat a = c.read(name), r;
            morphologyEx(a, r, MORPH_OPEN, Mat::ones(10, 10, CV_8U));
            c.show(name, a);
            c.show(name + "_open", r);
        }
        return;
    }
    Mat a = c.read(n == 2 ? "coins.jpg" : n == 3 ? "cc.bmp" : "pill3.jpg"), r;
    c.show("original", a);
    if (n == 3) {
        auto all = contours(binary(a), RETR_LIST);
        require(!all.empty(), "No contour");
        Point2f center;
        float radius;
        minEnclosingCircle(all[0], center, radius);
        circle(a, center, int(radius), Scalar::all(255), 2);
        c.show("circle", a);
        return;
    }
    r = binary(a, 0, (n == 2 ? THRESH_BINARY_INV : THRESH_BINARY) | THRESH_OTSU);
    c.show("threshold", r);
    morphologyEx(r, r, MORPH_OPEN, getStructuringElement(n == 2 ? MORPH_RECT : MORPH_CROSS, {3, 3}), {-1, -1},
                 n == 2 ? 2 : 1);
    c.show("opening", r);
    distanceTransform(r, r, DIST_L2, n == 2 ? 5 : 3);
    double max;
    minMaxLoc(r, nullptr, &max);
    threshold(r, r, (n == 2 ? .7 : .3) * max, 255, THRESH_BINARY);
    c.show("foreground", r / 255);
    if (n == 2) {
        return;
    }
    morphologyEx(r, r, MORPH_OPEN, Mat::ones(3, 3, CV_8U));
    r.convertTo(r, CV_8U);
    c.show("opening2", r);
    auto all = contours(r, RETR_TREE);
    for (const auto &contour : all) {
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
        int rad = int(radius);
        bool ok = rad > 0 && contourArea(contour) / (3.14 * rad * rad) >= .5;
        label(a, ok ? "OK" : "BAD", center, Scalar::all(255), 2);
    }
    label(a, "sum=" + std::to_string(all.size()), {20, 50}, Scalar::all(255), 2);
    c.show("result", a);
}
static std::string handType(const Mat &a) {
    auto cnt = largest(contours(gray(a), RETR_TREE));
    Contour hull;
    convexHull(cnt, hull);
    double area = contourArea(hull);
    return area > 0 && contourArea(cnt) / area > .9 ? "fist:0" : "finger:1";
}
void chapter8(Context &c, int n, int) {
    if (n == 5) {
        c.video([&](Mat &frame, int) {
            flip(frame, frame, 1);
            Rect roi(400, 10, 200, 200);
            roi &= Rect(0, 0, frame.cols, frame.rows);
            require(roi.area() > 0, "Camera frame too small for hand ROI");
            Mat hand = frame(roi), hsv, mask;
            cvtColor(hand, hsv, COLOR_BGR2HSV);
            inRange(hsv, Scalar(0, 28, 70), Scalar(20, 255, 255), mask);
            dilate(mask, mask, Mat::ones(2, 2, CV_8U), {-1, -1}, 4);
            GaussianBlur(mask, mask, {5, 5}, 100);
            auto all = contours(mask, RETR_TREE);
            std::string result = "no hand";
            if (!all.empty()) {
                auto cnt = largest(all);
                Contour hull;
                std::vector<int> indices;
                convexHull(cnt, hull);
                convexHull(cnt, indices);
                double area = contourArea(hull);
                std::vector<Vec4i> defects;
                if (indices.size() > 3) {
                    convexityDefects(cnt, indices, defects);
                }
                int fingers = 0;
                for (auto d : defects) {
                    Point s = cnt[d[0]], e = cnt[d[1]], f = cnt[d[2]];
                    double aa = norm(e - s), b = norm(f - s), cc = norm(e - f);
                    if (b * cc > 0) {
                        double angle =
                            std::acos(std::clamp((b * b + cc * cc - aa * aa) / (2 * b * cc), -1., 1.)) * 180 /
                            CV_PI;
                        if (angle <= 90 && d[3] > 20) {
                            ++fingers;
                            circle(hand, f, 3, {255, 0, 0}, FILLED);
                        }
                    }
                    line(hand, s, e, {0, 255, 0}, 2);
                }
                result = std::to_string(fingers ? fingers + 1
                                                : (area > 0 && contourArea(cnt) / area > .9 ? 0 : 1));
            }
            rectangle(frame, roi, {0, 0, 255});
            label(frame, result, {roi.x, 80}, Scalar(0, 0, 255), 2);
            c.show("hand", frame);
        });
        return;
    }
    if (n == 4 || n == 0) {
        for (const std::string name : {"zero.jpg", "one.jpg"}) {
            Mat a = c.read(name);
            if (n == 4) {
                label(a, handType(a), {0, 80}, {0, 0, 255}, 2);
            } else {
                auto all = contours(binary(a), RETR_LIST);
                require(!all.empty(), "No contour");
                Contour h;
                convexHull(all[0], h);
                polylines(a, h, true, {0, 255, 0}, 2);
            }
            c.show(name, a);
        }
        return;
    }
    if (n == 6) {
        std::vector<Mat> imgs{c.read("o1.jpg"), c.read("o2.jpg"), c.read("o3.jpg")};
        std::vector<Contour> cs;
        for (auto &a : imgs) {
            auto all = contours(binary(a), RETR_LIST);
            require(!all.empty(), "No contour");
            cs.push_back(all[0]);
        }
        for (int i = 0; i < 3; ++i) {
            std::cout << "match " << i << " = " << matchShapes(cs[0], cs[i], CONTOURS_MATCH_I1, 0) << '\n';
        }
        c.show("comparison", montage(imgs));
        return;
    }
    if (n == 7) {
        std::vector<std::string> names{"paper", "rock", "scissors"};
        std::vector<Contour> models;
        for (auto &name : names) {
            auto all = contours(binary(c.read(name + ".jpg")), RETR_LIST);
            require(!all.empty(), "No template contour");
            models.push_back(all[0]);
        }
        for (int j = 1; j <= 3; ++j) {
            Mat a = c.read("test" + std::to_string(j) + ".jpg");
            auto all = contours(binary(a), RETR_LIST);
            require(!all.empty(), "No test contour");
            int best = 0;
            double score = DBL_MAX;
            for (int i = 0; i < 3; ++i) {
                double s = matchShapes(all[0], models[i], CONTOURS_MATCH_I1, 0);
                if (s < score) {
                    score = s;
                    best = i;
                }
            }
            label(a, names[best], {0, 60}, Scalar::all(255), 2);
            c.show("test" + std::to_string(j), a);
        }
        return;
    }
    Mat a = c.read(n == 1 ? "contours.bmp" : "hand.bmp");
    auto all = contours(binary(a), n == 2 ? RETR_LIST : RETR_TREE);
    require(!all.empty(), "No hand contour");
    auto cnt = all[0];
    Contour hull;
    std::vector<int> indices;
    convexHull(cnt, hull);
    convexHull(cnt, indices);
    if (n == 1) {
        std::cout << Mat(hull) << '\n' << Mat(indices) << '\n';
        return;
    }
    c.show("original", a);
    if (n == 2) {
        polylines(a, hull, true, {0, 255, 0}, 2);
    } else {
        std::vector<Vec4i> ds;
        if (indices.size() > 3) {
            convexityDefects(cnt, indices, ds);
        }
        for (auto d : ds) {
            std::cout << d << '\n';
            line(a, cnt[d[0]], cnt[d[1]], {0, 0, 255}, 2);
            circle(a, cnt[d[2]], 5, {255, 0, 0}, FILLED);
        }
    }
    c.show("result", a);
}
} // namespace cv40

#include "cv40.hpp"
namespace cv40 {
static void sortContours(Contours &cs, bool vertical) {
    std::sort(cs.begin(), cs.end(), [&](const auto &a, const auto &b) {
        auto x = boundingRect(a), y = boundingRect(b);
        return vertical ? x.y < y.y : x.x < y.x;
    });
}
static Contours answerOptions(const Mat &image) {
    auto all = contours(image);
    Contours result;
    for (auto &c : all) {
        auto r = boundingRect(c);
        double ar = double(r.width) / r.height;
        if (r.width >= 25 && r.height >= 25 && ar >= .6 && ar <= 1.3) {
            result.push_back(c);
        }
    }
    return result;
}
static int choice(const Mat &threshold, const Contours &options, Context *c = nullptr) {
    require(!options.empty(), "No answer options");
    int best = 0, max = -1;
    for (std::size_t i = 0; i < options.size(); ++i) {
        Mat mask = Mat::zeros(threshold.size(), CV_8U);
        drawContours(mask, options, int(i), Scalar(255), FILLED);
        if (c) {
            c->show("mask", mask);
        }
        mask = masked(threshold, mask);
        int count = countNonZero(mask);
        if (count > max) {
            max = count;
            best = int(i);
        }
        if (c) {
            c->show("filled", mask);
        }
    }
    return best;
}
void chapter9(Context &c, int n, int) {
    if (n == 1) {
        Mat a = c.read("xiaogang.jpg"), t = binary(a, 0, THRESH_BINARY_INV | THRESH_OTSU);
        auto options = contours(t);
        sortContours(options, false);
        int best = choice(t, options, &c);
        require(best < 4, "Expected four answer choices");
        std::cout << "choice=" << char('A' + best) << " correct=" << (best == 2) << '\n';
        drawContours(a, options, best, best == 2 ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
        c.show("result", a);
        return;
    }
    if (n == 3) {
        for (const std::string name : {"xtest.jpg", "xtest2.jpg"}) {
            Mat a = c.read(name);
            auto cs = contours(binary(a));
            require(!cs.empty(), "No polygon");
            Contour approx;
            approxPolyDP(cs[0], approx, .01 * arcLength(cs[0], true), true);
            std::cout << name << ": vertices=" << cs[0].size() << ", approximate=" << approx.size() << '\n';
            c.show(name, a);
        }
        return;
    }
    if (n == 0 || n == 2 || n == 4 || n == 10) {
        Mat a = c.read("b.jpg"), g = gray(a), blurred, edge;
        GaussianBlur(g, blurred, {5, 5}, 0);
        Canny(blurred, edge, 50, 200);
        auto cs = contours(edge);
        if (n == 0 || n == 2) {
            if (n == 2) {
                c.show("original", a);
                c.show("gray", g);
                c.show("gaussian", blurred);
                c.show("edge", edge);
            }
            drawContours(a, cs, -1, {0, 0, 255}, 3);
            c.show("contours", a);
            return;
        }
        std::sort(cs.begin(), cs.end(),
                  [](const auto &a, const auto &b) { return contourArea(a) > contourArea(b); });
        Mat paper;
        for (auto &contour : cs) {
            Contour approx;
            approxPolyDP(contour, approx, .01 * arcLength(contour, true), true);
            if (approx.size() == 4) {
                std::vector<Point2f> p;
                for (auto pt : approx) {
                    p.emplace_back(pt);
                }
                paper = perspective(a, p);
                break;
            }
        }
        require(!paper.empty(), "No quadrilateral answer sheet found");
        c.show("paper", paper);
        if (n == 4) {
            return;
        }
        Mat t = binary(paper, 0, THRESH_BINARY_INV | THRESH_OTSU);
        auto opts = answerOptions(t);
        sortContours(opts, true);
        std::array<int, 5> answers{1, 2, 0, 2, 3};
        require(opts.size() == answers.size() * 4, "Expected 5 rows of 4 answer bubbles");
        int correct = 0;
        for (int row = 0; row < 5; ++row) {
            Contours group(opts.begin() + row * 4, opts.begin() + row * 4 + 4);
            sortContours(group, false);
            int best = choice(t, group);
            bool ok = best == answers[row];
            correct += ok;
            std::cout << row + 1 << ": " << char('A' + best) << '\n';
            drawContours(paper, group, best, ok ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
        }
        label(paper, "total:5 right:" + std::to_string(correct) + " score:" + std::to_string(correct * 20),
              {10, 30}, {0, 0, 255}, .5);
        c.show("score", paper);
        return;
    }
    if (n == 5) {
        Mat a = c.read("paper.jpg", IMREAD_GRAYSCALE), t = binary(a, 0, THRESH_BINARY_INV | THRESH_OTSU);
        c.show("paper", a);
        c.show("threshold", t);
        c.save("thresh.bmp", t);
        return;
    }
    Mat t = c.read("thresh.bmp", IMREAD_GRAYSCALE), a;
    cvtColor(t, a, COLOR_GRAY2BGR);
    auto all = contours(t);
    std::cout << "contours=" << all.size() << '\n';
    if (n == 6) {
        drawContours(a, all, -1, {0, 0, 255}, 3);
        c.show("result", a);
        return;
    }
    auto opts = answerOptions(t);
    std::cout << "options=" << opts.size() << '\n';
    if (n == 7) {
        drawContours(a, opts, -1, {0, 0, 255}, 5);
        c.show("options", a);
        return;
    }
    for (std::size_t i = 0; i < all.size(); ++i) {
        auto r = boundingRect(all[i]);
        if (r.width >= 25 && r.height >= 25 && double(r.width) / r.height >= .6 &&
            double(r.width) / r.height <= 1.3) {
            label(a, std::to_string(i), r.tl() - Point(1, 5), {0, 0, 255}, .5);
        }
    }
    c.show("unsorted", a);
    sortContours(opts, true);
    if (n == 8) {
        cvtColor(t, a, COLOR_GRAY2BGR);
        for (std::size_t i = 0; i < opts.size(); ++i) {
            label(a, std::to_string(i), boundingRect(opts[i]).tl() - Point(1, 5), {0, 0, 255}, .5);
        }
        c.show("sorted", a);
    } else {
        for (std::size_t i = 0; i < opts.size(); i += 4) {
            Contours group(opts.begin() + i, opts.begin() + std::min(i + 4, opts.size()));
            sortContours(group, false);
            Mat row = Mat::zeros(a.size(), a.type());
            for (std::size_t j = 0; j < group.size(); ++j) {
                drawContours(row, group, int(j), Scalar::all(255), FILLED);
                label(row, std::to_string(j), boundingRect(group[j]).tl() - Point(1, 5));
            }
            c.show("row" + std::to_string(i / 4), row);
        }
    }
}
} // namespace cv40

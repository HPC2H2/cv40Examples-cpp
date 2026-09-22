#include "cv40.hpp"
namespace cv40 {
void chapter11(Context &c, int n, int) {
    if (n == 6) {
        std::array<int, 4> a{1, 2, 3, 4};
        auto &view = a;
        view[1] = 666;
        auto copy = a;
        copy[0] = 99;
        std::cout << "view changes original: " << a[1] << ", copy does not: " << a[0] << '\n';
        return;
    }
    if (n == 8) {
        Mat t = (Mat_<uchar>(1, 4) << 0, 1, 1, 1);
        for (Mat other :
             std::vector<Mat>{(Mat_<uchar>(1, 4) << 0, 1, 1, 1), (Mat_<uchar>(1, 4) << 1, 1, 1, 1),
                              (Mat_<uchar>(1, 4) << 1, 0, 0, 0)}) {
            std::cout << hamming(t, other) << '\n';
        }
        return;
    }
    if (n == 10) {
        c.show("gallery",
               montage({c.read("image/fruit.jpg"), c.read("image/sunset.jpg"), c.read("image/tomato.jpg")}));
        return;
    }
    if (n == 9 || n == 11) {
        auto paths = imageFiles(c.chapterDir() / "image");
        require(!paths.empty(), "Empty image database");
        std::vector<std::pair<int, fs::path>> scores;
        Mat query;
        if (n == 11) {
            query = c.read("apple.jpg");
        }
        for (auto &p : paths) {
            Mat h = hashImage(readImage(p));
            if (n == 9) {
                std::cout << p.filename().u8string() << " " << h << '\n';
            } else {
                scores.emplace_back(hamming(hashImage(query), h), p);
            }
        }
        if (n == 11) {
            std::sort(scores.begin(), scores.end());
            std::vector<Mat> images{query};
            for (std::size_t i = 0; i < std::min<std::size_t>(3, scores.size()); ++i) {
                std::cout << scores[i].first << ' ' << scores[i].second.filename().u8string() << '\n';
                images.push_back(readImage(scores[i].second));
            }
            c.show("search", montage(images));
        }
        return;
    }
    Mat a = c.read(n == 1 || n == 7 ? "lena.bmp"
                   : n == 2         ? "rst.bmp"
                                    : "rst88.bmp",
                   n >= 3 && n <= 5 ? IMREAD_UNCHANGED : IMREAD_COLOR),
        r;
    if (n == 1) {
        resize(a, r, {8, 8});
        std::cout << a.size() << " -> " << r.size() << '\n';
        c.show("resized", r);
    } else if (n == 2) {
        r = gray(a);
        std::cout << "channels " << a.channels() << " -> " << r.channels() << '\n';
        c.show("gray", r);
    } else if (n == 7) {
        std::cout << hashImage(a) << '\n';
    } else {
        double m = mean(a)[0];
        std::cout << a << "\nmean=" << m << '\n';
        if (n >= 4) {
            r = (a > m) / 255;
            std::cout << r << '\n';
            if (n == 5) {
                std::cout << r.reshape(1, 1) << '\n';
            }
        }
    }
}
void chapter12(Context &c, int, int) {
    Mat query = c.read("image/test2/3.bmp", IMREAD_GRAYSCALE);
    double best = -DBL_MAX;
    int number = -1;
    for (int digit = 0; digit < 10; ++digit) {
        for (auto &p : imageFiles(c.chapterDir() / "image" / std::to_string(digit))) {
            Mat t = binary(readImage(p), 0, THRESH_BINARY | THRESH_OTSU), r;
            resize(t, t, query.size());
            matchTemplate(query, t, r, TM_CCOEFF);
            double score = r.at<float>(0);
            if (score > best) {
                best = score;
                number = digit;
            }
        }
    }
    require(number >= 0, "No digit templates");
    std::cout << "recognized digit=" << number << '\n';
    c.show("query", query);
}
} // namespace cv40

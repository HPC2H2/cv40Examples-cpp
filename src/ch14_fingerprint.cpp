// SIFT examples: 李立宗. MCC reference: Raffaele Cappelli, WSB 2021.
#include "cv40.hpp"
namespace cv40 {
void mccDemo(Context &);
struct SiftMatch {
    std::vector<KeyPoint> a, b;
    std::vector<DMatch> good;
};
static SiftMatch siftMatch(const Mat &a, const Mat &b, bool flann) {
    SiftMatch result;
    Mat da, db;
    auto sift = SIFT::create();
    sift->detectAndCompute(a, noArray(), result.a, da);
    sift->detectAndCompute(b, noArray(), result.b, db);
    if (da.empty() || db.rows < 2) {
        return result;
    }
    std::vector<std::vector<DMatch>> matches;
    if (flann) {
        FlannBasedMatcher matcher(makePtr<flann::KDTreeIndexParams>(5), makePtr<flann::SearchParams>(50));
        matcher.knnMatch(da, db, matches, 2);
    } else {
        BFMatcher matcher;
        matcher.knnMatch(da, db, matches, 2);
    }
    for (auto &pair : matches) {
        if (pair.size() == 2 && pair[0].distance < .8f * pair[1].distance) {
            result.good.push_back(pair[0]);
        }
    }
    return result;
}
void chapter14(Context &c, int n, int) {
    if (n == 0) {
        mccDemo(c);
        return;
    }
    if (n == 1) {
        Mat a = c.read("lena.bmp", IMREAD_GRAYSCALE);
        c.show("original", a);
        for (int i = 1; i <= 3; ++i) {
            pyrDown(a, a);
            std::cout << a.size() << '\n';
            c.show("pyramid" + std::to_string(i), a);
        }
        return;
    }
    if (n == 2) {
        Mat a = c.read("lena.bmp"), r;
        c.show("original", a);
        c.save("gaussian/o.bmp", a);
        for (int k : {3, 13, 21}) {
            GaussianBlur(a, r, {k, k}, 0);
            c.show("gaussian" + std::to_string(k), r);
            c.save("gaussian/r" + std::to_string(k) + ".bmp", r);
        }
        return;
    }
    if (n == 3) {
        Mat a = c.read("fingerprint.png"), d;
        std::vector<KeyPoint> kp;
        SIFT::create()->detectAndCompute(a, noArray(), kp, d);
        std::cout << "keypoints=" << kp.size() << " descriptors=" << d.size() << '\n';
        if (!kp.empty()) {
            auto p = kp[0];
            std::cout << p.pt << " size=" << p.size << " angle=" << p.angle << " response=" << p.response
                      << " octave=" << p.octave << " class=" << p.class_id << '\n'
                      << d.row(0) << '\n';
        }
        drawKeypoints(a, kp, a);
        c.show("points", a);
        return;
    }
    if (n == 4 || n == 5) {
        Mat a = c.read(n == 4 ? "a.png" : "gua1.jpg"), b = c.read(n == 4 ? "b.png" : "gua2.jpg"), rotated;
        rotate(b, rotated, ROTATE_90_CLOCKWISE);
        std::vector<std::pair<Mat, Mat>> pairs{{a, b}, {a, rotated}, {b, rotated}};
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            auto [x, y] = pairs[i];
            auto m = siftMatch(x, y, n == 5);
            Mat r;
            drawMatches(x, m.a, y, m.b, m.good, r, Scalar::all(-1), Scalar::all(-1), {},
                        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            c.show("matches" + std::to_string(i), r);
        }
        return;
    }
    if (n == 6) {
        Mat model = c.read("verification/model.bmp");
        for (const std::string name : {"src1", "src2"}) {
            Mat a = c.read("verification/" + name + ".bmp");
            auto m = siftMatch(a, model, true);
            std::cout << name << ": matches=" << m.good.size() << " verified=" << (m.good.size() >= 500)
                      << '\n';
            c.show(name, a);
        }
        c.show("model", model);
        return;
    }
    Mat a = c.read("identification/src.bmp");
    int best = 0;
    std::string filename;
    for (auto &p : imageFiles(c.chapterDir() / "identification/database")) {
        int count = int(siftMatch(a, readImage(p), true).good.size());
        std::cout << p.filename().u8string() << " matches=" << count << '\n';
        if (count > best) {
            best = count;
            filename = p.filename().u8string();
        }
    }
    const std::vector<std::string> names{u8"孙悟空", u8"猪八戒", u8"红孩儿", u8"刘能",   u8"赵四",
                                         u8"杰克",   u8"杰克森", "tonny",    u8"大柱子", u8"翠花"};
    std::string name = u8"没找到";
    if (best >= 100 && !filename.empty() && filename[0] >= '0' && filename[0] <= '9') {
        name = names[filename[0] - '0'];
    }
    std::cout << "identity=" << name << '\n';
}
} // namespace cv40

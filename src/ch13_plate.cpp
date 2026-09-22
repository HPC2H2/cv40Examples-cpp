#include "cv40.hpp"
namespace cv40 {
static Mat plate(Context &c) {
    Mat a = c.read("gua.jpg"), b, g, s, t;
    c.show("original", a);
    GaussianBlur(a, b, {3, 3}, 0);
    g = gray(b);
    Sobel(g, s, CV_16S, 1, 0);
    convertScaleAbs(s, s);
    t = binary(s, 0, THRESH_BINARY | THRESH_OTSU);
    c.show("sobel", s);
    c.show("threshold", t);
    morphologyEx(t, t, MORPH_CLOSE, getStructuringElement(MORPH_RECT, {17, 5}));
    c.show("close", t);
    morphologyEx(t, t, MORPH_OPEN, getStructuringElement(MORPH_RECT, {1, 19}));
    c.show("open", t);
    medianBlur(t, t, 15);
    c.show("median", t);
    auto cs = contours(t, RETR_TREE);
    Rect chosen;
    for (auto &cnt : cs) {
        auto r = boundingRect(cnt);
        if (r.width > 3 * r.height) {
            chosen = r;
        }
    }
    require(chosen.area() > 0, "No license plate candidate found");
    Mat result = a(chosen).clone();
    drawContours(a, cs, -1, {0, 0, 255}, 3);
    c.show("contours", a);
    c.show("plate", result);
    return result;
}
static std::vector<Mat> plateCharacters(Context &c, const Mat &a) {
    Mat b, g, t;
    GaussianBlur(a, b, {3, 3}, 0);
    cvtColor(b, g, COLOR_RGB2GRAY);
    threshold(g, t, 0, 255, THRESH_BINARY | THRESH_OTSU);
    dilate(t, t, getStructuringElement(MORPH_RECT, {2, 2}));
    auto cs = contours(t);
    std::vector<Rect> rs;
    for (auto &cnt : cs) {
        rs.push_back(boundingRect(cnt));
    }
    std::sort(rs.begin(), rs.end(), [](auto a, auto b) { return a.x < b.x; });
    std::vector<Mat> out;
    Mat marked = a.clone();
    for (auto r : rs) {
        rectangle(marked, r, {0, 0, 255});
        if (r.height > r.width * 1.5 && r.height < r.width * 8 && r.width > 3) {
            out.push_back(t(r).clone());
        }
    }
    c.show("binary", t);
    c.show("boxes", marked);
    for (std::size_t i = 0; i < out.size(); ++i) {
        c.show("character" + std::to_string(i), out[i]);
        c.save("characters/" + std::to_string(i) + ".bmp", out[i]);
    }
    return out;
}
void chapter13(Context &c, int n, int variant) {
    Mat a = n == 2 ? c.read("gg.bmp") : plate(c);
    if (n == 1) {
        return;
    }
    auto characters = plateCharacters(c, a);
    if (n == 2) {
        return;
    }
    require(!characters.empty(), "No plate characters found");
    const std::vector<std::string> classes = {
        "0",    "1",    "2",    "3",    "4",    "5",    "6",    "7",    "8",    "9",    "A",    "B",
        "C",    "D",    "E",    "F",    "G",    "H",    "J",    "K",    "L",    "M",    "N",    "P",
        "Q",    "R",    "S",    "T",    "U",    "V",    "W",    "X",    "Y",    "Z",    u8"京", u8"津",
        u8"冀", u8"晋", u8"蒙", u8"辽", u8"吉", u8"黑", u8"沪", u8"苏", u8"浙", u8"皖", u8"闽", u8"赣",
        u8"鲁", u8"豫", u8"鄂", u8"湘", u8"粤", u8"桂", u8"琼", u8"渝", u8"川", u8"贵", u8"云", u8"藏",
        u8"陕", u8"甘", u8"青", u8"宁", u8"新", u8"港", u8"澳", u8"台"};
    std::vector<std::vector<Mat>> templates(classes.size());
    for (std::size_t i = 0; i < classes.size(); ++i) {
        for (auto &p : imageFiles(c.chapterDir() / "template" / fs::u8path(classes[i]))) {
            templates[i].push_back(binary(readImage(p), 0, THRESH_BINARY | THRESH_OTSU));
        }
    }
    std::string result;
    for (std::size_t pos = 0; pos < characters.size(); ++pos) {
        double best = -DBL_MAX;
        int index = -1;
        for (std::size_t i = 0; i < classes.size(); ++i) {
            if (variant &&
                ((pos == 0 && i < 34) || (pos == 1 && (i < 10 || i >= 34)) || (pos > 1 && i >= 34))) {
                continue;
            }
            for (auto &t : templates[i]) {
                Mat scaled, score;
                resize(t, scaled, characters[pos].size());
                matchTemplate(characters[pos], scaled, score, TM_CCOEFF);
                if (score.at<float>(0) > best) {
                    best = score.at<float>(0);
                    index = int(i);
                }
            }
        }
        require(index >= 0, "No matching templates for plate position " + std::to_string(pos));
        result += classes[index];
    }
    std::cout << "plate=" << result << '\n';
}
} // namespace cv40

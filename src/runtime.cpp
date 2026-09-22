#include "cv40.hpp"
namespace cv40 {
void require(bool ok, const std::string &message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}
Mat readImage(const fs::path &path, int mode) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "Cannot open image: " + path.u8string());
    std::vector<uchar> data((std::istreambuf_iterator<char>(f)), {});
    Mat image = imdecode(data, mode);
    require(!image.empty(), "Cannot decode image: " + path.u8string());
    return image;
}
void writeImage(const fs::path &path, const Mat &image) {
    fs::create_directories(path.parent_path());
    std::vector<uchar> bytes;
    require(imencode(path.extension().string(), image, bytes), "Cannot encode image");
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(bool(f), "Cannot write: " + path.u8string());
}
fs::path Context::chapterDir() const {
    return root / *fs::u8path(example->source).begin();
}
fs::path Context::file(const std::string &name) const {
    const auto p = fs::u8path(name);
    std::vector<fs::path> candidates;
    if (!models.empty()) {
        candidates.push_back(models / p);
    }
    candidates.push_back(root / fs::u8path(example->source).parent_path() / p);
    candidates.push_back(chapterDir() / p);
    candidates.push_back(output / example->id / p);
    for (const auto &candidate : candidates) {
        if (fs::is_regular_file(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("Missing resource: " + name + " (example " + example->id +
                             "). Use --models DIR or check --data DIR.");
}
Mat Context::read(const std::string &name, int mode) const {
    return readImage(file(name), mode);
}
void Context::save(const std::string &name, const Mat &image) const {
    writeImage(output / example->id / fs::u8path(name), image);
}
void Context::show(const std::string &name, const Mat &image) {
    require(!image.empty(), "Empty image: " + name);
    if (!headless) {
        imshow(name, image);
        ++imageNumber;
        return;
    }
    Mat display;
    if (image.depth() == CV_8U) {
        display = image;
    } else if (image.depth() == CV_32F || image.depth() == CV_64F) {
        image.convertTo(display, CV_8U, 255);
    } else {
        normalize(image, display, 0, 255, NORM_MINMAX, CV_8U);
    }
    std::string safe = name;
    for (char &x : safe) {
        if (!std::isalnum(static_cast<unsigned char>(x)) && x != '-' && x != '_') {
            x = '_';
        }
    }
    save(std::to_string(++imageNumber) + "_" + safe + ".png", display);
}
double Context::ask(const std::string &prompt, double fallback) {
    if (inputIndex < input.size()) {
        return input[inputIndex++];
    }
    if (headless) {
        return fallback;
    }
    std::cout << prompt << ": ";
    double value;
    require(bool(std::cin >> value), "Expected numeric input");
    return value;
}
void Context::video(const std::function<void(Mat &, int)> &process) {
    require(maxFrames >= 0, "Frame limit must not be negative");
    if (!frames.empty()) {
        const auto files = imageFiles(frames);
        require(!files.empty(), "No replay images in " + frames.u8string());
        const std::size_t limit =
            maxFrames > 0 ? std::min(files.size(), std::size_t(maxFrames)) : files.size();
        std::size_t processed = 0;
        for (std::size_t index = 0; index < limit; ++index) {
            Mat frame = readImage(files[index]);
            process(frame, static_cast<int>(index));
            ++processed;
            if (!headless && (waitKey(10) & 255) == 27) {
                break;
            }
        }
        std::cout << "frames=" << processed << " source=replay\n";
        return;
    }
    require(!headless || maxFrames > 0, "Camera in headless mode requires --max-frames N");
    VideoCapture cap(camera);
    require(cap.isOpened(), "Cannot open camera " + std::to_string(camera));
    int processed = 0;
    for (int i = 0; !maxFrames || i < maxFrames; ++i) {
        Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        process(frame, i);
        ++processed;
        if (!headless && (waitKey(10) & 255) == 27) {
            break;
        }
    }
    require(processed > 0, "Camera opened but returned no frames");
    std::cout << "frames=" << processed << " source=camera\n";
}
std::vector<fs::path> imageFiles(const fs::path &dir) {
    std::vector<fs::path> result;
    if (!fs::is_directory(dir)) {
        return result;
    }
    for (const auto &entry : fs::directory_iterator(dir)) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char x) { return static_cast<char>(std::tolower(x)); });
        if (entry.is_regular_file() &&
            (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".gif")) {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}
Mat gray(const Mat &image) {
    if (image.channels() == 1) {
        return image.clone();
    }
    Mat r;
    cvtColor(image, r, COLOR_BGR2GRAY);
    return r;
}
Mat binary(const Mat &image, double t, int type) {
    Mat r;
    threshold(gray(image), r, t, 255, type);
    return r;
}
Contours contours(const Mat &image, int mode) {
    Contours r;
    findContours(gray(image), r, mode, CHAIN_APPROX_SIMPLE);
    return r;
}
Contour largest(const Contours &all) {
    require(!all.empty(), "No contours found");
    return *std::max_element(all.begin(), all.end(),
                             [](const auto &a, const auto &b) { return contourArea(a) < contourArea(b); });
}
Mat masked(const Mat &image, const Mat &mask) {
    Mat r;
    bitwise_and(image, image, r, mask);
    return r;
}
Mat wrapAdd(const Mat &a, const Mat &b) {
    require(a.type() == b.type() && a.size() == b.size() && a.depth() == CV_8U,
            "wrapAdd requires equally sized uint8 images");
    Mat r(a.size(), a.type());
    for (int y = 0; y < a.rows; ++y) {
        for (int x = 0; x < a.cols * a.channels(); ++x) {
            r.ptr<uchar>(y)[x] = static_cast<uchar>(unsigned(a.ptr<uchar>(y)[x]) + b.ptr<uchar>(y)[x]);
        }
    }
    return r;
}
Mat hashImage(const Mat &image) {
    Mat r;
    resize(image, r, {8, 8});
    r = gray(r);
    r = r > mean(r)[0];
    r /= 255;
    return r.reshape(1, 1).clone();
}
int hamming(const Mat &a, const Mat &b) {
    require(a.size() == b.size(), "Hash sizes differ");
    return countNonZero(a != b);
}
Mat montage(const std::vector<Mat> &images, int width) {
    require(!images.empty(), "No images to plot");
    Mat canvas(300, width * static_cast<int>(images.size()), CV_8UC3, Scalar::all(255));
    for (std::size_t i = 0; i < images.size(); ++i) {
        Mat m = images[i];
        if (m.channels() == 1) {
            cvtColor(m, m, COLOR_GRAY2BGR);
        }
        double s = std::min(double(width) / m.cols, 300.0 / m.rows);
        resize(m, m, {}, s, s);
        m.copyTo(canvas(Rect(static_cast<int>(i) * width, 0, m.cols, m.rows)));
    }
    return canvas;
}
void label(Mat &image, const std::string &text, Point at, Scalar color, double scale) {
    putText(image, text, at, FONT_HERSHEY_SIMPLEX, scale, color, 2);
}
Mat perspective(const Mat &image, std::vector<Point2f> p) {
    require(p.size() == 4, "Perspective transform requires four corners");
    std::sort(p.begin(), p.end(), [](auto a, auto b) { return a.x < b.x; });
    if (p[0].y > p[1].y) {
        std::swap(p[0], p[1]);
    }
    Point2f tl = p[0], bl = p[1], br = p[2], tr = p[3];
    if (norm(br - tl) < norm(tr - tl)) {
        std::swap(br, tr);
    }
    int w = std::max(2, int(std::max(norm(br - bl), norm(tr - tl)))),
        h = std::max(2, int(std::max(norm(tr - br), norm(tl - bl))));
    std::vector<Point2f> src{tl, tr, br, bl},
        dst{{0, 0}, {float(w - 1), 0}, {float(w - 1), float(h - 1)}, {0, float(h - 1)}};
    Mat result;
    warpPerspective(image, result, getPerspectiveTransform(src, dst), {w, h});
    return result;
}
void run(Context &c) {
    using Function = void (*)(Context &, int, int);
    static const std::map<int, Function> chapters = {
        {2, chapter2},   {3, chapter3},   {4, chapter4},   {5, chapter5},   {6, chapter6},
        {7, chapter7},   {8, chapter8},   {9, chapter9},   {10, chapter10}, {11, chapter11},
        {12, chapter12}, {13, chapter13}, {14, chapter14}, {15, chapter15}, {16, chapter16},
        {17, chapter17}, {18, chapter18}, {19, chapter19}, {20, chapter20}, {23, chapter23},
        {24, chapter24}, {25, chapter25}, {26, chapter26}, {27, chapter27}, {28, chapter28}};
    c.imageNumber = 0;
    c.inputIndex = 0;
    chapters.at(c.example->chapter)(c, c.example->number, c.example->variant);
}
} // namespace cv40

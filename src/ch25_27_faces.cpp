#include "faces.hpp"
#include "models.hpp"
namespace cv40 {
static CascadeClassifier cascade(Context &c, const std::string &name) {
    auto data = bytes(c.file(name));
    FileStorage storage(std::string(data.begin(), data.end()), FileStorage::READ | FileStorage::MEMORY);
    require(storage.isOpened(), "Cannot parse cascade XML");
    CascadeClassifier model;
    if (!model.read(storage.getFirstTopLevelNode())) {
        // load() also handles the legacy opencv-haar-classifier XML supplied with the book.
        auto cached = fs::path(CV40_DEFAULT_ROOT).parent_path() / "build/model-cache" / name;
        fs::create_directories(cached.parent_path());
        fs::copy_file(c.file(name), cached, fs::copy_options::overwrite_existing);
        require(model.load(cached.string()), "Invalid cascade model");
    }
    return model;
}
void chapter25(Context &c, int n, int) {
    auto model = cascade(c, "haarcascade_frontalface_default.xml");
    if (n == 1) {
        Mat a = c.read("manyPeople.jpg");
        std::vector<Rect> faces;
        model.detectMultiScale(gray(a), faces, 1.04, 18, 0, {8, 8});
        std::cout << "faces=" << faces.size() << '\n';
        for (auto r : faces) {
            std::cout << r << '\n';
            rectangle(a, r, {0, 255, 0}, 2);
        }
        c.show("faces", a);
    } else {
        auto smile = cascade(c, "haarcascade_smile.xml");
        c.video([&](Mat &frame, int) {
            flip(frame, frame, 1);
            Mat g = gray(frame);
            std::vector<Rect> faces;
            model.detectMultiScale(g, faces, 1.1, 5, 0, {5, 5});
            for (auto r : faces) {
                rectangle(frame, r, {0, 255, 0}, 2);
                std::vector<Rect> smiles;
                smile.detectMultiScale(g(r), smiles, 1.5, 25, 0, {50, 50});
                if (!smiles.empty()) {
                    label(frame, "smile", r.tl(), {0, 255, 255});
                }
            }
            c.show("smiles", frame);
        });
    }
}
void chapter26(Context &c, int n, int) {
    std::vector<Mat> images;
    Mat test;
    Ptr<face::FaceRecognizer> model;
    std::vector<int> labels{0, 0, 1, 1};
    if (n == 1) {
        for (auto name : {"a1.png", "a2.png", "b1.png", "b2.png"}) {
            images.push_back(c.read(name, IMREAD_GRAYSCALE));
        }
        test = c.read("a3.png", IMREAD_GRAYSCALE);
        model = face::LBPHFaceRecognizer::create();
    } else {
        std::string prefix = n == 2 ? "e" : "f";
        for (auto name : {"01.png", "02.png", "11.png", "12.png"}) {
            images.push_back(c.read(prefix + name, IMREAD_GRAYSCALE));
        }
        test = c.read(prefix + "Test.png", IMREAD_GRAYSCALE);
        if (n == 2) {
            model = face::EigenFaceRecognizer::create();
        } else {
            model = face::FisherFaceRecognizer::create();
        }
    }
    model->train(images, labels);
    int prediction;
    double confidence;
    model->predict(test, prediction, confidence);
    std::cout << "label=" << prediction << " confidence=" << confidence << '\n';
    if (n == 2) {
        label(test, prediction == 0 ? "first" : "second", {0, 30}, Scalar::all(255), .8);
        c.show("recognized", test);
    }
}
void cnnFaceDemo(Context &);
void chapter27(Context &c, int n, int variant) {
    if (n == 5) {
        cnnFaceDemo(c);
        return;
    }
    if (n == 1) {
        auto detect = [&](Mat &a, int) {
            for (auto r : faceBoxes(a, 1)) {
                drawFaceBox(a, r);
            }
            c.show("faces", a);
        };
        if (variant) {
            c.video(detect);
        } else {
            Mat a = c.read("people.jpg");
            detect(a, 0);
        }
        return;
    }
    auto predictor = landmarkModel(c);
    Mat a = c.read(n == 2 ? "y.jpg" : n == 3 ? "image.jpg" : "rotate.jpg");
    auto boxes = faceBoxes(a, n == 4 ? 1 : 0);
    std::cout << "faces=" << boxes.size() << '\n';
    if (n == 4) {
        auto image = faceImage(a);
        std::vector<dlib::full_object_detection> shapes;
        for (auto &r : boxes) {
            shapes.push_back(predictor(image, r));
        }
        dlib::array<dlib::matrix<dlib::rgb_pixel>> chips;
        dlib::extract_image_chips(image, dlib::get_face_chip_details(shapes, 120), chips);
        for (unsigned long i = 0; i < chips.size(); ++i) {
            Mat bgr;
            cvtColor(dlib::toMat(chips[i]), bgr, COLOR_RGB2BGR);
            c.show("aligned" + std::to_string(i), bgr);
        }
        c.show("original", a);
        return;
    }
    for (auto &r : boxes) {
        auto p = landmarks(a, r, predictor);
        if (n == 2) {
            for (int i = 0; i < 68; ++i) {
                circle(a, p[i], 2, {0, 255, 0}, FILLED);
                putText(a, std::to_string(i + 1), p[i], FONT_HERSHEY_SIMPLEX, .4, Scalar::all(255), 1,
                        LINE_AA);
            }
        } else {
            for (auto [begin, end] :
                 std::vector<std::pair<int, int>>{{48, 59}, {60, 68}, {42, 48}, {36, 42}}) {
                Contour h;
                convexHull(Contour(p.begin() + begin, p.begin() + end), h);
                polylines(a, h, true, {0, 255, 0}, 2);
            }
            for (auto [begin, end] :
                 std::vector<std::pair<int, int>>{{0, 17}, {17, 22}, {22, 27}, {27, 36}}) {
                polylines(a, Contour(p.begin() + begin, p.begin() + end), false, {0, 255, 0}, 2);
            }
        }
    }
    c.show("landmarks", a);
}
} // namespace cv40

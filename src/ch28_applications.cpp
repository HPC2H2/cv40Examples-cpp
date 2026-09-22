#include "faces.hpp"
#include "models.hpp"
namespace cv40 {
static void hullRange(Mat &a, const Contour &p, int start, int end) {
    Contour hull;
    convexHull(Contour(p.begin() + start, p.begin() + end), hull);
    polylines(a, hull, true, {0, 255, 0}, 1);
}
static double eyeRatio(const Contour &p, int start) {
    double width = norm(p[start] - p[start + 3]);
    return width > 0 ? (norm(p[start + 1] - p[start + 5]) + norm(p[start + 2] - p[start + 4])) / (2 * width)
                     : 0;
}
static std::vector<int> alignmentPoints() {
    std::vector<int> ids;
    for (auto [start, end] :
         std::vector<std::pair<int, int>>{{22, 27}, {36, 42}, {42, 48}, {17, 22}, {27, 35}, {48, 61}}) {
        for (int i = start; i < end; ++i) {
            ids.push_back(i);
        }
    }
    return ids;
}
static Mat faceMask(Size size, const Contour &points) {
    auto ids = alignmentPoints();
    Contour selected, hull;
    for (int i : ids) {
        selected.push_back(points[i]);
    }
    convexHull(selected, hull);
    Mat mask = Mat::zeros(size, CV_64F);
    fillConvexPoly(mask, hull, Scalar(1));
    GaussianBlur(mask, mask, {15, 15}, 0);
    Mat color;
    merge(std::vector<Mat>{mask, mask, mask}, color);
    return color;
}
static Mat similarity(const Contour &from, const Contour &to) {
    auto ids = alignmentPoints();
    Mat a(int(ids.size()), 2, CV_64F), b(a.size(), a.type());
    Point2d ca{}, cb{};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ca += Point2d(from[ids[i]]);
        cb += Point2d(to[ids[i]]);
    }
    ca *= 1. / ids.size();
    cb *= 1. / ids.size();
    for (int i = 0; i < a.rows; ++i) {
        a.at<double>(i, 0) = from[ids[i]].x - ca.x;
        a.at<double>(i, 1) = from[ids[i]].y - ca.y;
        b.at<double>(i, 0) = to[ids[i]].x - cb.x;
        b.at<double>(i, 1) = to[ids[i]].y - cb.y;
    }
    double sa = norm(a) / std::sqrt(double(a.total())), sb = norm(b) / std::sqrt(double(b.total()));
    require(sa > 0 && sb > 0, "Degenerate face landmarks");
    a /= sa;
    b /= sb;
    SVD svd(a.t() * b);
    Mat rotation = (svd.u * svd.vt).t();
    Mat linear = rotation * (sb / sa), result(2, 3, CV_64F);
    linear.copyTo(result(Rect(0, 0, 2, 2)));
    Mat ac = (Mat_<double>(2, 1) << ca.x, ca.y);
    Mat bc = (Mat_<double>(2, 1) << cb.x, cb.y);
    Mat translation = bc - linear * ac;
    translation.copyTo(result.col(2));
    return result;
}
void chapter28(Context &c, int n, int variant) {
    if (n == 0) {
        std::vector<int> a{32, 234, 523, 5, 2};
        std::pair<int, int> b{32, 2134};
        std::map<std::string, int> d{{u8"李立宗", 66}, {u8"刘能", 88}, {u8"赵四", 99}};
        std::cout << "std::vector<int> " << a.size() << "\nstd::pair<int,int> " << b.first
                  << "\nstd::map<string,int> " << d.size() << '\n';
        return;
    }
    if (n == 4) {
        auto faceNet =
                 tensorflow(c, "model/opencv_face_detector_uint8.pb", "model/opencv_face_detector.pbtxt"),
             ageNet = caffe(c, "model/deploy_age.prototxt", "model/age_net.caffemodel"),
             genderNet = caffe(c, "model/deploy_gender.prototxt", "model/gender_net.caffemodel");
        std::vector<std::string> ages{"(0-2)",   "(4-6)",   "(8-12)",  "(15-20)",
                                      "(25-32)", "(38-43)", "(48-53)", "(60-100)"},
            genders{"Male", "Female"};
        c.video([&](Mat &frame, int) {
            faceNet.setInput(dnn::blobFromImage(frame, 1, {300, 300}, {104, 117, 123}, true));
            Mat detections = faceNet.forward();
            detections = detections.reshape(1, int(detections.total() / 7));
            for (int i = 0; i < detections.rows; ++i) {
                auto p = detections.ptr<float>(i);
                if (p[2] <= .7) {
                    continue;
                }
                auto box = detectionBox(p + 3, frame.size());
                if (box.empty()) {
                    continue;
                }
                Mat blob = dnn::blobFromImage(frame(box), 1, {227, 227},
                                              {78.4263377603, 87.7689143744, 114.895847746});
                genderNet.setInput(blob);
                ageNet.setInput(blob);
                std::string result = className(genders, argmax(genderNet.forward())) + "," +
                                     className(ages, argmax(ageNet.forward()));
                rectangle(frame, box, {0, 255, 0}, 2);
                label(frame, result, box.tl() - Point(0, 10), {0, 255, 255});
            }
            c.show("age_gender", frame);
        });
        return;
    }
    auto predictor = landmarkModel(c);
    if (n == 1 || n == 2) {
        int counter = 0;
        c.video([&](Mat &frame, int) {
            auto boxes = faceBoxes(frame);
            if (boxes.empty()) {
                counter = 0;
            }
            for (auto &box : boxes) {
                auto p = landmarks(frame, box, predictor);
                if (n == 1) {
                    double width = norm(p[48] - p[54]), jaw = norm(p[3] - p[13]);
                    double mar = width > 0
                                     ? (norm(p[51] - p[57]) + norm(p[50] - p[58]) + norm(p[52] - p[56])) /
                                           (3 * width)
                                     : 0,
                           mjr = jaw > 0 ? width / jaw : 0;
                    label(frame, mar > .5 ? "laugh" : mjr > .45 ? "smile" : "normal", {50, 100});
                    hullRange(frame, p, 48, 61);
                } else {
                    double ear = (eyeRatio(p, 42) + eyeRatio(p, 36)) / 2;
                    if (ear < .3) {
                        ++counter;
                    } else {
                        counter = 0;
                    }
                    if (counter >= 48) {
                        label(frame, "DANGEROUS", {50, 200}, {0, 0, 255}, 2);
                    }
                    hullRange(frame, p, 42, 48);
                    hullRange(frame, p, 36, 42);
                    label(frame, "EAR: " + std::to_string(ear), {0, 30}, {0, 255, 0});
                }
            }
            c.show("face_application", frame);
        });
        return;
    }
    Mat a = c.read("person/image2.jpg"), b = c.read("person/image7.jpg");
    auto abox = faceBoxes(a, 1), bbox = faceBoxes(b, 1);
    require(!abox.empty() && !bbox.empty(), "Both face-swap photos must contain a face");
    auto ap = landmarks(a, abox[0], predictor), bp = landmarks(b, bbox[0], predictor);
    Mat am = faceMask(a.size(), ap), bm = faceMask(b.size(), bp), transform = similarity(ap, bp),
        warpedMask = Mat::zeros(am.size(), am.type()), warped = Mat::zeros(a.size(), a.type());
    warpAffine(bm, warpedMask, transform, a.size(), WARP_INVERSE_MAP | INTER_LINEAR, BORDER_CONSTANT);
    warpAffine(b, warped, transform, a.size(), WARP_INVERSE_MAP | INTER_LINEAR, BORDER_CONSTANT);
    Mat mask;
    cv::max(am, warpedMask, mask);
    Mat af, bf, ag, bg;
    a.convertTo(af, CV_64F);
    warped.convertTo(bf, CV_64F);
    GaussianBlur(af, ag, {111, 111}, 0);
    GaussianBlur(bf, bg, {111, 111}, 0);
    cv::max(bg, 1., bg);
    Mat weight;
    divide(ag, bg, weight);
    Mat corrected = bf.mul(weight), result = af.mul(Scalar::all(1) - mask) + corrected.mul(mask);
    result.convertTo(result, CV_8U);
    if (variant) {
        c.show("a_mask", am);
        c.show("b_mask", bm);
        c.show("warped_mask", warpedMask);
        c.show("combined_mask", mask);
        c.show("warped", warped);
        c.show("color_corrected", corrected / 255);
    }
    c.show("a", a);
    c.show("b", b);
    c.show("face_swap", result);
}
} // namespace cv40

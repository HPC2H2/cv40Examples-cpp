#pragma once
#include "cv40.hpp"
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
namespace cv40 {
inline dlib::matrix<dlib::rgb_pixel> faceImage(const Mat &a) {
    dlib::matrix<dlib::rgb_pixel> out;
    if (a.channels() == 1) {
        dlib::assign_image(out, dlib::cv_image<uchar>(a));
    } else {
        dlib::assign_image(out, dlib::cv_image<dlib::bgr_pixel>(a));
    }
    return out;
}
inline std::vector<dlib::rectangle> faceBoxes(const Mat &a, int upsample = 0) {
    auto image = faceImage(a);
    for (int i = 0; i < upsample; ++i) {
        dlib::pyramid_up(image);
    }
    auto detector = dlib::get_frontal_face_detector();
    auto boxes = detector(image);
    dlib::pyramid_down<2> pyramid;
    for (auto &r : boxes) {
        for (int i = 0; i < upsample; ++i) {
            r = pyramid.rect_down(r);
        }
    }
    return boxes;
}
inline dlib::shape_predictor landmarkModel(Context &c) {
    std::ifstream f(c.file("shape_predictor_68_face_landmarks.dat"), std::ios::binary);
    dlib::shape_predictor model;
    dlib::deserialize(model, f);
    return model;
}
inline Contour landmarks(const Mat &a, const dlib::rectangle &box, dlib::shape_predictor &predictor) {
    auto image = faceImage(a);
    auto shape = predictor(image, box);
    Contour out;
    for (unsigned long i = 0; i < shape.num_parts(); ++i) {
        out.emplace_back(int(shape.part(i).x()), int(shape.part(i).y()));
    }
    require(out.size() == 68, "Expected 68-point landmark model");
    return out;
}
inline void drawFaceBox(Mat &a, const dlib::rectangle &r) {
    rectangle(a, {int(r.left()), int(r.top())}, {int(r.right()), int(r.bottom())}, {0, 255, 0}, 2);
}
} // namespace cv40

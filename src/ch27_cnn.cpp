#include "faces.hpp"
#include <dlib/dnn.h>
namespace cv40 {
namespace cnn {
using namespace dlib;
template <long N, typename S> using con5d = con<N, 5, 5, 2, 2, S>;
template <long N, typename S> using con5 = con<N, 5, 5, 1, 1, S>;
template <typename S>
using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, S>>>>>>>>>;
template <typename S> using rcon5 = relu<affine<con5<45, S>>>;
using Network =
    loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
} // namespace cnn
void cnnFaceDemo(Context &c) {
    std::ifstream f(c.file("mmod_human_face_detector.dat"), std::ios::binary);
    cnn::Network net;
    dlib::deserialize(net, f);
    Mat a = c.read("people.jpg");
    auto image = faceImage(a);
    dlib::pyramid_up(image);
    auto detections = net(image);
    std::cout << "faces=" << detections.size() << '\n';
    dlib::pyramid_down<2> pyramid;
    for (auto &d : detections) {
        dlib::rectangle r = pyramid.rect_down(d.rect);
        drawFaceBox(a, r);
        std::cout << "confidence=" << d.detection_confidence << '\n';
    }
    c.show("cnn_faces", a);
}
} // namespace cv40

#include "cv40.hpp"
namespace cv40 {
void chapter23(Context &context, int exampleNumber, int) {
    Mat image;
    if (exampleNumber == 1 || exampleNumber == 6) {
        image = Mat(3, 3, CV_8U, Scalar(100));
    } else if (exampleNumber == 5) {
        image = Mat(4, 4, CV_8U);
        randu(image, 0, 256);
    } else {
        image = context.read("lena.bmp");
        resize(image, image, {3, 3});
    }
    std::cout << "original=" << image << '\n';
    auto blob = [&](double scale, Size size, Scalar mean = Scalar(), bool swap = false, bool crop = false,
                    int depth = CV_32F) {
        Mat blobData = dnn::blobFromImage(image, scale, size, mean, swap, crop, depth);
        std::cout << "NCHW=";
        for (int i = 0; i < blobData.dims; ++i) {
            std::cout << blobData.size[i] << ' ';
        }
        std::cout << " depth=" << blobData.depth() << '\n' << blobData.reshape(1, 1) << '\n';
    };
    if (exampleNumber == 1) {
        blob(1, {});
        blob(.1, {});
    } else if (exampleNumber == 2) {
        blob(1, {3, 3});
        blob(1, {4, 3});
    } else if (exampleNumber == 3) {
        blob(1, {3, 3});
        blob(1, {3, 3}, {10, 20, 50});
    } else if (exampleNumber == 4) {
        blob(1, {3, 3});
        blob(1, {3, 3}, {}, true);
    } else if (exampleNumber == 5) {
        for (bool crop : {true, false}) {
            blob(1, {4, 2}, {}, true, crop);
            blob(1, {2, 4}, {}, true, crop);
        }
    } else {
        blob(1, {3, 3});
        blob(1, {3, 3}, {}, false, false, CV_8U);
    }
}
} // namespace cv40

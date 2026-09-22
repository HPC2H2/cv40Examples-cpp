#include "learning.hpp"
namespace cv40 {
static Mat deskew(const Mat &image) {
    auto imageMoments = moments(image);
    if (std::abs(imageMoments.mu02) < 1e-2) {
        return image.clone();
    }
    double skew = imageMoments.mu11 / imageMoments.mu02;
    Mat transform = (Mat_<float>(2, 3) << 1, float(skew), float(-.5 * image.rows * skew), 0, 1, 0), out;
    warpAffine(image, out, transform, image.size(), WARP_INVERSE_MAP | INTER_LINEAR);
    return out;
}
static Mat digitHog(const Mat &image) {
    Mat gx, gy, magnitude, angle;
    Sobel(image, gx, CV_32F, 1, 0);
    Sobel(image, gy, CV_32F, 0, 1);
    cartToPolar(gx, gy, magnitude, angle);
    Mat histogram = Mat::zeros(1, 64, CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int cell = (x >= 10 ? 2 : 0) + (y >= 10 ? 1 : 0),
                bin = std::min(15, int(16 * angle.at<float>(y, x) / (2 * CV_PI)));
            histogram.at<float>(cell * 16 + bin) += magnitude.at<float>(y, x);
        }
    }
    return histogram;
}
void chapter18(Context &context, int exampleNumber, int) {
    if (exampleNumber == 1 || exampleNumber == 2) {
        Mat image = context.read(exampleNumber == 1 ? "rotatex.png" : "number2.bmp", IMREAD_GRAYSCALE);
        context.show("original", image);
        if (exampleNumber == 1) {
            Mat result = deskew(image);
            context.show("deskew", result);
            context.save("re.bmp", result);
        } else {
            std::cout << digitHog(image) << '\n';
        }
        return;
    }
    Mat train, test, trainLabels, testLabels;
    for (int digit = 0; digit < 10; ++digit) {
        auto files = imageFiles(context.chapterDir() / "data" / std::to_string(digit));
        require(files.size() >= 10, "Expected ten digit images per class");
        for (std::size_t i = 0; i < files.size(); ++i) {
            Mat image = readImage(files[i], IMREAD_GRAYSCALE);
            resize(image, image, {20, 20});
            Mat row = digitHog(deskew(image));
            if (i < 8) {
                train.push_back(row);
                trainLabels.push_back(digit);
            } else {
                test.push_back(row);
                testLabels.push_back(digit);
            }
        }
    }
    std::cout << "train=" << train.size() << " labels=" << trainLabels.size() << " test=" << test.size()
              << '\n';
    if (exampleNumber == 3) {
        return;
    }
    auto svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(train, ml::ROW_SAMPLE, trainLabels);
    Mat results;
    svm->predict(test, results);
    printAccuracy(results, testLabels);
}
} // namespace cv40

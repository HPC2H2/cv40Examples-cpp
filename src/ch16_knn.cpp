#include "learning.hpp"
namespace cv40 {
void chapter16(Context &context, int exampleNumber, int) {
    Mat train, test, trainLabels, testLabels;
    if (exampleNumber == 1) {
        Mat digits = context.read("digits.png", IMREAD_GRAYSCALE);
        require(digits.size() == Size(2000, 1000), "Expected OpenCV 5000-digit sheet");
        for (int row = 0; row < 50; ++row) {
            for (int col = 0; col < 100; ++col) {
                Mat item = flattenToFloat(digits(Rect(col * 20, row * 20, 20, 20)).clone());
                if (col < 50) {
                    train.push_back(item);
                    trainLabels.push_back(float(row / 5));
                } else {
                    test.push_back(item);
                    testLabels.push_back(float(row / 5));
                }
            }
        }
    } else {
        std::ifstream f(context.file("letter-recognition.data"));
        std::string line;
        Mat data, labels;
        while (std::getline(f, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream ss(line);
            char ch;
            ss >> ch;
            Mat row(1, 16, CV_32F);
            for (int i = 0; i < 16; ++i) {
                require(bool(ss >> row.at<float>(i)), "Invalid letter-recognition row");
            }
            data.push_back(row);
            labels.push_back(float(ch - 'A'));
        }
        require(data.rows >= 2, "Empty letter dataset");
        int split = data.rows / 2;
        train = data.rowRange(0, split);
        test = data.rowRange(split, data.rows);
        trainLabels = labels.rowRange(0, split);
        testLabels = labels.rowRange(split, labels.rows);
    }
    auto knn = ml::KNearest::create();
    knn->train(train, ml::ROW_SAMPLE, trainLabels);
    Mat result;
    knn->findNearest(test, 5, result);
    printAccuracy(result, testLabels);
}
} // namespace cv40

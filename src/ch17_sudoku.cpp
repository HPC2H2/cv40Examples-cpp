#include "learning.hpp"
namespace cv40 {
static Mat examplePuzzle() {
    return (Mat_<int>(9, 9) << 4, 0, 6, 0, 0, 0, 0, 9, 0, 0, 0, 0, 3, 1, 0, 0, 0, 6, 1, 0, 8, 0, 0, 7, 0, 0,
            0, 0, 8, 0, 0, 4, 0, 6, 0, 0, 0, 6, 0, 7, 0, 3, 0, 2, 0, 0, 0, 7, 0, 9, 0, 0, 8, 0, 0, 0, 0, 8, 0,
            0, 2, 0, 3, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 7);
}
static Ptr<ml::KNearest> trainDigitClassifier(Context &context) {
    Mat train, test, trainLabels, testLabels;
    for (int digit = 1; digit <= 9; ++digit) {
        auto files = imageFiles(context.chapterDir() / "template" / std::to_string(digit));
        require(files.size() >= 10, "Expected ten Sudoku templates per digit");
        for (std::size_t i = 0; i < files.size(); ++i) {
            Mat digitImage = readImage(files[i], IMREAD_GRAYSCALE);
            resize(digitImage, digitImage, {15, 20});
            adaptiveThreshold(digitImage, digitImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
            Mat row = flattenToFloat(digitImage);
            if (i < 8) {
                train.push_back(row);
                trainLabels.push_back(float(digit));
            } else {
                test.push_back(row);
                testLabels.push_back(float(digit));
            }
        }
    }
    auto model = ml::KNearest::create();
    model->train(train, ml::ROW_SAMPLE, trainLabels);
    Mat results;
    model->findNearest(test, 5, results);
    printAccuracy(results, testLabels);
    return model;
}
// RETR_TREE 中，大网格的直接子轮廓是单元格，单元格内的子轮廓是数字。
static Contours locateDigits(Context &context, const Mat &image, Mat &marked) {
    Mat thresholded = binary(image, 200, THRESH_BINARY_INV);
    dilate(thresholded, thresholded, getStructuringElement(MORPH_CROSS, {5, 5}));
    Contours outlines;
    std::vector<Vec4i> hierarchy;
    findContours(thresholded, outlines, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    require(!outlines.empty(), "No Sudoku grid");
    const int outer = int(std::max_element(outlines.begin(), outlines.end(),
                                           [](const auto &left, const auto &right) {
                                               return contourArea(left) < contourArea(right);
                                           }) -
                          outlines.begin());
    Contours digits;
    Mat boxes = image.clone();
    marked = image.clone();
    for (std::size_t index = 0; index < hierarchy.size(); ++index) {
        const int parent = hierarchy[index][3];
        const int firstChild = hierarchy[index][2];
        if (parent != outer) {
            continue;
        }
        drawContours(boxes, outlines, int(index), {0, 0, 255});
        if (firstChild >= 0) {
            const auto &digit = outlines[firstChild];
            digits.push_back(digit);
            rectangle(marked, boundingRect(digit), {0, 0, 255}, 2);
        }
    }
    context.show("boxes", boxes);
    context.show("numbers", marked);
    return digits;
}

static Mat recognizeBoard(Context &context, const Mat &image, const Contours &digits, Mat &marked) {
    auto classifier = trainDigitClassifier(context);
    Mat board = Mat::zeros(9, 9, CV_32S);
    Mat thresholded;
    adaptiveThreshold(gray(image), thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
    for (const auto &outline : digits) {
        const Rect bounds = boundingRect(outline);
        Mat digitImage;
        resize(thresholded(bounds), digitImage, {15, 20});
        Mat prediction;
        classifier->findNearest(flattenToFloat(digitImage), 5, prediction);
        const int digit = int(prediction.at<float>(0));
        // 教材图片已正对网格；用数字左上角映射到 9×9 盘面。
        const int row = std::min(8, bounds.y * 9 / image.rows);
        const int column = std::min(8, bounds.x * 9 / image.cols);
        board.at<int>(row, column) = digit;
        label(marked, std::to_string(digit), {bounds.x + bounds.width - 6, bounds.y + bounds.height - 15},
              {255, 0, 0}, 1);
    }
    context.show("recognized", marked);
    std::cout << board << '\n';
    return board;
}

static void drawSolution(Mat &image, const Mat &board) {
    for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
            label(image, std::to_string(board.at<int>(row, column)),
                  {column * image.cols / 9, row * image.rows / 9 + 40}, {0, 0, 255}, 1);
        }
    }
}

void chapter17(Context &context, int exampleNumber, int) {
    if (exampleNumber == 2) {
        trainDigitClassifier(context);
        return;
    }
    if (exampleNumber == 4) {
        Mat board = examplePuzzle();
        std::cout << board << '\n';
        require(solveSudoku(board), "Sudoku has no solution");
        std::cout << board << '\n';
        return;
    }
    Mat image = context.read(exampleNumber == 1 ? "x.jpg" : "xt.jpg");
    Mat board = examplePuzzle();
    context.show("original", image);
    if (exampleNumber != 5) {
        Mat marked;
        const Contours digits = locateDigits(context, image, marked);
        if (exampleNumber == 1) {
            return;
        }
        board = recognizeBoard(context, image, digits, marked);
        if (exampleNumber == 3) {
            return;
        }
    }
    require(solveSudoku(board),
            "Recognized Sudoku is inconsistent or has no solution; inspect recognized output");
    std::cout << "solution=" << board << '\n';
    drawSolution(image, board);
    context.show("solution", image);
    if (exampleNumber == 5) {
        context.save("xxx.bmp", image);
    }
}
} // namespace cv40

#include "cv40.hpp"
namespace cv40 {
void chapter15(Context &, int exampleNumber, int) {
    Mat data = exampleNumber == 1 ? (Mat_<float>(6, 2) << 5, 6, 9, 8, 3, 8, 99, 94, 89, 91, 92, 96)
                                  : (Mat_<float>(6, 2) << 6, 3, 4, 5, 9, 8, 12, 12, 15, 13, 18, 17);
    Mat labels = (Mat_<int>(6, 1) << 0, 0, 0, 1, 1, 1),
        query = exampleNumber == 1 ? (Mat_<float>(1, 2) << 31, 28) : (Mat_<float>(1, 2) << 12, 18), results;
    if (exampleNumber == 1) {
        auto model = ml::KNearest::create();
        Mat floatLabels, neighbors, squaredDistances;
        labels.convertTo(floatLabels, CV_32F);
        model->train(data, ml::ROW_SAMPLE, floatLabels);
        model->findNearest(query, 3, results, neighbors, squaredDistances);
        std::cout << "class=" << results << "\nneighbors=" << neighbors
                  << "\nsquared distances=" << squaredDistances << '\n';
    } else {
        auto model = ml::SVM::create();
        model->train(data, ml::ROW_SAMPLE, labels);
        model->predict(query, results);
        std::cout << "class=" << results << '\n';
    }
}
} // namespace cv40

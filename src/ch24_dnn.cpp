#include "models.hpp"
namespace cv40 {
namespace {
void estimatePose(Context &context) {
    auto network = tensorflow(context, "graph_opt.pb");
    const std::vector<std::pair<int, int>> skeletonEdges{
        {1, 2},  {1, 5},   {2, 3},   {3, 4}, {5, 6},  {6, 7},   {1, 8},  {8, 9},  {9, 10},
        {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
    context.video([&](Mat &frame, int) {
        network.setInput(dnn::blobFromImage(frame, 1, {368, 368}, {127.5, 127.5, 127.5}, true));
        Mat output = network.forward();
        require(output.dims == 4 && output.size[1] >= 19, "Unexpected pose output");
        std::array<std::optional<Point>, 19> points;
        for (int i = 0; i < 19; ++i) {
            Mat heatmap(output.size[2], output.size[3], CV_32F, output.ptr<float>(0, i));
            double confidence;
            Point peak;
            minMaxLoc(heatmap, nullptr, &confidence, nullptr, &peak);
            if (confidence > .2) {
                points[i] = Point(peak.x * frame.cols / heatmap.cols, peak.y * frame.rows / heatmap.rows);
            }
        }
        for (auto [startJoint, endJoint] : skeletonEdges) {
            if (points[startJoint] && points[endJoint]) {
                line(frame, *points[startJoint], *points[endJoint], {0, 255, 0}, 3);
                circle(frame, *points[startJoint], 3, {0, 0, 255}, FILLED);
                circle(frame, *points[endJoint], 3, {0, 0, 255}, FILLED);
            }
        }
        context.show("pose", frame);
    });
}

void classifyImage(Context &context) {
    auto network = caffe(context, "model/bvlc_googlenet.prototxt", "model/bvlc_googlenet.caffemodel");
    auto names = lines(context, "model/label.txt");
    Mat image = context.read("tower.jpg");
    network.setInput(dnn::blobFromImage(image, 1, {224, 224}, {104, 117, 123}));
    Mat output = network.forward();
    int index = argmax(output);
    std::string result =
        className(names, index) + ": " + std::to_string(output.reshape(1, 1).at<float>(index) * 100) + "%";
    std::cout << result << '\n';
    label(image, result, {25, 45});
    context.show("classification", image);
}

void detectWithYolo(Context &context) {
    auto weights = bytes(context.file("yolov3.weights")), cfg = bytes(context.file("yolov3.cfg"));
    auto network = dnn::readNetFromDarknet(cfg, weights);
    auto names = lines(context, "coco.names");
    Mat image = context.read("test2.jpg");
    network.setInput(dnn::blobFromImage(image, 1. / 255, {416, 416}, {}, true));
    std::vector<Mat> outputs;
    network.forward(outputs, network.getUnconnectedOutLayersNames());
    std::vector<Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    // 每行依次为中心 x/y、宽/高、objectness、各类别分数；沿用教材的类别分数阈值。
    for (auto &output : outputs) {
        for (int row = 0; row < output.rows; ++row) {
            auto detection = output.ptr<float>(row);
            Mat classScores = output.row(row).colRange(5, output.cols);
            int classId = argmax(classScores);
            float confidence = classScores.at<float>(classId);
            if (confidence > .5f) {
                int w = int(detection[2] * image.cols), h = int(detection[3] * image.rows);
                boxes.emplace_back(int(detection[0] * image.cols - w / 2.),
                                   int(detection[1] * image.rows - h / 2.), w, h);
                scores.push_back(confidence);
                classIds.push_back(classId);
            }
        }
    }
    std::vector<int> keep;
    dnn::NMSBoxes(boxes, scores, .5f, .4f, keep);
    for (int i : keep) {
        auto box = boxes[i] & Rect({}, image.size());
        rectangle(image, box, {0, 255, 0}, 2);
        label(image, className(names, classIds[i]) + " " + std::to_string(scores[i]),
              box.tl() + Point(0, 25));
    }
    context.show("YOLO", image);
}

void detectWithSsd(Context &context) {
    auto network = caffe(context, "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel");
    auto names = lines(context, "object_detection_classes_pascal_voc.txt");
    Mat image = context.read("test2.jpg");
    network.setInput(dnn::blobFromImage(image, .007843, {300, 300}, Scalar::all(127.5)));
    Mat output = network.forward();
    require(output.total() % 7 == 0, "Unexpected SSD output");
    // SSD 每条结果有 7 个值：[image_id, class_id, confidence, left, top, right, bottom]。
    Mat detections = output.reshape(1, int(output.total() / 7));
    for (int i = 0; i < detections.rows; ++i) {
        auto detection = detections.ptr<float>(i);
        if (detection[2] > .3) {
            std::cout << "detection=" << className(names, int(detection[1])) << " confidence=" << detection[2]
                      << '\n';
            auto box = detectionBox(detection + 3, image.size());
            rectangle(image, box, {0, 255, 0}, 2);
            label(image, className(names, int(detection[1])) + " " + std::to_string(detection[2]),
                  box.tl() + Point(0, 25));
        }
    }
    context.show("SSD", image);
}

void segmentSemantics(Context &context) {
    auto network = caffe(context, "fcn8s-heavy-pascal.prototxt", "fcn8s-heavy-pascal.caffemodel");
    auto names = lines(context, "object_detection_classes_pascal_voc.txt");
    Mat image = context.read("a.jpg");
    network.setInput(dnn::blobFromImage(image, 1, image.size()));
    // 网络输出为 NCHW：批次、类别、行、列；逐像素取最高分的类别。
    Mat score = network.forward();
    require(score.dims == 4, "Unexpected segmentation output");
    std::vector<Vec3b> colors(score.size[1]);
    for (auto &v : colors) {
        v = Vec3b(uchar(theRNG().uniform(0, 256)), uchar(theRNG().uniform(0, 256)),
                  uchar(theRNG().uniform(0, 256)));
    }
    colors[0] = {0, 0, 0};
    Mat chart(int(colors.size()) * 30, 240, CV_8UC3);
    for (int i = 0; i < int(colors.size()); ++i) {
        Mat row = chart.rowRange(i * 30, (i + 1) * 30);
        row.setTo(Scalar(colors[i]));
        label(row, className(names, i), {0, 20}, Scalar::all(255), .5);
    }
    context.show("legend", chart);
    Mat mask(score.size[2], score.size[3], CV_8UC3);
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            int best = 0;
            for (int k = 1; k < score.size[1]; ++k) {
                if (score.ptr<float>(0, k, y)[x] > score.ptr<float>(0, best, y)[x]) {
                    best = k;
                }
            }
            mask.at<Vec3b>(y, x) = colors[best];
        }
    }
    resize(mask, mask, image.size(), 0, 0, INTER_NEAREST);
    Mat result;
    addWeighted(image, .2, mask, .8, 0, result);
    context.show("semantic", result);
}

void segmentInstances(Context &context) {
    auto network = tensorflow(context, "dnn/frozen_inference_graph.pb",
                              "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
    auto names = lines(context, "object_detection_classes_coco.txt");
    Mat image = context.read("e.jpg"), background(image.size(), CV_8UC3, Scalar(100, 100, 0));
    network.setInput(dnn::blobFromImage(image, 1, {}, {}, true));
    std::vector<Mat> output;
    network.forward(output, std::vector<String>{"detection_out_final", "detection_masks"});
    require(output.size() == 2 && output[1].dims == 4, "Unexpected Mask R-CNN outputs");
    Mat detections = output[0].reshape(1, int(output[0].total() / 7));
    for (int i = 0; i < detections.rows; ++i) {
        auto detection = detections.ptr<float>(i);
        int classId = int(detection[1]);
        if (detection[2] <= .5 || classId < 0 || classId >= output[1].size[1] || i >= output[1].size[0]) {
            continue;
        }
        Rect box = detectionBox(detection + 3, image.size());
        if (box.empty()) {
            continue;
        }
        Mat mask(output[1].size[2], output[1].size[3], CV_32F, output[1].ptr<float>(i, classId));
        resize(mask, mask, box.size());
        Mat region = background(box);
        region.setTo(Scalar(theRNG().uniform(0, 256), theRNG().uniform(0, 256), theRNG().uniform(0, 256)),
                     mask > .5);
        label(background, className(names, classId), box.tl() + Point(0, 25), Scalar::all(255));
    }
    Mat result;
    addWeighted(image, .2, background, .8, 0, result);
    context.show("original", image);
    context.show("instances", result);
}

void transferStyle(Context &context) {
    // Torch has only a filename loader; stage the user-supplied model in an ASCII path.
    auto source = context.file("model/eccv16/starry_night.t7");
    auto staged = fs::path(CV40_DEFAULT_ROOT).parent_path() / "build/model-cache/starry_night.t7";
    fs::create_directories(staged.parent_path());
    fs::copy_file(source, staged, fs::copy_options::overwrite_existing);
    auto network = dnn::readNetFromTorch(staged.string());
    Mat image = context.read("tute.jpg");
    network.setInput(dnn::blobFromImage(image, 1, image.size()));
    Mat output = network.forward();
    require(output.dims == 4 && output.size[1] == 3, "Unexpected style transfer output");
    normalize(output, output, 0, 1, NORM_MINMAX);
    std::vector<Mat> planes;
    for (int i = 0; i < 3; ++i) {
        planes.emplace_back(output.size[2], output.size[3], CV_32F, output.ptr<float>(0, i));
    }
    Mat result;
    merge(planes, result);
    context.show("original", image);
    context.show("style", result);
}
} // namespace

void chapter24(Context &context, int exampleNumber, int) {
    switch (exampleNumber) {
    case 1:
        classifyImage(context);
        break;
    case 2:
        detectWithYolo(context);
        break;
    case 3:
        detectWithSsd(context);
        break;
    case 4:
        segmentSemantics(context);
        break;
    case 5:
        segmentInstances(context);
        break;
    case 6:
        transferStyle(context);
        break;
    case 7:
        estimatePose(context);
        break;
    default:
        throw std::runtime_error("Unknown DNN example");
    }
}
} // namespace cv40

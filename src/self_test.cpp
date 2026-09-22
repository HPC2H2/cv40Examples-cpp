#include "cv40.hpp"
#include <chrono>

namespace cv40 {
namespace {
void expectFailure(const std::function<void()> &operation, const std::string &description) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, description);
}

void testModularAddition() {
    Mat left = (Mat_<uchar>(1, 5) << 0, 1, 127, 200, 255);
    Mat right = (Mat_<uchar>(1, 5) << 255, 255, 129, 100, 1);
    Mat expected = (Mat_<uchar>(1, 5) << 255, 0, 0, 44, 0);
    require(norm(wrapAdd(left, right), expected, NORM_INF) == 0, "uint8 modular addition");
    // ROI 的步长可能大于行宽，不能把整幅矩阵误当连续内存。
    Mat color(4, 7, CV_8UC3, Scalar(200, 127, 255));
    Mat roi = color(Rect(2, 1, 3, 2));
    Mat doubled(roi.size(), CV_8UC3, Scalar(144, 254, 254));
    require(norm(wrapAdd(roi, roi), doubled, NORM_INF) == 0, "Strided multichannel addition");
    expectFailure([&] { wrapAdd(left, Mat(2, 5, CV_8U)); }, "Reject incompatible image sizes");
}

void testWatermarks() {
    Mat original = (Mat_<uchar>(1, 5) << 0, 1, 127, 200, 255);
    Mat key = (Mat_<uchar>(1, 5) << 19, 254, 64, 200, 7);
    Mat encrypted, decoded;
    bitwise_xor(original, key, encrypted);
    bitwise_xor(encrypted, key, decoded);
    require(norm(original, decoded, NORM_INF) == 0, "XOR lossless round trip");
    Mat bits = (Mat_<uchar>(1, 5) << 1, 0, 1, 0, 1);
    Mat cleared, embedded, extracted;
    bitwise_and(original, Scalar(254), cleared);
    bitwise_or(cleared, bits, embedded);
    bitwise_and(embedded, Scalar(1), extracted);
    require(norm(extracted, bits, NORM_INF) == 0 && norm(original, embedded, NORM_INF) <= 1,
            "LSB watermark fidelity");
}

void testImageFeatures() {
    Mat first = (Mat_<uchar>(1, 4) << 0, 1, 1, 1);
    Mat opposite = (Mat_<uchar>(1, 4) << 1, 0, 0, 0);
    require(hamming(first, first) == 0 && hamming(first, opposite) == 4, "Hamming distance");
    Mat flat(16, 16, CV_8UC3, Scalar::all(90));
    require(countNonZero(hashImage(flat)) == 0, "Constant average hash");
    Mat pattern = Mat::zeros(30, 30, CV_8U);
    rectangle(pattern, {3, 3, 8, 8}, Scalar(255), FILLED);
    rectangle(pattern, {18, 18, 6, 6}, Scalar(255), FILLED);
    require(contours(pattern).size() == 2, "Contour object count");
}

Mat knownPuzzle() {
    return (Mat_<int>(9, 9) << 5, 3, 0, 0, 7, 0, 0, 0, 0, 6, 0, 0, 1, 9, 5, 0, 0, 0, 0, 9, 8, 0, 0, 0, 0, 6,
            0, 8, 0, 0, 0, 6, 0, 0, 0, 3, 4, 0, 0, 8, 0, 3, 0, 0, 1, 7, 0, 0, 0, 2, 0, 0, 0, 6, 0, 6, 0, 0, 0,
            0, 2, 8, 0, 0, 0, 0, 4, 1, 9, 0, 0, 5, 0, 0, 0, 0, 8, 0, 0, 7, 9);
}

void checkSolvedBoard(const Mat &board, const Mat &clues) {
    const std::vector<int> digits{1, 2, 3, 4, 5, 6, 7, 8, 9};
    // 用排序检查行、列、宫，独立于求解器所用的位掩码约束。
    for (int index = 0; index < 9; ++index) {
        std::vector<int> row, column, box;
        for (int offset = 0; offset < 9; ++offset) {
            row.push_back(board.at<int>(index, offset));
            column.push_back(board.at<int>(offset, index));
            box.push_back(board.at<int>(index / 3 * 3 + offset / 3, index % 3 * 3 + offset % 3));
            if (clues.at<int>(index, offset)) {
                require(clues.at<int>(index, offset) == board.at<int>(index, offset), "Sudoku clue changed");
            }
        }
        std::sort(row.begin(), row.end());
        std::sort(column.begin(), column.end());
        std::sort(box.begin(), box.end());
        require(row == digits && column == digits && box == digits, "Invalid Sudoku solution");
    }
}

void testSudoku() {
    Mat clues = knownPuzzle();
    Mat board = clues.clone();
    require(solveSudoku(board), "Known Sudoku must be solvable");
    require(board.at<int>(0, 2) == 4 && board.at<int>(8, 0) == 3, "Known Sudoku solution");
    checkSolvedBoard(board, clues);
    Mat solved = board.clone();
    require(solveSudoku(board) && norm(board, solved, NORM_INF) == 0, "Solved Sudoku must stay unchanged");
    for (int badDigit : {5, -1, 10}) {
        Mat invalid = clues.clone();
        invalid.at<int>(0, 2) = badDigit;
        const Mat original = invalid.clone();
        require(!solveSudoku(invalid), "Reject conflicting or out-of-range Sudoku");
        require(norm(invalid, original, NORM_INF) == 0, "Invalid Sudoku must stay unchanged");
    }
    // 1 不与当前行/列/宫的线索直接冲突，但会使这道唯一解数独无解。
    Mat impossible = clues.clone();
    impossible.at<int>(0, 2) = 1;
    const Mat original = impossible.clone();
    require(!solveSudoku(impossible), "Reject Sudoku requiring unsuccessful search");
    require(norm(impossible, original, NORM_INF) == 0, "Backtracking must restore unsolvable board");
    Mat wrongType(9, 9, CV_8U, Scalar(0));
    expectFailure([&] { solveSudoku(wrongType); }, "Reject non-integer Sudoku storage");
}

// 仅清理本次成功创建的专用目录，不与其他同时运行的自检共用文件。
class TemporaryDirectory {
  public:
    fs::path path;
    TemporaryDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path = fs::temp_directory_path() /
                   fs::u8path(u8"cv40_中文自检_" + std::to_string(stamp) + "_" + std::to_string(attempt));
            if (fs::create_directory(path)) {
                return;
            }
        }
        throw std::runtime_error("Cannot create unique self-test directory");
    }
    ~TemporaryDirectory() {
        std::error_code error;
        for (const auto &entry : fs::directory_iterator(path, error)) {
            fs::remove(entry.path(), error);
        }
        fs::remove(path, error);
    }
};

void testUnicodeAndReplay() {
    TemporaryDirectory temporary;
    Mat dark(16, 16, CV_8UC3, Scalar::all(20));
    Mat bright(16, 16, CV_8UC3, Scalar::all(200));
    writeImage(temporary.path / fs::u8path(u8"02_亮.png"), bright);
    writeImage(temporary.path / fs::u8path(u8"01_暗.png"), dark);
    require(norm(dark, readImage(temporary.path / fs::u8path(u8"01_暗.png")), NORM_INF) == 0,
            "Unicode image path round trip");
    Context context;
    context.headless = true;
    context.frames = temporary.path;
    std::vector<int> intensities;
    context.video([&](Mat &frame, int index) {
        require(index == int(intensities.size()), "Replay frame indices");
        intensities.push_back(frame.at<Vec3b>(0, 0)[0]);
    });
    require(intensities == std::vector<int>({20, 200}), "Replay must sort frame filenames");
    context.maxFrames = 1;
    int processed = 0;
    context.video([&](Mat &, int) { ++processed; });
    require(processed == 1, "Replay frame limit");
    context.maxFrames = -1;
    expectFailure([&] { context.video([](Mat &, int) {}); }, "Reject negative frame limit");
    context.maxFrames = 0;
    context.frames /= "missing";
    expectFailure([&] { context.video([](Mat &, int) {}); }, "Reject missing replay images");
    expectFailure([&] { readImage(temporary.path / "missing.png"); }, "Reject missing image");
}
} // namespace

void selfTest() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"modular addition", testModularAddition},
        {"XOR and LSB watermark", testWatermarks},
        {"hash and contours", testImageFeatures},
        {"Sudoku constraints and rollback", testSudoku},
        {"Unicode I/O and frame replay", testUnicodeAndReplay}};
    for (const auto &[name, test] : tests) {
        test();
        std::cout << "PASS self-test: " << name << '\n';
    }
}
} // namespace cv40

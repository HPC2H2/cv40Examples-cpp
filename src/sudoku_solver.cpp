#include "cv40.hpp"

namespace cv40 {
namespace {
constexpr int kGridSize = 9;
constexpr int kBoxSize = 3;
// 第 1～9 位代表数字 1～9；第 0 位不用，因为 0 表示空格。
constexpr int kAllDigits = 0x3fe;

int boxIndex(int row, int column) {
    return row / kBoxSize * kBoxSize + column / kBoxSize;
}

struct CandidateCell {
    int row = -1;
    int column = -1;
    int digits = 0;
    int count = kGridSize + 1;
};

class SudokuSearch {
  public:
    explicit SudokuSearch(Mat &board) : board_(board) {}

    bool solve() {
        return initializeConstraints() && search();
    }

  private:
    Mat &board_;
    std::array<int, kGridSize> rowDigits_{};
    std::array<int, kGridSize> columnDigits_{};
    std::array<int, kGridSize> boxDigits_{};

    int availableDigits(int row, int column) const {
        return kAllDigits & ~(rowDigits_[row] | columnDigits_[column] | boxDigits_[boxIndex(row, column)]);
    }

    bool initializeConstraints() {
        for (int row = 0; row < kGridSize; ++row) {
            for (int column = 0; column < kGridSize; ++column) {
                const int digit = board_.at<int>(row, column);
                if (digit < 0 || digit > kGridSize) {
                    return false;
                }
                if (digit == 0) {
                    continue;
                }
                const int bit = 1 << digit;
                if (!(availableDigits(row, column) & bit)) {
                    return false;
                }
                rowDigits_[row] |= bit;
                columnDigits_[column] |= bit;
                boxDigits_[boxIndex(row, column)] |= bit;
            }
        }
        return true;
    }

    CandidateCell chooseNextCell() const {
        CandidateCell best;
        for (int row = 0; row < kGridSize; ++row) {
            for (int column = 0; column < kGridSize; ++column) {
                if (board_.at<int>(row, column) != 0) {
                    continue;
                }
                const int digits = availableDigits(row, column);
                int count = 0;
                for (int digit = 1; digit <= kGridSize; ++digit) {
                    count += (digits >> digit) & 1;
                }
                if (count < best.count) {
                    best = {row, column, digits, count};
                }
                if (count == 0) {
                    return best;
                }
            }
        }
        return best;
    }

    bool search() {
        // 优先尝试候选数字最少的空格，减少回溯分支；没有空格即已求解。
        const CandidateCell cell = chooseNextCell();
        if (cell.row < 0) {
            return true;
        }
        const int box = boxIndex(cell.row, cell.column);
        for (int digit = 1; digit <= kGridSize; ++digit) {
            const int bit = 1 << digit;
            if (!(cell.digits & bit)) {
                continue;
            }
            board_.at<int>(cell.row, cell.column) = digit;
            rowDigits_[cell.row] |= bit;
            columnDigits_[cell.column] |= bit;
            boxDigits_[box] |= bit;
            if (search()) {
                return true;
            }
            // 失败分支必须恢复盘面和约束；无解时调用者仍能查看原始识别结果。
            rowDigits_[cell.row] &= ~bit;
            columnDigits_[cell.column] &= ~bit;
            boxDigits_[box] &= ~bit;
            board_.at<int>(cell.row, cell.column) = 0;
        }
        return false;
    }
};
} // namespace

bool solveSudoku(Mat &board) {
    require(board.type() == CV_32S && board.size() == Size(kGridSize, kGridSize),
            "Expected 9x9 integer Sudoku");
    return SudokuSearch(board).solve();
}
} // namespace cv40

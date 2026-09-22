#include "cv40.hpp"
#include "my_modules.hpp"
namespace cv40 {
template <class T> void print(const T &items) {
    std::cout << '[';
    for (const auto &v : items) {
        std::cout << v << ' ';
    }
    std::cout << "]\n";
}
void chapter2(Context &c, int n, int) {
    std::vector<int> a{5, 8, 7, 3, 9, 6, 1, 0, 2};
    auto slice = [&](int begin, int end, int step = 1) {
        std::vector<int> r;
        for (int i = begin; i < end; i += step) {
            r.push_back(a.at(i));
        }
        print(r);
    };
    switch (n) {
    case 0:
    case 26:
        std::cout << "x2(9)=" << my_modules::x2(9) << " x10(9)=" << my_modules::x10(9) << '\n';
        break;
    case 1:
        std::cout << 5 * 5 * 5 << '\n';
        break;
    case 2:
        print(std::vector<int>{3, 4, 5, 6, 7, 8, 9, 666, 99, 0});
        print(std::vector<int>{});
        break;
    case 3:
        std::cout << a[2] << ' ' << a[a.size() - 2] << '\n';
        a[2] = 666;
        print(a);
        break;
    case 4:
        slice(2, 6);
        slice(2, 9, 2);
        slice(0, 6);
        slice(6, 9);
        slice(6, 8);
        break;
    case 5:
        a = {5, 8, 7, 3};
        a.push_back(666);
        print(a);
        break;
    case 6:
        a = {5, 8, 6, 2};
        a.erase(a.begin() + 2);
        print(a);
        break;
    case 7: {
        const std::array<int, 9> tuple{5, 8, 7, 3, 9, 6, 1, 0, 2};
        std::cout << tuple[2] << ' ' << tuple[7] << '\n';
        slice(2, 6);
        slice(2, 6, 2);
        slice(0, 6);
        slice(6, 9);
        slice(6, 8);
        break;
    }
    case 8: {
        const std::vector<int> x{1, 2, 3}, y{4, 5, 6};
        auto joined = x;
        joined.insert(joined.end(), y.begin(), y.end());
        print(joined);
        std::cout << joined.size() << '\n';
        std::vector<int> repeated;
        for (int i = 0; i < 3; ++i) {
            repeated.insert(repeated.end(), x.begin(), x.end());
        }
        print(repeated);
        break;
    }
    case 9:
    case 10: {
        std::map<std::string, int> d{{u8"李立宗", 66}, {u8"刘能", 88}, {u8"赵四", 99}};
        if (n == 10) {
            d[u8"李立宗"] = 90;
            d[u8"小明"] = 100;
            d.erase(u8"李立宗");
        }
        for (auto &[k, v] : d) {
            std::cout << k << '=' << v << '\n';
        }
        if (n == 9) {
            std::cout << d.at(u8"李立宗") << '\n';
        }
        break;
    }
    case 11:
    case 12:
    case 13: {
        double score = c.ask(u8"输入成绩", 85);
        if (score > 90) {
            std::cout << u8"A级\n";
        } else if (n == 12) {
            std::cout << u8"加油\n";
        } else if (n == 13) {
            std::cout << (score > 80 ? "B" : score > 70 ? "C" : score >= 60 ? "D" : "E") << '\n';
        }
        break;
    }
    case 14: {
        double x = c.ask("a", 2), y = c.ask("b", 3);
        std::cout << (x > y ? x : y) << '\n';
        break;
    }
    case 15:
        print(std::vector<int>{7, 9, 8});
        print(std::array<int, 3>{1, 7, 1});
        print(std::string("PYTHON"));
        break;
    case 16: {
        std::vector<std::string> v{"Python", u8"人工智能", u8"大数据"};
        for (std::size_t i = 0; i < v.size(); ++i) {
            std::cout << i << ' ' << v[i] << '\n';
        }
        break;
    }
    case 17:
        for (int i = 1; i < 10; i += 3) {
            std::cout << i << ' ';
        }
        std::cout << '\n';
        for (int i = 1; i < 5; ++i) {
            std::cout << i << ' ';
        }
        std::cout << '\n';
        for (int i = 0; i < 5; ++i) {
            std::cout << i << ' ';
        }
        break;
    case 18:
        for (int i = 1; i < 5; ++i) {
            std::cout << i << u8" 循环\n";
        }
        break;
    case 19: {
        int i = 0;
        while (i < 5) {
            std::cout << i++ << '\n';
        }
        break;
    }
    case 20: {
        a = {7, 9, 8, 666};
        std::size_t i = 0;
        while (i < a.size()) {
            std::cout << a[i++] << '\n';
        }
        break;
    }
    case 21:
    case 22:
        for (int i : {7, 9, 8, 666, 999, 973, 985, 211}) {
            if (i == 666) {
                if (n == 21) {
                    break;
                }
                continue;
            }
            std::cout << i << '\n';
        }
        break;
    case 23: {
        a = {1, 2, 3, 0, 5};
        std::sort(a.begin(), a.end());
        std::cout << std::abs(-45) << ' ' << std::pow(2, 3) << '\n';
        print(a);
        std::cout << a.back() << ' ' << a.front() << ' ' << std::accumulate(a.begin(), a.end(), 0) << '\n';
        break;
    }
    case 24: {
        auto area1 = []() { std::cout << 3 * 3 << '\n'; };
        auto area2 = []() { return 3 * 3; };
        auto area3 = [](int r) { std::cout << r * r << '\n'; };
        auto area4 = [](int r) { return r * r; };
        area1();
        std::cout << area2() << '\n';
        area3(3);
        std::cout << area4(3) << '\n';
        break;
    }
    case 25:
        for (int i = 0; i < 7; ++i) {
            std::cout << theRNG().uniform(0, 10) << ' ';
        }
        std::cout << '\n';
        break;
    }
}
} // namespace cv40

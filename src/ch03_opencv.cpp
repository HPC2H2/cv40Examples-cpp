#include "cv40.hpp"
namespace cv40 {
void chapter3(Context &c, int n, int) {
    Mat a, r;
    if (n == 1) {
        std::cout << c.read("lenacolor.png") << '\n';
        return;
    }
    if (n == 4) {
        a = Mat::zeros(8, 8, CV_8U);
        c.show("one", a);
        std::cout << int(a.at<uchar>(0, 3)) << '\n';
        a.at<uchar>(0, 3) = 255;
        c.show("two", a);
        std::cout << int(a.at<uchar>(0, 3)) << '\n';
        return;
    }
    if (n == 11) {
        a = Mat::zeros(600, 600, CV_8U);
        a(Rect(200, 200, 200, 200)) = 255;
        c.show("m1", a);
        a /= 255;
        c.show("m2", a);
        c.show("m2x255", a * 255);
        return;
    }
    std::string name = (n == 6 || n == 7 || n == 8 || n >= 10 && n <= 14) ? "lenacolor.png"
                       : n == 9                                           ? "test.bmp"
                       : n == 15                                          ? "x.jpg"
                       : n >= 16 && n <= 18                               ? "lenaNoise.png"
                       : n == 19                                          ? "erode.bmp"
                       : n == 20                                          ? "dilation.bmp"
                                                                          : "lena.bmp";
    a = c.read(name, n == 5 ? IMREAD_GRAYSCALE : (n == 10 || n >= 19 ? IMREAD_UNCHANGED : IMREAD_COLOR));
    c.show("original", a);
    switch (n) {
    case 2:
        c.show("demo2", a);
        break;
    case 3:
        c.save("result.bmp", a);
        break;
    case 5:
        std::cout << int(a.at<uchar>(50, 90)) << '\n';
        a(Rect(80, 10, 20, 90)) = 255;
        c.show("after", a);
        std::cout << int(a.at<uchar>(50, 90)) << '\n';
        break;
    case 6:
        std::cout << a.at<Vec3b>(0, 0) << ' ' << a.at<Vec3b>(50, 0) << ' ' << a.at<Vec3b>(100, 0) << '\n';
        for (int i = 0; i < 4; ++i) {
            a(Rect(0, i * 50, 100, 50)) = i == 0   ? Scalar::all(255)
                                          : i == 1 ? Scalar::all(128)
                                          : i == 2 ? Scalar::all(0)
                                                   : Scalar(0, 0, 255);
        }
        c.show("after", a);
        break;
    case 7:
    case 8: {
        std::vector<Mat> p;
        split(a, p);
        if (n == 7) {
            c.show("b", p[0]);
            c.show("g", p[1]);
            c.show("r", p[2]);
            p[0] = Scalar(0);
            merge(p, r);
            c.show("b0", r);
            p[1] = Scalar(0);
            merge(p, r);
            c.show("b0g0", r);
        } else {
            merge(p, r);
            c.show("bgr", r);
            std::swap(p[0], p[2]);
            merge(p, r);
            c.show("rgb", r);
        }
        break;
    }
    case 9:
        resize(a, r, {int(a.cols * .9), int(a.rows * .5)});
        c.show("resize", r);
        std::cout << a.size() << " -> " << r.size() << '\n';
        break;
    case 10:
        c.show("face", a(Rect(250, 220, 100, 180)));
        break;
    case 12:
    case 13:
    case 14: {
        Mat m = Mat::zeros(a.size(), CV_8U);
        m(Rect(200, 100, 200, 300)) = 255;
        m(Rect(100, 100, 100, 400)) = 255;
        c.show("mask", m);
        if (n == 14) {
            add(a, c.read("text.png"), r, m);
        } else {
            r = masked(a, m);
        }
        c.show("result", r);
        break;
    }
    case 15:
        cvtColor(a, r, COLOR_BGR2HSV);
        inRange(r, Scalar(0, 10, 80), Scalar(33, 255, 255), r);
        c.show("mask", r);
        c.show("skin", masked(a, r));
        break;
    case 16:
        for (int k : {3, 11}) {
            blur(a, r, {k, k});
            c.show("blur" + std::to_string(k), r);
        }
        break;
    case 17:
        for (double sigma : {0., .1, 1.}) {
            GaussianBlur(a, r, {5, 5}, sigma, sigma);
            c.show("gaussian" + std::to_string(sigma), r);
        }
        break;
    case 18:
        medianBlur(a, r, 3);
        c.show("median", r);
        break;
    case 19:
        erode(a, r, Mat::ones(5, 5, CV_8U));
        c.show("erode1", r);
        erode(a, r, Mat::ones(9, 9, CV_8U), {-1, -1}, 5);
        c.show("erode5", r);
        break;
    case 20:
        dilate(a, r, Mat::ones(5, 5, CV_8U));
        c.show("dilate1", r);
        dilate(a, r, Mat::ones(5, 5, CV_8U), {-1, -1}, 9);
        c.show("dilate9", r);
        break;
    }
}
} // namespace cv40

#include "cv40.hpp"
namespace cv40 {
static Mat redMask(const Mat &a, bool alternate = false) {
    Mat hsv, x, y;
    cvtColor(a, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, alternate ? 120 : 100, alternate ? 70 : 100), Scalar(10, 255, 255), x);
    inRange(hsv, Scalar(alternate ? 170 : 160, alternate ? 120 : 100, alternate ? 70 : 100),
            Scalar(179, 255, 255), y);
    return x | y;
}
void chapter10(Context &c, int n, int) {
    Mat a, b, r;
    if (n == 12) {
        Mat background;
        c.video([&](Mat &frame, int i) {
            if (i == 0) {
                background = frame.clone();
            }
            Mat mask = redMask(frame);
            dilate(mask, mask, Mat::ones(3, 3, CV_8U));
            c.show("fore", frame);
            c.show("result", masked(background, mask) + masked(frame, 255 - mask));
        });
        return;
    }
    if (n == 1) {
        a = c.read("fore.jpg");
        r = redMask(a);
        c.show("fore", a);
        c.show("mask1", r);
        c.save("mask1.bmp", r);
        return;
    }
    if (n == 2) {
        a = c.read("mask1.bmp", IMREAD_GRAYSCALE);
        bitwise_not(a, r);
        c.show("mask1", a);
        c.show("inverse", r);
        c.show("subtract", 255 - a);
        c.show("threshold", binary(a, 127, THRESH_BINARY_INV));
        return;
    }
    if (n == 3 || n == 4) {
        a = c.read(n == 3 ? "back.jpg" : "fore.jpg");
        b = c.read(n == 3 ? "mask1.bmp" : "mask2.bmp", IMREAD_GRAYSCALE);
        c.show("image", a);
        c.show("mask", b);
        c.show("result", masked(a, b));
        return;
    }
    if (n == 5 || n == 6 || n == 8) {
        a = Mat(3, n == 8 ? 4 : 3, CV_8U);
        b = Mat(a.size(), a.type());
        randu(a, 0, 256);
        randu(b, 0, 256);
        if (n == 5) {
            r = wrapAdd(a, b);
        } else if (n == 6) {
            add(a, b, r);
        } else {
            addWeighted(a, 2, b, 1, 3, r);
        }
        std::cout << "a=" << a << "\nb=" << b << "\nresult=" << r << '\n';
        return;
    }
    if (n == 7) {
        a = c.read("lena.bmp", IMREAD_GRAYSCALE);
        c.show("original", a);
        c.show("wrapped", wrapAdd(a, a));
        add(a, a, r);
        c.show("saturated", r);
        return;
    }
    if (n == 9) {
        a = c.read("boat.bmp", IMREAD_GRAYSCALE);
        b = c.read("lena.bmp", IMREAD_GRAYSCALE);
        addWeighted(a, .6, b, .4, 0, r);
        c.show("boat", a);
        c.show("lena", b);
        c.show("blend", r);
        return;
    }
    if (n == 10) {
        a = c.read("c.bmp");
        b = c.read("d.bmp");
        c.show("c", a);
        c.show("d", b);
        c.show("wrapped", wrapAdd(a, b));
        add(a, b, r);
        c.show("add", r);
        addWeighted(a, 1, b, 1, 0, r);
        c.show("weighted", r);
        return;
    }
    a = c.read("back.jpg");
    b = c.read("fore.jpg");
    Mat mask = redMask(b, true), back = masked(a, mask), fore = masked(b, 255 - mask);
    c.show("background", a);
    c.show("foreground", b);
    c.show("mask1", mask);
    c.show("mask2", 255 - mask);
    c.show("C", back);
    c.show("D", fore);
    c.show("result", back + fore);
}
} // namespace cv40

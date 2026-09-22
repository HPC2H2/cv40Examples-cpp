#include "cv40.hpp"
namespace cv40 {
void chapter4(Context &c, int n, int variant) {
    Mat original = c.read("lena.bmp", IMREAD_GRAYSCALE), key(original.size(), CV_8U), encrypted, decrypted;
    randu(key, 0, 256);
    bitwise_xor(original, key, encrypted);
    bitwise_xor(encrypted, key, decrypted);
    c.show("lena", original);
    if (n == 1) {
        c.show("key", key);
        c.show("encryption", encrypted);
        c.show("decryption", decrypted);
        return;
    }
    Rect face(250, 220, 100, 180);
    Mat mask = Mat::zeros(original.size(), CV_8U);
    mask(face) = 255;
    Mat hidden = original.clone();
    encrypted(face).copyTo(hidden(face));
    Mat decoded;
    bitwise_xor(hidden, key, decoded);
    Mat restored = original.clone();
    decoded(face).copyTo(restored(face));
    if (n == 2 && variant) {
        c.show("mask", mask);
        c.show("key", key);
        c.show("xor_all", encrypted);
        c.show("encrypted_face", masked(encrypted, mask));
        c.show("outside", masked(original, 255 - mask));
        c.show("decoded_all", decoded);
    }
    if (n == 3) {
        c.show("secretFace", encrypted(face));
        c.show("face", decoded(face));
    }
    c.show("maskFace", hidden);
    c.show("extractLena", restored);
    require(norm(original, restored, NORM_INF) == 0, "XOR round-trip failed");
}
void chapter5(Context &c, int n, int variant) {
    if (n == 6) {
        Mat a = c.read("image/lenacolor.png"), w = c.read("image/watermark.bmp"), r;
        bitwise_or(a, w, r);
        c.show("original", a);
        c.show("watermark", w);
        c.show("result", r);
        return;
    }
    Mat a = c.read("image/lena.bmp", IMREAD_GRAYSCALE);
    c.show("original", a);
    if (n == 1) {
        for (int i = 0; i < 8; ++i) {
            Mat plane;
            bitwise_and(a, Scalar(1 << i), plane);
            c.show("plane" + std::to_string(i), plane > 0);
        }
        return;
    }
    Mat w = c.read("image/watermark.bmp", IMREAD_GRAYSCALE);
    require(a.size() == w.size(), "Watermark and source dimensions must match");
    c.show("watermark", w);
    if (n == 2 || n == 5) {
        Mat clear, bit, embedded, extracted;
        bitwise_and(a, Scalar(254), clear);
        bit = n == 5 ? ((w == 255) / 255) : ((w > 0) / 255);
        if (n == 5 && variant) {
            bit = w.clone();
            bit.setTo(255, w > 1);
        }
        embedded = wrapAdd(clear, bit);
        bitwise_and(embedded, Scalar(1), extracted);
        c.show("without_lsb", clear);
        c.show("embedded", embedded);
        c.show("extracted", extracted * 255);
        require(countNonZero(extracted != ((bit & 1))) == 0, "Watermark extraction failed");
    } else if (n == 3) {
        Mat b = (w > 0) / 255, d = a.mul(b), e = 255 - w;
        c.show("binary", b * 255);
        c.show("D", d);
        c.show("E", e);
        c.show("F", wrapAdd(d, e));
    } else if (n == 4) {
        Mat r;
        c.show("uint8_wrap", wrapAdd(a, w));
        add(a, w, r);
        c.show("saturated", r);
        addWeighted(a, .6, w, .3, 55, r);
        c.show("weighted", r);
    }
}
} // namespace cv40

"""Optional independent checks against the unmodified textbook Python examples.

This script is a verification tool, never a dependency of the C++ executable.
It redirects GUI calls and image writes, and fails on missing results or excessive
pixel differences. Run verify.cmd first to produce a fresh C++ output directory.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import types
from unittest.mock import patch

import cv2
import numpy as np


class UnusedPlot:
    """The MCC example plots two intermediate charts; neither affects its math."""

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


@contextlib.contextmanager
def original_environment(directory, reference_data=None):
    previous_directory = Path.cwd()
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.__getattr__ = lambda _name: lambda *_args, **_kwargs: None
    pyplot.subplots = lambda *_args, **_kwargs: (UnusedPlot(), [UnusedPlot(), UnusedPlot()])
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    try:
        os.chdir(directory)
        with contextlib.ExitStack() as stack:
            for name in ("imshow", "waitKey", "destroyAllWindows"):
                stack.enter_context(patch.object(cv2, name, lambda *_args, **_kwargs: 0))
            stack.enter_context(patch.object(cv2, "imwrite", lambda *_args, **_kwargs: True))
            stack.enter_context(patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}))
            if reference_data is not None:
                # The supplied NPZ contains an object array. Use its exact exported
                # numeric CSV data so verification does not enable pickle loading.
                reference = {
                    "arr_0": np.loadtxt(reference_data / "sample_1_2_arr_0.csv", delimiter=","),
                    "arr_1": np.loadtxt(reference_data / "sample_1_2_arr_1.csv", delimiter=","),
                }
                stack.enter_context(patch.object(np, "load", lambda *_args, **_kwargs: reference))
            yield
    finally:
        os.chdir(previous_directory)


def execute_original(source, reference_data=None):
    namespace = {"__name__": "__main__", "__file__": str(source)}
    with original_environment(source.parent, reference_data), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source.read_text(encoding="utf-8-sig"), str(source), "exec"), namespace)
    return namespace


def compare_pixels(name, expected, actual_path, max_difference=0, max_changed=0):
    if not actual_path.is_file():
        raise AssertionError(f"Missing C++ output: {actual_path}")
    actual = cv2.imdecode(np.fromfile(actual_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if actual is None or actual.shape != expected.shape:
        raise AssertionError(f"{name}: decoded image shape does not match {expected.shape}")
    differences = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    result = {
        "name": name,
        "maximum_difference": int(differences.max()),
        "different_values": int(np.count_nonzero(differences)),
    }
    result["passed"] = result["maximum_difference"] <= max_difference and result["different_values"] <= max_changed
    print(result)
    return result


def compare_hog(original, cpp_log):
    captured = []
    original_factory = cv2.HOGDescriptor

    class ObservedHog:
        def __init__(self):
            self.detector = original_factory()

        def setSVMDetector(self, detector):
            self.detector.setSVMDetector(detector)

        def detectMultiScale(self, *args, **kwargs):
            boxes, weights = self.detector.detectMultiScale(*args, **kwargs)
            captured.append(sorted(tuple(map(int, box)) for box in boxes))
            return boxes, weights

    previous_threads = cv2.getNumThreads()
    try:
        # Match the C++ workaround for OpenCV HOG's parallel box/weight ordering.
        cv2.setNumThreads(1)
        with patch.object(cv2, "HOGDescriptor", ObservedHog):
            execute_original(original / "19行人检测/例19.5.py")
    finally:
        cv2.setNumThreads(previous_threads)
    block = re.search(r"RUN 19\.5 [^\n]*\n(.*?)PASS 19\.5\r?\n", cpp_log, re.S)
    if block is None or len(captured) != 2:
        raise AssertionError("Missing HOG comparison data")
    results = []
    for index, name in enumerate(("without_meanshift", "meanshift")):
        rows = re.findall(rf"^{name}: rect=(\d+),(\d+),(\d+),(\d+)", block[1], re.M)
        actual = sorted(tuple(map(int, row)) for row in rows)
        results.append({"name": "19.5:" + name, "passed": actual == captured[index],
                        "python_boxes": captured[index], "cpp_boxes": actual})
    return results


def run_checks(original, outputs, data):
    results = []
    cases = [
        ("3OpenCV/例3.9.py", "3.9", "rst", "2_resize.png"),
        ("5数字水印/例5.3.py", "5.3", "F", "6_F.png"),
        ("6物体计数/例6.8.py", "6.8", "gaussian", "2_gaussian.png"),
        ("10隐身术/例10.11.py", "10.11", "E", "7_result.png"),
    ]
    for relative_source, example_id, variable, filename in cases:
        namespace = execute_original(original / relative_source)
        results.append(compare_pixels(example_id, namespace[variable], outputs / example_id / filename))

    fingerprint = execute_original(original / "14指纹识别/FingerprintRecognition.py", data / "14指纹识别")
    # Allow only the two one-level rounding differences observed between the
    # independent C++ and NumPy enhancement paths. The skeleton must match exactly.
    results.append(compare_pixels("14.0:enhanced", fingerprint["enhanced"], outputs / "14.0/14_enhanced.png", 1, 2))
    results.append(compare_pixels("14.0:skeleton", fingerprint["skeleton"], outputs / "14.0/16_skeleton.png"))
    cpp_log = (outputs.parent / "logs/basic-164.log").read_text(encoding="utf-8")
    match = re.search(r"minutiae=(\d+) descriptor=\[(\d+) x (\d+)\] comparison score=([\d.]+)", cpp_log)
    period = re.search(r"ridge period=([\d.]+)", cpp_log)
    if match is None or period is None:
        raise AssertionError("Missing C++ fingerprint measurements")
    cpp_count, cpp_width, cpp_rows = map(int, match.groups()[:3])
    numerical_match = (
        cpp_count == cpp_rows == len(fingerprint["valid_minutiae"])
        and cpp_width == 208
        and abs(float(match[4]) - float(fingerprint["score"])) < 1e-6
        and abs(float(period[1]) - float(fingerprint["ridge_period"])) < 1e-5
    )
    results.append({"name": "14.0:numerical-result", "passed": numerical_match, "python_score": float(fingerprint["score"])})
    results.extend(compare_hog(original, cpp_log))
    return results


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Original repository or its code directory")
    parser.add_argument("--output", type=Path, help="C++ images directory; defaults to latest verify.cmd run")
    arguments = parser.parse_args()
    original = arguments.original.resolve()
    if (original / "code").is_dir():
        original /= "code"
    outputs = arguments.output
    if outputs is None:
        latest = json.loads((project / "output/latest-verification.json").read_text(encoding="utf-8"))
        outputs = Path(latest["images"])
    outputs = outputs.resolve()
    report_path = outputs.parent / "python-comparison.json"
    try:
        results = run_checks(original, outputs, project / "data")
    except Exception as error:
        results = [{"name": "comparison-error", "passed": False, "error": str(error)}]
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

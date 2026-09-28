"""
CSc 8830 - Computer Vision
Module 4 - Question 3

Frequency-Domain Edge Detection and Segmentation

This program demonstrates:
1. 2-D Discrete Fourier Transform (DFT)
2. Fourier magnitude spectrum
3. Ideal Low-Pass Filter (ILPF)
4. Ideal High-Pass Filter (IHPF)
5. Frequency-domain edge detection
6. Frequency-domain smoothing
7. Threshold-based segmentation after frequency filtering

The implementation uses NumPy and OpenCV.
"""

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# We use the RGB human image from Question 1.
IMAGE_PATH = (
    BASE_DIR
    / "images"
    / "rgb"
    / "person.jpg"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "frequency_domain"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )


print("=" * 65)
print("QUESTION 3 - FREQUENCY DOMAIN ANALYSIS")
print("=" * 65)

print("\nImage loaded successfully.")
print("Original shape:", image.shape)


# ============================================================
# RESIZE IMAGE
# ============================================================

MAX_HEIGHT = 900

original_height, original_width = (
    image.shape[:2]
)


if original_height > MAX_HEIGHT:

    scale = (
        MAX_HEIGHT
        / original_height
    )

    new_width = int(
        original_width
        * scale
    )

    image = cv2.resize(
        image,
        (
            new_width,
            MAX_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )


height, width = image.shape[:2]


print(
    f"Processing size: "
    f"{width} x {height}"
)


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# STEP 1 - 2-D FOURIER TRANSFORM
# ============================================================
#
# For a digital image:
#
# F(u,v) =
#
# SUM_x SUM_y
# f(x,y) *
# exp[-j2*pi*(ux/M + vy/N)]
#
# np.fft.fft2() computes the 2-D DFT.
# ============================================================

fourier = np.fft.fft2(
    gray
)


# Move zero frequency from corner
# to center of spectrum.

fourier_shifted = np.fft.fftshift(
    fourier
)


# ============================================================
# STEP 2 - MAGNITUDE SPECTRUM
# ============================================================
#
# Magnitude:
#
# |F(u,v)|
#
# Log transformation makes the spectrum
# visible because Fourier magnitude values
# have a very large dynamic range.
# ============================================================

magnitude_spectrum = (
    20
    * np.log(
        np.abs(
            fourier_shifted
        )
        + 1
    )
)


magnitude_normalized = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(
    np.uint8
)


# ============================================================
# FREQUENCY COORDINATES
# ============================================================

center_y = (
    height // 2
)

center_x = (
    width // 2
)


y_coordinates, x_coordinates = (
    np.ogrid[
        :height,
        :width
    ]
)


distance = np.sqrt(
    (
        x_coordinates
        - center_x
    ) ** 2
    +
    (
        y_coordinates
        - center_y
    ) ** 2
)


# ============================================================
# CUTOFF FREQUENCY
# ============================================================
#
# D0 controls how much of the frequency
# spectrum is preserved.
#
# Frequencies close to the center are low.
# Frequencies far from the center are high.
# ============================================================

D0 = max(
    20,
    int(
        min(
            height,
            width
        )
        * 0.08
    )
)


print(
    f"Cutoff frequency D0: {D0}"
)


# ============================================================
# STEP 3 - IDEAL LOW-PASS FILTER
# ============================================================
#
# H_LP(u,v) =
#
# 1, D(u,v) <= D0
# 0, D(u,v) > D0
#
# Low frequencies are retained.
# High frequencies are suppressed.
# ============================================================

low_pass_filter = np.zeros(
    (
        height,
        width
    ),
    dtype=np.float32
)


low_pass_filter[
    distance <= D0
] = 1.0


# ============================================================
# APPLY LOW-PASS FILTER
# ============================================================

low_pass_frequency = (
    fourier_shifted
    * low_pass_filter
)


# ============================================================
# INVERSE FOURIER TRANSFORM
# ============================================================

low_pass_inverse_shift = (
    np.fft.ifftshift(
        low_pass_frequency
    )
)


low_pass_image_complex = (
    np.fft.ifft2(
        low_pass_inverse_shift
    )
)


low_pass_image = np.abs(
    low_pass_image_complex
)


low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(
    np.uint8
)


# ============================================================
# STEP 4 - IDEAL HIGH-PASS FILTER
# ============================================================
#
# H_HP(u,v) =
#
# 0, D(u,v) <= D0
# 1, D(u,v) > D0
#
# High frequencies correspond to rapid
# intensity changes such as edges.
# ============================================================

high_pass_filter = np.ones(
    (
        height,
        width
    ),
    dtype=np.float32
)


high_pass_filter[
    distance <= D0
] = 0.0


# ============================================================
# APPLY HIGH-PASS FILTER
# ============================================================

high_pass_frequency = (
    fourier_shifted
    * high_pass_filter
)


# ============================================================
# INVERSE FOURIER TRANSFORM
# ============================================================

high_pass_inverse_shift = (
    np.fft.ifftshift(
        high_pass_frequency
    )
)


high_pass_image_complex = (
    np.fft.ifft2(
        high_pass_inverse_shift
    )
)


high_pass_image = np.abs(
    high_pass_image_complex
)


high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(
    np.uint8
)


# ============================================================
# STEP 5 - FREQUENCY-DOMAIN EDGE MAP
# ============================================================
#
# High-pass filtering emphasizes high
# frequencies.
#
# We threshold the high-pass result to
# obtain a binary edge map.
# ============================================================

edge_threshold, edge_map = (
    cv2.threshold(
        high_pass_image,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )
)


print(
    f"Edge Otsu threshold: "
    f"{edge_threshold:.2f}"
)


# Remove small isolated responses.

edge_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)


edge_map = cv2.morphologyEx(
    edge_map,
    cv2.MORPH_OPEN,
    edge_kernel,
    iterations=1
)


# ============================================================
# STEP 6 - SEGMENTATION USING LOW-PASS RESULT
# ============================================================
#
# Frequency-domain low-pass filtering
# suppresses high-frequency variation.
#
# We then apply thresholding to the
# smoothed image:
#
# S(x,y) =
#
# 1, g(x,y) >= T
# 0, g(x,y) < T
# ============================================================

segmentation_threshold, segmentation = (
    cv2.threshold(
        low_pass_image,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )
)


print(
    f"Segmentation Otsu threshold: "
    f"{segmentation_threshold:.2f}"
)


# ============================================================
# MORPHOLOGICAL CLEANUP OF SEGMENTATION
# ============================================================

segmentation_kernel = (
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )
)


segmentation = cv2.morphologyEx(
    segmentation,
    cv2.MORPH_CLOSE,
    segmentation_kernel,
    iterations=2
)


segmentation = cv2.morphologyEx(
    segmentation,
    cv2.MORPH_OPEN,
    np.ones(
        (3, 3),
        dtype=np.uint8
    ),
    iterations=1
)


# ============================================================
# STEP 7 - FILTER VISUALIZATIONS
# ============================================================

low_pass_visual = (
    low_pass_filter
    * 255
).astype(
    np.uint8
)


high_pass_visual = (
    high_pass_filter
    * 255
).astype(
    np.uint8
)


# ============================================================
# STEP 8 - SEGMENTED IMAGE
# ============================================================

segmented_image = cv2.bitwise_and(
    image,
    image,
    mask=segmentation
)


# ============================================================
# STEP 9 - EDGE OVERLAY
# ============================================================

edge_overlay = (
    image.copy()
)


edge_contours, _ = (
    cv2.findContours(
        edge_map,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
)


cv2.drawContours(
    edge_overlay,
    edge_contours,
    -1,
    (0, 0, 255),
    1
)


# ============================================================
# STEP 10 - CREATE COMPARISON PANELS
# ============================================================

def to_bgr(
    img
):

    if len(
        img.shape
    ) == 2:

        return cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    return img.copy()


def add_title(
    img,
    title
):

    result = to_bgr(
        img
    )


    cv2.rectangle(
        result,
        (0, 0),
        (
            result.shape[1],
            32
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        result,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    return result


# ============================================================
# FOURIER COMPARISON
# ============================================================

fourier_comparison = np.hstack(
    [
        add_title(
            gray,
            "Grayscale Input"
        ),

        add_title(
            magnitude_normalized,
            "Fourier Magnitude"
        ),

        add_title(
            low_pass_visual,
            "Low-Pass Filter"
        ),

        add_title(
            high_pass_visual,
            "High-Pass Filter"
        ),
    ]
)


# ============================================================
# EDGE COMPARISON
# ============================================================

edge_comparison = np.hstack(
    [
        add_title(
            gray,
            "Original"
        ),

        add_title(
            high_pass_image,
            "High-Pass Result"
        ),

        add_title(
            edge_map,
            "Frequency Edge Map"
        ),

        add_title(
            edge_overlay,
            "Edge Overlay"
        ),
    ]
)


# ============================================================
# SEGMENTATION COMPARISON
# ============================================================

segmentation_comparison = np.hstack(
    [
        add_title(
            image,
            "Original"
        ),

        add_title(
            low_pass_image,
            "Low-Pass Result"
        ),

        add_title(
            segmentation,
            "Segmentation Mask"
        ),

        add_title(
            segmented_image,
            "Segmented Image"
        ),
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_original.jpg"
    ),
    image
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_grayscale.jpg"
    ),
    gray
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "fourier_magnitude_spectrum.jpg"
    ),
    magnitude_normalized
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "ideal_low_pass_filter.jpg"
    ),
    low_pass_visual
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "low_pass_result.jpg"
    ),
    low_pass_image
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "ideal_high_pass_filter.jpg"
    ),
    high_pass_visual
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "high_pass_result.jpg"
    ),
    high_pass_image
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_edge_map.jpg"
    ),
    edge_map
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_edge_overlay.jpg"
    ),
    edge_overlay
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_segmentation_mask.jpg"
    ),
    segmentation
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_segmented_image.jpg"
    ),
    segmented_image
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "fourier_comparison.jpg"
    ),
    fourier_comparison
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_edge_comparison.jpg"
    ),
    edge_comparison
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "frequency_segmentation_comparison.jpg"
    ),
    segmentation_comparison
)


# ============================================================
# SAVE NUMERICAL INFORMATION
# ============================================================

results_file = (
    OUTPUT_DIR
    / "frequency_domain_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CSc 8830 - Module 4 - Question 3\n"
    )

    file.write(
        "Frequency Domain Analysis\n"
    )

    file.write(
        "=" * 50
        + "\n\n"
    )

    file.write(
        f"Image width: {width}\n"
    )

    file.write(
        f"Image height: {height}\n"
    )

    file.write(
        f"Cutoff frequency D0: {D0}\n"
    )

    file.write(
        f"Edge Otsu threshold: "
        f"{edge_threshold:.2f}\n"
    )

    file.write(
        f"Segmentation Otsu threshold: "
        f"{segmentation_threshold:.2f}\n"
    )

    file.write(
        "\n"
        "Edge Detection:\n"
    )

    file.write(
        "High-frequency components were "
        "preserved using an ideal "
        "high-pass filter.\n"
    )

    file.write(
        "The inverse DFT produced a "
        "spatial-domain edge-enhanced "
        "image.\n"
    )

    file.write(
        "\n"
        "Segmentation:\n"
    )

    file.write(
        "Low-frequency components were "
        "preserved using an ideal "
        "low-pass filter.\n"
    )

    file.write(
        "The smoothed inverse-DFT image "
        "was segmented using Otsu "
        "thresholding.\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("FREQUENCY DOMAIN ANALYSIS COMPLETE")
print("=" * 65)

print(
    "\nGenerated files:"
)

print(
    "1. frequency_original.jpg"
)

print(
    "2. frequency_grayscale.jpg"
)

print(
    "3. fourier_magnitude_spectrum.jpg"
)

print(
    "4. ideal_low_pass_filter.jpg"
)

print(
    "5. low_pass_result.jpg"
)

print(
    "6. ideal_high_pass_filter.jpg"
)

print(
    "7. high_pass_result.jpg"
)

print(
    "8. frequency_edge_map.jpg"
)

print(
    "9. frequency_edge_overlay.jpg"
)

print(
    "10. frequency_segmentation_mask.jpg"
)

print(
    "11. frequency_segmented_image.jpg"
)

print(
    "12. fourier_comparison.jpg"
)

print(
    "13. frequency_edge_comparison.jpg"
)

print(
    "14. frequency_segmentation_comparison.jpg"
)

print(
    "15. frequency_domain_results.txt"
)


print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nDone.")
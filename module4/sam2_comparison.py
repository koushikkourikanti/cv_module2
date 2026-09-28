"""
CSc 8830 - Computer Vision
Module 4 - Question 1
OpenCV Classical Segmentation vs SAM2.1 Comparison

This script:
1. Loads the same RGB human image used for the classical OpenCV method.
2. Runs SAM2.1 using a bounding-box prompt.
3. Extracts the SAM2 human segmentation mask.
4. Finds the SAM2 human boundary.
5. Compares the classical OpenCV mask with SAM2.
6. Computes IoU and Dice agreement metrics.
7. Saves visual comparison results.

NOTE:
SAM2 is used ONLY for comparison.
The primary Question 1 implementation is classical OpenCV without ML/DL.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(
    "/mnt/c/Users/kouri/Desktop/Computer Vision/CSc8830_Module2"
)

IMAGE_PATH = (
    PROJECT_DIR
    / "module4"
    / "images"
    / "rgb"
    / "person.jpg"
)

OUTPUT_DIR = PROJECT_DIR / "module4" / "outputs"

OPENCV_MASK_PATH = OUTPUT_DIR / "rgb_binary_mask.jpg"

SAM2_CHECKPOINT = Path(
    "/home/koushik_kourikanti/sam2/checkpoints/"
    "sam2.1_hiera_small.pt"
)

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

print("=" * 60)
print("SAM2.1 HUMAN SEGMENTATION")
print("=" * 60)

image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )

print(f"Image loaded: {IMAGE_PATH}")
print(f"Original image size: {image_bgr.shape}")


# ---------------------------------------------------------
# RESIZE IMAGE
# Match the classical OpenCV processing height
# ---------------------------------------------------------

max_height = 900

h_original, w_original = image_bgr.shape[:2]

if h_original > max_height:
    scale = max_height / h_original

    new_width = int(w_original * scale)

    image_bgr = cv2.resize(
        image_bgr,
        (new_width, max_height),
        interpolation=cv2.INTER_AREA,
    )

h, w = image_bgr.shape[:2]

print(f"Processing size: {w} x {h}")


# SAM2 expects RGB
image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB,
)


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ---------------------------------------------------------
# LOAD SAM2.1
# ---------------------------------------------------------

print("\nLoading SAM2.1 model...")

sam2_model = build_sam2(
    MODEL_CONFIG,
    str(SAM2_CHECKPOINT),
    device=device,
)

predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM2.1 model loaded successfully.")


# ---------------------------------------------------------
# SET IMAGE
# ---------------------------------------------------------

print("Encoding image with SAM2...")

with torch.inference_mode():
    predictor.set_image(image_rgb)


# ---------------------------------------------------------
# BOUNDING BOX PROMPT
# ---------------------------------------------------------
# The person is approximately centered in the image.
# This broad box contains the complete person while
# excluding much of the surrounding background.
#
# Coordinates:
# [x_min, y_min, x_max, y_max]
# ---------------------------------------------------------

input_box = np.array(
    [
        int(0.25 * w),
        int(0.02 * h),
        int(0.75 * w),
        int(0.98 * h),
    ],
    dtype=np.float32,
)

print(
    "SAM2 bounding box:",
    input_box.astype(int),
)


# Draw prompt box for documentation
prompt_image = image_bgr.copy()

x1, y1, x2, y2 = input_box.astype(int)

cv2.rectangle(
    prompt_image,
    (x1, y1),
    (x2, y2),
    (255, 0, 0),
    3,
)

cv2.imwrite(
    str(OUTPUT_DIR / "sam2_prompt_box.jpg"),
    prompt_image,
)


# ---------------------------------------------------------
# RUN SAM2 SEGMENTATION
# ---------------------------------------------------------

print("Running SAM2 segmentation...")

with torch.inference_mode():

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,
    )


print("SAM2 candidate scores:", scores)


# ---------------------------------------------------------
# SELECT BEST MASK
# ---------------------------------------------------------

best_index = int(np.argmax(scores))

sam2_mask = masks[best_index]

best_score = float(scores[best_index])

print(f"Selected mask index: {best_index}")
print(f"SAM2 predicted score: {best_score:.4f}")


# Convert boolean mask to uint8
sam2_mask_uint8 = (
    sam2_mask.astype(np.uint8) * 255
)


# ---------------------------------------------------------
# CLEAN SAM2 MASK
# ---------------------------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5),
)

sam2_mask_uint8 = cv2.morphologyEx(
    sam2_mask_uint8,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1,
)


# ---------------------------------------------------------
# FIND MAIN SAM2 CONTOUR
# ---------------------------------------------------------

contours, _ = cv2.findContours(
    sam2_mask_uint8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE,
)

if not contours:
    raise RuntimeError(
        "SAM2 produced no usable human contour."
    )

largest_contour = max(
    contours,
    key=cv2.contourArea,
)

sam2_area = cv2.contourArea(
    largest_contour
)

print(
    f"SAM2 contour area: {sam2_area:.1f} pixels"
)


# ---------------------------------------------------------
# CREATE FINAL FILLED SAM2 MASK
# ---------------------------------------------------------

final_sam2_mask = np.zeros(
    (h, w),
    dtype=np.uint8,
)

cv2.drawContours(
    final_sam2_mask,
    [largest_contour],
    -1,
    255,
    thickness=cv2.FILLED,
)


# ---------------------------------------------------------
# SEGMENTED PERSON
# ---------------------------------------------------------

sam2_segmented = cv2.bitwise_and(
    image_bgr,
    image_bgr,
    mask=final_sam2_mask,
)


# ---------------------------------------------------------
# DRAW SAM2 BOUNDARY
# ---------------------------------------------------------

sam2_boundary = image_bgr.copy()

cv2.drawContours(
    sam2_boundary,
    [largest_contour],
    -1,
    (0, 0, 255),
    3,
)


# Boundary-only visualization
boundary_only = np.full_like(
    image_bgr,
    255,
)

cv2.drawContours(
    boundary_only,
    [largest_contour],
    -1,
    (0, 0, 0),
    3,
)


# ---------------------------------------------------------
# LOAD CLASSICAL OPENCV MASK
# ---------------------------------------------------------

opencv_mask = cv2.imread(
    str(OPENCV_MASK_PATH),
    cv2.IMREAD_GRAYSCALE,
)

if opencv_mask is None:
    raise FileNotFoundError(
        f"Could not load OpenCV mask:\n"
        f"{OPENCV_MASK_PATH}"
    )


# Ensure same dimensions
opencv_mask = cv2.resize(
    opencv_mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST,
)

_, opencv_mask = cv2.threshold(
    opencv_mask,
    127,
    255,
    cv2.THRESH_BINARY,
)


# ---------------------------------------------------------
# AGREEMENT METRICS
# ---------------------------------------------------------

opencv_bool = opencv_mask > 0
sam2_bool = final_sam2_mask > 0

intersection = np.logical_and(
    opencv_bool,
    sam2_bool,
).sum()

union = np.logical_or(
    opencv_bool,
    sam2_bool,
).sum()

opencv_pixels = opencv_bool.sum()
sam2_pixels = sam2_bool.sum()

iou = (
    intersection / union
    if union > 0
    else 0.0
)

dice = (
    (2.0 * intersection)
    / (opencv_pixels + sam2_pixels)
    if (opencv_pixels + sam2_pixels) > 0
    else 0.0
)


print("\n" + "=" * 60)
print("OPENCV vs SAM2 AGREEMENT")
print("=" * 60)

print(
    f"OpenCV foreground pixels: {opencv_pixels}"
)

print(
    f"SAM2 foreground pixels:   {sam2_pixels}"
)

print(
    f"Intersection pixels:       {intersection}"
)

print(
    f"Union pixels:              {union}"
)

print(
    f"IoU agreement:             {iou:.4f}"
)

print(
    f"Dice agreement:            {dice:.4f}"
)


# ---------------------------------------------------------
# CREATE OPENCV BOUNDARY FOR COMPARISON
# ---------------------------------------------------------

opencv_contours, _ = cv2.findContours(
    opencv_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE,
)

opencv_boundary = image_bgr.copy()

if opencv_contours:

    opencv_largest = max(
        opencv_contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        opencv_boundary,
        [opencv_largest],
        -1,
        (0, 0, 255),
        3,
    )


# ---------------------------------------------------------
# SIDE-BY-SIDE COMPARISON
# ---------------------------------------------------------

def add_title(image, title):

    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        result,
        title,
        (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


original_panel = add_title(
    image_bgr,
    "Original",
)

opencv_panel = add_title(
    opencv_boundary,
    "Classical OpenCV",
)

sam2_panel = add_title(
    sam2_boundary,
    "SAM2.1",
)

comparison = np.hstack(
    [
        original_panel,
        opencv_panel,
        sam2_panel,
    ]
)


# ---------------------------------------------------------
# MASK COMPARISON VISUALIZATION
# ---------------------------------------------------------
# White = agreement foreground
# Gray/background areas are not segmented.
#
# OpenCV-only and SAM2-only regions are shown using
# different channels for visual comparison.
# ---------------------------------------------------------

agreement_visual = np.zeros(
    (h, w, 3),
    dtype=np.uint8,
)

both = np.logical_and(
    opencv_bool,
    sam2_bool,
)

opencv_only = np.logical_and(
    opencv_bool,
    np.logical_not(sam2_bool),
)

sam2_only = np.logical_and(
    sam2_bool,
    np.logical_not(opencv_bool),
)

agreement_visual[both] = (
    255,
    255,
    255,
)

agreement_visual[opencv_only] = (
    255,
    0,
    0,
)

agreement_visual[sam2_only] = (
    0,
    0,
    255,
)


# ---------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------

cv2.imwrite(
    str(OUTPUT_DIR / "sam2_mask.jpg"),
    final_sam2_mask,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "sam2_segmented_person.jpg"
    ),
    sam2_segmented,
)

cv2.imwrite(
    str(OUTPUT_DIR / "sam2_boundary.jpg"),
    sam2_boundary,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "sam2_boundary_only.jpg"
    ),
    boundary_only,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "opencv_vs_sam2.jpg"
    ),
    comparison,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "opencv_sam2_mask_agreement.jpg"
    ),
    agreement_visual,
)


# Save metrics as text
metrics_path = (
    OUTPUT_DIR
    / "opencv_vs_sam2_metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "OpenCV vs SAM2.1 Segmentation Agreement\n"
    )

    file.write(
        "======================================\n"
    )

    file.write(
        f"SAM2 score: {best_score:.6f}\n"
    )

    file.write(
        f"OpenCV foreground pixels: "
        f"{opencv_pixels}\n"
    )

    file.write(
        f"SAM2 foreground pixels: "
        f"{sam2_pixels}\n"
    )

    file.write(
        f"Intersection pixels: "
        f"{intersection}\n"
    )

    file.write(
        f"Union pixels: {union}\n"
    )

    file.write(
        f"IoU agreement: {iou:.6f}\n"
    )

    file.write(
        f"Dice agreement: {dice:.6f}\n"
    )

    file.write(
        "\nImportant: IoU and Dice here measure "
        "agreement between the OpenCV and SAM2 "
        "segmentations. They are not ground-truth "
        "accuracy measurements.\n"
    )


print("\n" + "=" * 60)
print("SAM2 COMPARISON COMPLETE")
print("=" * 60)

print(f"Outputs saved to:\n{OUTPUT_DIR}")

print("\nGenerated files:")
print("1. sam2_prompt_box.jpg")
print("2. sam2_mask.jpg")
print("3. sam2_segmented_person.jpg")
print("4. sam2_boundary.jpg")
print("5. sam2_boundary_only.jpg")
print("6. opencv_vs_sam2.jpg")
print("7. opencv_sam2_mask_agreement.jpg")
print("8. opencv_vs_sam2_metrics.txt")

print("\nComparison metrics:")
print(f"IoU  = {iou:.4f}")
print(f"Dice = {dice:.4f}")
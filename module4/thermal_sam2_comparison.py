"""
CSc 8830 - Computer Vision
Module 4 - Question 2

SAM2.1 Comparison for Thermal Human Segmentation

This script:
1. Loads the same thermal image used by the classical OpenCV method.
2. Loads the OpenCV thermal mask.
3. Runs SAM2.1 using a bounding-box prompt.
4. Extracts the SAM2 human mask and boundary.
5. Compares classical OpenCV and SAM2.1.
6. Calculates IoU and Dice agreement.

IMPORTANT:
IoU and Dice measure agreement between OpenCV and SAM2.
They are NOT ground-truth segmentation accuracy.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "images"
    / "thermal"
    / "person_thermal.jpg"
)

OPENCV_MASK_PATH = (
    BASE_DIR
    / "outputs"
    / "thermal_binary_mask.jpg"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAM2 PATHS
# ============================================================

CHECKPOINT = (
    "/home/koushik_kourikanti/"
    "sam2/checkpoints/"
    "sam2.1_hiera_small.pt"
)

MODEL_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_s.yaml"
)


# ============================================================
# LOAD THERMAL IMAGE
# ============================================================

image_bgr = cv2.imread(
    str(IMAGE_PATH)
)

if image_bgr is None:
    raise FileNotFoundError(
        f"Could not load thermal image:\n{IMAGE_PATH}"
    )


original_height, original_width = (
    image_bgr.shape[:2]
)


print("=" * 65)
print("SAM2.1 THERMAL HUMAN SEGMENTATION")
print("=" * 65)

print("\nThermal image loaded successfully.")

print(
    f"Original image size: "
    f"{original_width} x "
    f"{original_height}"
)


# ============================================================
# RESIZE FOR SAM2
# ============================================================
#
# The thermal image is fairly large.
# We reduce its height to 900 pixels for faster inference.
# The OpenCV mask will be resized to the same dimensions.
# ============================================================

MAX_HEIGHT = 900


if original_height > MAX_HEIGHT:

    scale = (
        MAX_HEIGHT
        / original_height
    )

    processing_width = int(
        original_width
        * scale
    )

    processing_height = (
        MAX_HEIGHT
    )


    image_bgr = cv2.resize(
        image_bgr,
        (
            processing_width,
            processing_height
        ),
        interpolation=cv2.INTER_AREA
    )

else:

    processing_height = (
        original_height
    )

    processing_width = (
        original_width
    )


height, width = (
    image_bgr.shape[:2]
)


print(
    f"Processing size: "
    f"{width} x {height}"
)


# ============================================================
# LOAD OPENCV MASK
# ============================================================

opencv_mask = cv2.imread(
    str(OPENCV_MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if opencv_mask is None:
    raise FileNotFoundError(
        f"Could not load OpenCV mask:\n"
        f"{OPENCV_MASK_PATH}"
    )


opencv_mask = cv2.resize(
    opencv_mask,
    (
        width,
        height
    ),
    interpolation=cv2.INTER_NEAREST
)


_, opencv_mask = cv2.threshold(
    opencv_mask,
    127,
    255,
    cv2.THRESH_BINARY
)


# ============================================================
# CONVERT THERMAL IMAGE TO RGB
# ============================================================

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# SELECT DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    f"\nDevice: {device}"
)


if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )


# ============================================================
# LOAD SAM2.1
# ============================================================

print(
    "\nLoading SAM2.1 model..."
)


sam2_model = build_sam2(
    MODEL_CONFIG,
    CHECKPOINT,
    device=device
)


predictor = SAM2ImagePredictor(
    sam2_model
)


print(
    "SAM2.1 model loaded successfully."
)


# ============================================================
# ENCODE IMAGE
# ============================================================

print(
    "\nEncoding thermal image with SAM2..."
)


with torch.inference_mode():

    predictor.set_image(
        image_rgb
    )


# ============================================================
# BOUNDING BOX PROMPT
# ============================================================
#
# The thermal subject occupies the center of the frame.
#
# The prompt box intentionally includes the entire person
# with some background around the body.
#
# This is the prompt supplied to SAM2; SAM2 determines the
# final segmentation boundary.
# ============================================================

box = np.array(
    [
        int(
            0.25
            * width
        ),

        int(
            0.04
            * height
        ),

        int(
            0.75
            * width
        ),

        int(
            0.94
            * height
        ),
    ],
    dtype=np.float32
)


print(
    "SAM2 bounding box:",
    box.astype(
        np.int32
    )
)


# ============================================================
# PROMPT VISUALIZATION
# ============================================================

prompt_visualization = (
    image_bgr.copy()
)


cv2.rectangle(
    prompt_visualization,

    (
        int(box[0]),
        int(box[1])
    ),

    (
        int(box[2]),
        int(box[3])
    ),

    (255, 255, 255),

    2
)


# ============================================================
# RUN SAM2 SEGMENTATION
# ============================================================

print(
    "\nRunning SAM2 segmentation..."
)


with torch.inference_mode():

    masks, scores, logits = (
        predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True
        )
    )


print(
    "SAM2 candidate scores:",
    scores
)


# ============================================================
# SELECT BEST SAM2 MASK
# ============================================================

best_index = int(
    np.argmax(
        scores
    )
)


sam2_score = float(
    scores[
        best_index
    ]
)


print(
    "Selected mask index:",
    best_index
)


print(
    f"SAM2 predicted score: "
    f"{sam2_score:.4f}"
)


sam2_mask = (
    masks[
        best_index
    ]
    > 0
).astype(
    np.uint8
) * 255


# ============================================================
# MORPHOLOGICAL CLEANUP
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


sam2_mask = cv2.morphologyEx(
    sam2_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


# ============================================================
# KEEP LARGEST SAM2 COMPONENT
# ============================================================

contours, _ = cv2.findContours(
    sam2_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)


if not contours:

    raise RuntimeError(
        "SAM2 did not generate "
        "a valid human contour."
    )


sam2_contour = max(
    contours,
    key=cv2.contourArea
)


sam2_mask[:] = 0


cv2.drawContours(
    sam2_mask,
    [sam2_contour],
    -1,
    255,
    cv2.FILLED
)


sam2_contour_area = (
    cv2.contourArea(
        sam2_contour
    )
)


print(
    f"SAM2 contour area: "
    f"{sam2_contour_area:.1f} pixels"
)


# ============================================================
# SAM2 SEGMENTED THERMAL PERSON
# ============================================================

sam2_segmented = cv2.bitwise_and(
    image_bgr,
    image_bgr,
    mask=sam2_mask
)


# ============================================================
# SAM2 BOUNDARY
# ============================================================

sam2_boundary = (
    image_bgr.copy()
)


cv2.drawContours(
    sam2_boundary,
    [sam2_contour],
    -1,
    (0, 0, 255),
    2
)


# ============================================================
# SAM2 BOUNDARY ONLY
# ============================================================

sam2_boundary_only = np.zeros(
    (
        height,
        width,
        3
    ),
    dtype=np.uint8
)


cv2.drawContours(
    sam2_boundary_only,
    [sam2_contour],
    -1,
    (255, 255, 255),
    2
)


# ============================================================
# CALCULATE OPENCV vs SAM2 AGREEMENT
# ============================================================

opencv_bool = (
    opencv_mask > 0
)

sam2_bool = (
    sam2_mask > 0
)


intersection = np.logical_and(
    opencv_bool,
    sam2_bool
)


union = np.logical_or(
    opencv_bool,
    sam2_bool
)


intersection_pixels = int(
    np.sum(
        intersection
    )
)


union_pixels = int(
    np.sum(
        union
    )
)


opencv_pixels = int(
    np.sum(
        opencv_bool
    )
)


sam2_pixels = int(
    np.sum(
        sam2_bool
    )
)


if union_pixels > 0:

    iou = (
        intersection_pixels
        / union_pixels
    )

else:

    iou = 0.0


denominator = (
    opencv_pixels
    + sam2_pixels
)


if denominator > 0:

    dice = (
        2.0
        * intersection_pixels
        / denominator
    )

else:

    dice = 0.0


# ============================================================
# PRINT AGREEMENT
# ============================================================

print(
    "\n"
    + "=" * 65
)

print(
    "OPENCV vs SAM2 THERMAL AGREEMENT"
)

print(
    "=" * 65
)


print(
    f"OpenCV foreground pixels: "
    f"{opencv_pixels}"
)


print(
    f"SAM2 foreground pixels:   "
    f"{sam2_pixels}"
)


print(
    f"Intersection pixels:       "
    f"{intersection_pixels}"
)


print(
    f"Union pixels:              "
    f"{union_pixels}"
)


print(
    f"IoU agreement:             "
    f"{iou:.4f}"
)


print(
    f"Dice agreement:            "
    f"{dice:.4f}"
)


# ============================================================
# AGREEMENT VISUALIZATION
# ============================================================
#
# WHITE = both OpenCV and SAM2 agree foreground
# RED   = OpenCV only
# BLUE  = SAM2 only
# BLACK = both background
# ============================================================

agreement = np.zeros(
    (
        height,
        width,
        3
    ),
    dtype=np.uint8
)


# Shared foreground = white

agreement[
    intersection
] = (
    255,
    255,
    255
)


# OpenCV only = red in final BGR image

opencv_only = np.logical_and(
    opencv_bool,
    np.logical_not(
        sam2_bool
    )
)


agreement[
    opencv_only
] = (
    0,
    0,
    255
)


# SAM2 only = blue

sam2_only = np.logical_and(
    sam2_bool,
    np.logical_not(
        opencv_bool
    )
)


agreement[
    sam2_only
] = (
    255,
    0,
    0
)


# ============================================================
# COMPARISON IMAGE
# ============================================================

def add_title(
    img,
    title
):

    result = img.copy()


    cv2.rectangle(
        result,
        (0, 0),
        (
            result.shape[1],
            28
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        result,
        title,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    return result


opencv_boundary = (
    image_bgr.copy()
)


opencv_contours, _ = (
    cv2.findContours(
        opencv_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
)


if opencv_contours:

    largest_opencv_contour = max(
        opencv_contours,
        key=cv2.contourArea
    )


    cv2.drawContours(
        opencv_boundary,
        [largest_opencv_contour],
        -1,
        (0, 0, 255),
        2
    )


comparison = np.hstack(
    [
        add_title(
            image_bgr,
            "Original Thermal"
        ),

        add_title(
            opencv_boundary,
            "Classical OpenCV"
        ),

        add_title(
            sam2_boundary,
            "SAM2.1"
        ),
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_sam2_prompt_box.jpg"
    ),
    prompt_visualization
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_sam2_mask.jpg"
    ),
    sam2_mask
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_sam2_segmented_person.jpg"
    ),
    sam2_segmented
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_sam2_boundary.jpg"
    ),
    sam2_boundary
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_sam2_boundary_only.jpg"
    ),
    sam2_boundary_only
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_opencv_vs_sam2.jpg"
    ),
    comparison
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_opencv_sam2_agreement.jpg"
    ),
    agreement
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    OUTPUT_DIR
    / "thermal_sam2_metrics.txt"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CSc 8830 - Module 4\n"
    )

    file.write(
        "Thermal OpenCV vs SAM2.1 Comparison\n"
    )

    file.write(
        "=" * 50
        + "\n\n"
    )

    file.write(
        f"SAM2 predicted score: "
        f"{sam2_score:.4f}\n"
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
        f"{intersection_pixels}\n"
    )

    file.write(
        f"Union pixels: "
        f"{union_pixels}\n"
    )

    file.write(
        f"IoU agreement: "
        f"{iou:.4f}\n"
    )

    file.write(
        f"Dice agreement: "
        f"{dice:.4f}\n"
    )

    file.write(
        "\nNOTE:\n"
    )

    file.write(
        "IoU and Dice represent agreement "
        "between the classical OpenCV "
        "segmentation and SAM2.1. "
        "They do not represent ground-truth "
        "segmentation accuracy.\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nGenerated files:"
)

print(
    "1. thermal_sam2_prompt_box.jpg"
)

print(
    "2. thermal_sam2_mask.jpg"
)

print(
    "3. thermal_sam2_segmented_person.jpg"
)

print(
    "4. thermal_sam2_boundary.jpg"
)

print(
    "5. thermal_sam2_boundary_only.jpg"
)

print(
    "6. thermal_opencv_vs_sam2.jpg"
)

print(
    "7. thermal_opencv_sam2_agreement.jpg"
)

print(
    "8. thermal_sam2_metrics.txt"
)


print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR}"
)


print(
    "\nSAM2 thermal comparison complete."
)

"""
CSc 8830 - Computer Vision
Module 4 - Question 1

Human Boundary Detection from an RGB Image

Classical Computer Vision implementation.
NO machine learning or deep learning is used.

Pipeline:
1. Load RGB image
2. Resize image
3. Initialize GrabCut using foreground/background masks
4. Extract binary foreground
5. Morphological cleanup
6. Remove thin horizontal background structures
7. Select the human connected component
8. Refine the human mask
9. Extract exact contour
10. Save segmentation and boundary results
"""

import cv2
import numpy as np
import os


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "images",
    "rgb",
    "person.jpg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not find image:\n{IMAGE_PATH}"
    )

print("Image loaded successfully.")
print("Original image size:", image.shape)


# =========================================================
# RESIZE
# =========================================================

max_height = 900

if image.shape[0] > max_height:

    scale = max_height / image.shape[0]

    new_width = int(
        image.shape[1] * scale
    )

    image = cv2.resize(
        image,
        (new_width, max_height),
        interpolation=cv2.INTER_AREA
    )


original = image.copy()

height, width = image.shape[:2]

print(
    "Processing size:",
    width,
    "x",
    height
)


# =========================================================
# STEP 1
# GRABCUT MASK INITIALIZATION
# =========================================================

# Start with probable background everywhere.

grabcut_mask = np.full(
    (height, width),
    cv2.GC_PR_BGD,
    dtype=np.uint8
)


# ---------------------------------------------------------
# Definite background
# ---------------------------------------------------------
# Outer image borders cannot belong to the centered person.

border_x = int(width * 0.10)
border_y = int(height * 0.02)

grabcut_mask[:, :border_x] = cv2.GC_BGD
grabcut_mask[:, width - border_x:] = cv2.GC_BGD

grabcut_mask[:border_y, :] = cv2.GC_BGD
grabcut_mask[height - border_y:, :] = cv2.GC_BGD


# ---------------------------------------------------------
# Probable foreground
# ---------------------------------------------------------
# The person in our experimental image is centered.
# Use a vertical region covering the human body.

center_x = width // 2

person_left = int(width * 0.27)
person_right = int(width * 0.73)

person_top = int(height * 0.04)
person_bottom = int(height * 0.96)

grabcut_mask[
    person_top:person_bottom,
    person_left:person_right
] = cv2.GC_PR_FGD


# ---------------------------------------------------------
# Strong foreground seed
# ---------------------------------------------------------
# A narrow center region is safely inside the person's
# torso and legs.

seed_left = int(width * 0.42)
seed_right = int(width * 0.58)

seed_top = int(height * 0.18)
seed_bottom = int(height * 0.78)

grabcut_mask[
    seed_top:seed_bottom,
    seed_left:seed_right
] = cv2.GC_FGD


# =========================================================
# STEP 2
# RUN GRABCUT
# =========================================================

background_model = np.zeros(
    (1, 65),
    np.float64
)

foreground_model = np.zeros(
    (1, 65),
    np.float64
)

print("Running refined GrabCut...")

cv2.grabCut(
    image,
    grabcut_mask,
    None,
    background_model,
    foreground_model,
    10,
    cv2.GC_INIT_WITH_MASK
)


# =========================================================
# STEP 3
# CREATE BINARY MASK
# =========================================================

binary_mask = np.where(
    (grabcut_mask == cv2.GC_FGD) |
    (grabcut_mask == cv2.GC_PR_FGD),
    255,
    0
).astype("uint8")


# =========================================================
# STEP 4
# MORPHOLOGICAL CLEANUP
# =========================================================

kernel_small = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    kernel_small,
    iterations=1
)

binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_CLOSE,
    kernel_small,
    iterations=2
)


# =========================================================
# STEP 5
# REMOVE LONG HORIZONTAL BACKGROUND STRUCTURES
# =========================================================
# The previous segmentation picked up part of the walkway
# because it formed long horizontal structures near the
# hands.
#
# A horizontal morphological opening identifies structures
# that are much wider than normal human limbs.

horizontal_length = max(
    25,
    int(width * 0.10)
)

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (horizontal_length, 3)
)

horizontal_objects = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    horizontal_kernel,
    iterations=1
)


# Restrict removal mostly to the side regions.
# This protects the torso from accidental removal.

side_mask = np.zeros_like(
    binary_mask
)

left_limit = int(width * 0.32)
right_limit = int(width * 0.68)

side_mask[:, :left_limit] = 255
side_mask[:, right_limit:] = 255

horizontal_side_objects = cv2.bitwise_and(
    horizontal_objects,
    side_mask
)

refined_mask = cv2.subtract(
    binary_mask,
    horizontal_side_objects
)


# =========================================================
# STEP 6
# CONNECTED COMPONENT ANALYSIS
# =========================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        refined_mask,
        connectivity=8
    )
)

if num_labels <= 1:
    raise RuntimeError(
        "No foreground object was detected."
    )


# Find component containing / closest to image center.

image_center = np.array(
    [width / 2, height / 2]
)

best_label = None
best_score = -1

for label in range(1, num_labels):

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    cx, cy = centroids[label]

    distance = np.linalg.norm(
        np.array([cx, cy]) -
        image_center
    )

    # Prefer large components located near image center.

    score = area / (
        1.0 + distance
    )

    if score > best_score:

        best_score = score
        best_label = label


human_mask = np.zeros_like(
    refined_mask
)

human_mask[
    labels == best_label
] = 255


# =========================================================
# STEP 7
# FINAL MASK CLEANUP
# =========================================================

final_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

human_mask = cv2.morphologyEx(
    human_mask,
    cv2.MORPH_CLOSE,
    final_kernel,
    iterations=2
)

human_mask = cv2.morphologyEx(
    human_mask,
    cv2.MORPH_OPEN,
    kernel_small,
    iterations=1
)


# =========================================================
# STEP 8
# FIND FINAL HUMAN CONTOUR
# =========================================================

contours, _ = cv2.findContours(
    human_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

if not contours:
    raise RuntimeError(
        "Human contour could not be detected."
    )

human_contour = max(
    contours,
    key=cv2.contourArea
)

human_area = cv2.contourArea(
    human_contour
)

print(
    "Final human contour area:",
    round(human_area, 2),
    "pixels"
)


# =========================================================
# STEP 9
# RECREATE CLEAN MASK FROM FINAL CONTOUR
# =========================================================

final_mask = np.zeros_like(
    human_mask
)

cv2.drawContours(
    final_mask,
    [human_contour],
    -1,
    255,
    thickness=cv2.FILLED
)


# =========================================================
# STEP 10
# SEGMENT HUMAN
# =========================================================

segmented_person = cv2.bitwise_and(
    original,
    original,
    mask=final_mask
)


# =========================================================
# STEP 11
# DRAW HUMAN BOUNDARY
# =========================================================

boundary_result = original.copy()

cv2.drawContours(
    boundary_result,
    [human_contour],
    -1,
    (0, 0, 255),
    3
)


# =========================================================
# STEP 12
# BOUNDARY-ONLY IMAGE
# =========================================================

boundary_only = np.zeros_like(
    original
)

cv2.drawContours(
    boundary_only,
    [human_contour],
    -1,
    (255, 255, 255),
    2
)


# =========================================================
# STEP 13
# CREATE VISUALIZATION OF PROCESS
# =========================================================

mask_bgr = cv2.cvtColor(
    final_mask,
    cv2.COLOR_GRAY2BGR
)

comparison = np.hstack(
    (
        original,
        mask_bgr,
        boundary_result
    )
)


# =========================================================
# STEP 14
# SAVE OUTPUTS
# =========================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "rgb_original.jpg"
    ),
    original
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "rgb_binary_mask.jpg"
    ),
    final_mask
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "rgb_segmented_person.jpg"
    ),
    segmented_person
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "rgb_human_boundary.jpg"
    ),
    boundary_result
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "rgb_boundary_only.jpg"
    ),
    boundary_only
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "rgb_opencv_comparison.jpg"
    ),
    comparison
)


# =========================================================
# RESULTS
# =========================================================

foreground_pixels = cv2.countNonZero(
    final_mask
)

total_pixels = (
    final_mask.shape[0] *
    final_mask.shape[1]
)

foreground_percentage = (
    foreground_pixels /
    total_pixels
) * 100


print()
print("=" * 50)
print("RGB HUMAN BOUNDARY DETECTION COMPLETE")
print("=" * 50)

print(
    "Human contour area:",
    round(human_area, 2),
    "pixels"
)

print(
    "Foreground pixels:",
    foreground_pixels
)

print(
    "Foreground percentage:",
    round(
        foreground_percentage,
        2
    ),
    "%"
)

print()
print(
    "Outputs saved to:",
    OUTPUT_DIR
)

print()
print("Generated files:")
print("1. rgb_original.jpg")
print("2. rgb_binary_mask.jpg")
print("3. rgb_segmented_person.jpg")
print("4. rgb_human_boundary.jpg")
print("5. rgb_boundary_only.jpg")
print("6. rgb_opencv_comparison.jpg")
"""
CSc 8830 - Computer Vision
Module 4 - Question 2

Human Boundary Detection in a Thermal Image
Classical Computer Vision / OpenCV Only

NO machine learning or deep learning is used in this script.

Pipeline:
1. Read false-color thermal image.
2. Convert BGR image to HSV.
3. Separate warm human regions from the cooler background.
4. Apply morphological operations.
5. Find connected foreground components.
6. Select the main central human component.
7. Fill holes and smooth the mask.
8. Extract the exact outer contour.
9. Save mask, segmented person, boundary, and comparison images.
"""

from pathlib import Path

import cv2
import numpy as np


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

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD THERMAL IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not load thermal image:\n{IMAGE_PATH}"
    )


height, width = image.shape[:2]


print("=" * 65)
print("THERMAL HUMAN BOUNDARY DETECTION")
print("CLASSICAL OPENCV - NO ML/DL")
print("=" * 65)

print("\nImage loaded successfully.")
print(
    f"Image size: "
    f"{width} x {height}"
)


# ============================================================
# STEP 1 - LIGHT SMOOTHING
# ============================================================

blurred = cv2.GaussianBlur(
    image,
    (5, 5),
    0
)


# ============================================================
# STEP 2 - CONVERT TO HSV
# ============================================================

hsv = cv2.cvtColor(
    blurred,
    cv2.COLOR_BGR2HSV
)

h_channel, s_channel, v_channel = cv2.split(
    hsv
)


# ============================================================
# STEP 3 - THERMAL COLOR SEGMENTATION
# ============================================================
#
# The new image is a false-color thermal image.
#
# Cooler background pixels are mainly blue/dark-blue.
# Warmer human pixels move toward:
#
# cyan
# green
# yellow
# orange
# red
# white
#
# Therefore we remove the dominant cool-blue background
# rather than trying to detect a single exact body color.
#
# OpenCV HSV hue:
#
# 0   -> red
# ~30 -> yellow
# ~60 -> green
# ~90 -> cyan
# ~120 -> blue
#
# The main cool background is expected in the blue range.
# ============================================================

cool_blue_mask = cv2.inRange(
    hsv,
    np.array(
        [90, 45, 15],
        dtype=np.uint8
    ),
    np.array(
        [140, 255, 255],
        dtype=np.uint8
    )
)


# Invert:
# blue background -> black
# warmer regions  -> white

thermal_foreground = cv2.bitwise_not(
    cool_blue_mask
)


# ============================================================
# STEP 4 - REMOVE VERY DARK PIXELS
# ============================================================
#
# Dark pixels may also appear outside the actual subject.
# ============================================================

brightness_mask = cv2.inRange(
    v_channel,
    25,
    255
)


thermal_foreground = cv2.bitwise_and(
    thermal_foreground,
    brightness_mask
)


# ============================================================
# STEP 5 - CENTRAL SEARCH REGION
# ============================================================
#
# The subject occupies the central region of this frame.
#
# This is only a geometric classical-CV constraint.
# It is not an ML/DL detector.
# ============================================================

search_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


left = int(
    width * 0.12
)

right = int(
    width * 0.88
)

top = int(
    height * 0.02
)

bottom = int(
    height * 0.98
)


search_mask[
    top:bottom,
    left:right
] = 255


thermal_foreground = cv2.bitwise_and(
    thermal_foreground,
    search_mask
)


print(
    "\nSearch region:"
)

print(
    f"x = {left}:{right}"
)

print(
    f"y = {top}:{bottom}"
)


# ============================================================
# STEP 6 - MORPHOLOGICAL OPENING
# ============================================================
#
# Remove isolated thermal noise.
# ============================================================

opening_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)


thermal_foreground = cv2.morphologyEx(
    thermal_foreground,
    cv2.MORPH_OPEN,
    opening_kernel,
    iterations=1
)


# ============================================================
# STEP 7 - MORPHOLOGICAL CLOSING
# ============================================================
#
# Connect nearby warm regions belonging to the body.
# ============================================================

closing_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (9, 9)
)


thermal_foreground = cv2.morphologyEx(
    thermal_foreground,
    cv2.MORPH_CLOSE,
    closing_kernel,
    iterations=2
)


# ============================================================
# STEP 8 - CONNECTED COMPONENT ANALYSIS
# ============================================================

(
    number_labels,
    labels,
    stats,
    centroids
) = cv2.connectedComponentsWithStats(
    thermal_foreground,
    connectivity=8
)


print(
    f"\nConnected foreground components: "
    f"{number_labels - 1}"
)


image_center = np.array(
    [
        width / 2.0,
        height / 2.0
    ]
)


best_label = None

best_score = -1


for label in range(
    1,
    number_labels
):

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    x = stats[
        label,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        label,
        cv2.CC_STAT_TOP
    ]

    component_width = stats[
        label,
        cv2.CC_STAT_WIDTH
    ]

    component_height = stats[
        label,
        cv2.CC_STAT_HEIGHT
    ]


    center = centroids[
        label
    ]


    aspect_ratio = (
        component_height
        / max(
            component_width,
            1
        )
    )


    distance = np.linalg.norm(
        center
        - image_center
    )


    print(
        f"Component {label}: "
        f"area={area}, "
        f"bbox="
        f"({x},{y},"
        f"{component_width},"
        f"{component_height}), "
        f"aspect={aspect_ratio:.2f}, "
        f"center_distance={distance:.1f}"
    )


    # --------------------------------------------------------
    # BASIC HUMAN-SHAPE FILTERS
    # --------------------------------------------------------

    if area < (
        height
        * width
        * 0.01
    ):
        continue


    if component_height < (
        height
        * 0.30
    ):
        continue


    if component_width > (
        width
        * 0.80
    ):
        continue


    # --------------------------------------------------------
    # COMPONENT SCORE
    # --------------------------------------------------------
    #
    # Prefer:
    # - large regions
    # - tall regions
    # - regions near image center
    # --------------------------------------------------------

    center_score = (
        1.0
        / (
            1.0
            + distance
        )
    )


    score = (
        area
        * (
            1.0
            + min(
                aspect_ratio,
                4.0
            )
        )
        * (
            1.0
            + 50.0
            * center_score
        )
    )


    if score > best_score:

        best_score = score

        best_label = label


# ============================================================
# STEP 9 - FALLBACK
# ============================================================

if best_label is None:

    print(
        "\nNo component passed "
        "all human-shape filters."
    )

    print(
        "Using largest central "
        "foreground component."
    )


    best_fallback_score = -1


    for label in range(
        1,
        number_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]


        if area < 50:
            continue


        center = centroids[
            label
        ]


        distance = np.linalg.norm(
            center
            - image_center
        )


        fallback_score = (
            area
            / (
                1.0
                + distance
            )
        )


        if (
            fallback_score
            > best_fallback_score
        ):

            best_fallback_score = (
                fallback_score
            )

            best_label = label


if best_label is None:

    raise RuntimeError(
        "Unable to locate the "
        "thermal human foreground."
    )


# ============================================================
# STEP 10 - CREATE HUMAN MASK
# ============================================================

human_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


human_mask[
    labels == best_label
] = 255


selected_area = stats[
    best_label,
    cv2.CC_STAT_AREA
]


selected_x = stats[
    best_label,
    cv2.CC_STAT_LEFT
]

selected_y = stats[
    best_label,
    cv2.CC_STAT_TOP
]

selected_w = stats[
    best_label,
    cv2.CC_STAT_WIDTH
]

selected_h = stats[
    best_label,
    cv2.CC_STAT_HEIGHT
]


print(
    "\nSelected human component:"
)

print(
    f"Area: "
    f"{selected_area} pixels"
)

print(
    f"Bounding box: "
    f"({selected_x}, "
    f"{selected_y}, "
    f"{selected_w}, "
    f"{selected_h})"
)


# ============================================================
# STEP 11 - FILL INTERNAL HOLES
# ============================================================
#
# The human should form one solid silhouette.
# Flood-fill the background and recover holes.
# ============================================================

flood_fill = human_mask.copy()


flood_mask = np.zeros(
    (
        height + 2,
        width + 2
    ),
    dtype=np.uint8
)


cv2.floodFill(
    flood_fill,
    flood_mask,
    (0, 0),
    255
)


flood_inverse = cv2.bitwise_not(
    flood_fill
)


human_mask = (
    human_mask
    | flood_inverse
)


# ============================================================
# STEP 12 - FINAL MASK SMOOTHING
# ============================================================

final_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


human_mask = cv2.morphologyEx(
    human_mask,
    cv2.MORPH_CLOSE,
    final_kernel,
    iterations=1
)


human_mask = cv2.morphologyEx(
    human_mask,
    cv2.MORPH_OPEN,
    np.ones(
        (3, 3),
        dtype=np.uint8
    ),
    iterations=1
)


# ============================================================
# STEP 13 - EXTRACT FINAL HUMAN CONTOUR
# ============================================================

contours, _ = cv2.findContours(
    human_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)


if not contours:

    raise RuntimeError(
        "No final human contour found."
    )


human_contour = max(
    contours,
    key=cv2.contourArea
)


contour_area = cv2.contourArea(
    human_contour
)


# Rebuild the final mask using only
# the largest contour.

human_mask[:] = 0


cv2.drawContours(
    human_mask,
    [human_contour],
    -1,
    255,
    cv2.FILLED
)


print(
    f"Final contour area: "
    f"{contour_area:.1f} pixels"
)


# ============================================================
# STEP 14 - SEGMENTED THERMAL PERSON
# ============================================================

segmented_person = cv2.bitwise_and(
    image,
    image,
    mask=human_mask
)


# ============================================================
# STEP 15 - HUMAN BOUNDARY
# ============================================================

human_boundary = image.copy()


cv2.drawContours(
    human_boundary,
    [human_contour],
    -1,
    (0, 0, 255),
    2
)


# ============================================================
# STEP 16 - BOUNDARY-ONLY IMAGE
# ============================================================

boundary_only = np.zeros(
    (
        height,
        width,
        3
    ),
    dtype=np.uint8
)


cv2.drawContours(
    boundary_only,
    [human_contour],
    -1,
    (255, 255, 255),
    2
)


# ============================================================
# STEP 17 - SEARCH REGION VISUALIZATION
# ============================================================

search_visualization = (
    image.copy()
)


cv2.rectangle(
    search_visualization,
    (
        left,
        top
    ),
    (
        right,
        bottom
    ),
    (255, 255, 255),
    2
)


# ============================================================
# STEP 18 - CREATE COMPARISON IMAGE
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
            30
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        result,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    return result


mask_bgr = cv2.cvtColor(
    human_mask,
    cv2.COLOR_GRAY2BGR
)


original_panel = add_title(
    image,
    "Original Thermal"
)


mask_panel = add_title(
    mask_bgr,
    "OpenCV Human Mask"
)


boundary_panel = add_title(
    human_boundary,
    "Detected Boundary"
)


comparison = np.hstack(
    [
        original_panel,
        mask_panel,
        boundary_panel
    ]
)


# ============================================================
# STEP 19 - SAVE OUTPUTS
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_original.jpg"
    ),
    image
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_search_region.jpg"
    ),
    search_visualization
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_initial_mask.jpg"
    ),
    thermal_foreground
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_binary_mask.jpg"
    ),
    human_mask
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_segmented_person.jpg"
    ),
    segmented_person
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_human_boundary.jpg"
    ),
    human_boundary
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_boundary_only.jpg"
    ),
    boundary_only
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "thermal_opencv_comparison.jpg"
    ),
    comparison
)


# ============================================================
# STEP 20 - STATISTICS
# ============================================================

foreground_pixels = np.count_nonzero(
    human_mask
)


foreground_percentage = (
    foreground_pixels
    / (
        height
        * width
    )
    * 100
)


print("\n" + "=" * 65)
print("THERMAL SEGMENTATION COMPLETE")
print("=" * 65)


print(
    f"Foreground pixels: "
    f"{foreground_pixels}"
)


print(
    f"Foreground percentage: "
    f"{foreground_percentage:.2f}%"
)


print("\nGenerated files:")

print(
    "1. thermal_original.jpg"
)

print(
    "2. thermal_search_region.jpg"
)

print(
    "3. thermal_initial_mask.jpg"
)

print(
    "4. thermal_binary_mask.jpg"
)

print(
    "5. thermal_segmented_person.jpg"
)

print(
    "6. thermal_human_boundary.jpg"
)

print(
    "7. thermal_boundary_only.jpg"
)

print(
    "8. thermal_opencv_comparison.jpg"
)


print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR}"
)


print("\nDone.")
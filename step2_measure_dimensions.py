"""
CSc 8830 - Computer Vision
Module 2 Assignment

Step 2:
Estimate the real-world 2D dimensions of a planar rectangular object
using perspective projection equations and the calibrated smartphone camera.

Usage:
    python step2_measure_dimensions.py

The user:
1. Enters an image path.
2. Enters the camera-to-object distance.
3. Clicks four object corners:
       1. Top-left
       2. Top-right
       3. Bottom-right
       4. Bottom-left
4. The program estimates real-world width and height.
"""

import cv2
import numpy as np
import os
from pathlib import Path


# ============================================================
# FILE LOCATIONS
# ============================================================

CALIBRATION_FILE = os.path.join(
    "calibration_output",
    "calibration_results.npz"
)

OUTPUT_FOLDER = "step2_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

if not os.path.exists(CALIBRATION_FILE):

    print("\nERROR:")
    print("Calibration file not found:")
    print(CALIBRATION_FILE)

    print(
        "\nRun Step 1 calibration first."
    )

    raise SystemExit


calibration = np.load(
    CALIBRATION_FILE,
    allow_pickle=True
)

camera_matrix = calibration["camera_matrix"]

dist_coeffs = calibration[
    "distortion_coefficients"
]


if "image_size" in calibration:

    calibration_size = calibration[
        "image_size"
    ]

    calibration_width = int(
        calibration_size[0]
    )

    calibration_height = int(
        calibration_size[1]
    )

else:

    calibration_width = 1200
    calibration_height = 1600


print("\n============================================")
print("     STEP 2 - OBJECT DIMENSION MEASUREMENT")
print("============================================")

print("\nCamera calibration successfully loaded.")

print("\nCamera Matrix K:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)


# ============================================================
# IMAGE INPUT
# ============================================================

print(
    "\nPut your object photograph inside "
    "the measurement_images folder."
)

image_name = input(
    "\nEnter image filename "
    "(example: object1.jpeg): "
).strip()


image_path = os.path.join(
    "measurement_images",
    image_name
)


if not os.path.exists(image_path):

    print("\nERROR: Image not found:")
    print(image_path)

    raise SystemExit


image = cv2.imread(image_path)


if image is None:

    print("\nERROR: OpenCV could not open image.")

    raise SystemExit


image_height, image_width = image.shape[:2]


print(
    f"\nMeasurement image resolution: "
    f"{image_width} x {image_height}"
)

print(
    f"Calibration image resolution: "
    f"{calibration_width} x "
    f"{calibration_height}"
)


# ============================================================
# ADJUST CAMERA MATRIX IF IMAGE RESOLUTION CHANGED
# ============================================================

K = camera_matrix.copy()


if (
    image_width != calibration_width
    or
    image_height != calibration_height
):

    scale_x = (
        image_width /
        calibration_width
    )

    scale_y = (
        image_height /
        calibration_height
    )


    calibration_aspect = (
        calibration_width /
        calibration_height
    )

    image_aspect = (
        image_width /
        image_height
    )


    if abs(
        calibration_aspect -
        image_aspect
    ) > 0.02:

        print("\nWARNING:")
        print(
            "Measurement image aspect ratio "
            "differs from calibration images."
        )

        print(
            "For accurate results, use the "
            "same phone camera, orientation, "
            "zoom and resolution."
        )


    # Scale intrinsic parameters
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x

    K[1, 1] *= scale_y
    K[1, 2] *= scale_y


    print(
        "\nCamera matrix was scaled "
        "to match the measurement image."
    )


# ============================================================
# ENTER CAMERA DISTANCE
# ============================================================

distance_m = float(
    input(
        "\nEnter camera-to-object distance "
        "in meters (example 2.5): "
    )
)


if distance_m <= 0:

    print("ERROR: Distance must be positive.")

    raise SystemExit


# Convert meters to millimeters

Z = distance_m * 1000.0


print(
    f"\nDistance used: "
    f"{Z:.2f} mm"
)


# ============================================================
# DISPLAY IMAGE
# ============================================================

# Resize only for display purposes.
# Calculations still use original image coordinates.

MAX_WIDTH = 1000
MAX_HEIGHT = 750


display_scale = min(
    MAX_WIDTH / image_width,
    MAX_HEIGHT / image_height,
    1.0
)


display_width = int(
    image_width * display_scale
)

display_height = int(
    image_height * display_scale
)


display_image = cv2.resize(
    image,
    (
        display_width,
        display_height
    )
)


clicked_points = []


corner_names = [
    "TOP-LEFT",
    "TOP-RIGHT",
    "BOTTOM-RIGHT",
    "BOTTOM-LEFT"
]


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global clicked_points
    global display_image


    if (
        event == cv2.EVENT_LBUTTONDOWN
        and
        len(clicked_points) < 4
    ):

        # Convert display coordinates back
        # into original image coordinates.

        original_x = (
            x / display_scale
        )

        original_y = (
            y / display_scale
        )


        clicked_points.append(
            (
                original_x,
                original_y
            )
        )


        point_number = len(
            clicked_points
        )


        cv2.circle(
            display_image,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            display_image,
            str(point_number),
            (
                x + 10,
                y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        print(
            f"Point {point_number} "
            f"({corner_names[point_number - 1]}): "
            f"({original_x:.2f}, "
            f"{original_y:.2f})"
        )


# ============================================================
# COLLECT FOUR CORNERS
# ============================================================

window_name = (
    "Select Object Corners"
)

cv2.namedWindow(
    window_name
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


print("\n--------------------------------------------")
print("CLICK THE FOUR OBJECT CORNERS IN THIS ORDER:")
print("--------------------------------------------")

print("1. TOP-LEFT")
print("2. TOP-RIGHT")
print("3. BOTTOM-RIGHT")
print("4. BOTTOM-LEFT")

print("\nPress R to restart.")
print("Press ESC to cancel.")


while True:

    cv2.imshow(
        window_name,
        display_image
    )


    key = cv2.waitKey(20) & 0xFF


    if len(clicked_points) == 4:

        break


    if key == ord("r"):

        clicked_points = []

        display_image = cv2.resize(
            image,
            (
                display_width,
                display_height
            )
        )

        print("\nPoints cleared.")
        print("Select the four corners again.")


    if key == 27:

        cv2.destroyAllWindows()

        print("\nMeasurement cancelled.")

        raise SystemExit


cv2.destroyAllWindows()


# ============================================================
# CONVERT PIXEL POINTS INTO NORMALIZED CAMERA COORDINATES
# ============================================================

pixel_points = np.array(
    clicked_points,
    dtype=np.float64
).reshape(-1, 1, 2)


# This operation:
#
# 1. Corrects lens distortion
# 2. Removes the effect of the camera intrinsic matrix
#
# Output corresponds approximately to:
#
# x = X/Z
# y = Y/Z

normalized_points = cv2.undistortPoints(
    pixel_points,
    K,
    dist_coeffs
)


normalized_points = (
    normalized_points
    .reshape(-1, 2)
)


# ============================================================
# PERSPECTIVE PROJECTION
# ============================================================

# Perspective equations:
#
# x = X / Z
# y = Y / Z
#
# Therefore:
#
# X = xZ
# Y = yZ
#
# For a planar object approximately parallel
# to the camera image plane, all four corners
# have approximately the same Z.


world_points = []


for point in normalized_points:

    xn = point[0]
    yn = point[1]

    X = xn * Z
    Y = yn * Z

    world_points.append(
        [
            X,
            Y,
            Z
        ]
    )


world_points = np.array(
    world_points
)


# ============================================================
# COMPUTE REAL-WORLD DISTANCES
# ============================================================

def distance_between_points(
    point1,
    point2
):

    return np.linalg.norm(
        point1 - point2
    )


# Points:
#
# 0 = Top-left
# 1 = Top-right
# 2 = Bottom-right
# 3 = Bottom-left


top_width = distance_between_points(
    world_points[0],
    world_points[1]
)


bottom_width = distance_between_points(
    world_points[3],
    world_points[2]
)


left_height = distance_between_points(
    world_points[0],
    world_points[3]
)


right_height = distance_between_points(
    world_points[1],
    world_points[2]
)


estimated_width = (
    top_width +
    bottom_width
) / 2.0


estimated_height = (
    left_height +
    right_height
) / 2.0


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("================================================")
print("        REAL-WORLD DIMENSION RESULTS")
print("================================================")

print(
    f"\nCamera distance: "
    f"{distance_m:.3f} meters"
)


print(
    f"\nTop width: "
    f"{top_width:.2f} mm"
)

print(
    f"Bottom width: "
    f"{bottom_width:.2f} mm"
)


print(
    f"\nLeft height: "
    f"{left_height:.2f} mm"
)

print(
    f"Right height: "
    f"{right_height:.2f} mm"
)


print("\nFINAL ESTIMATED DIMENSIONS:")


print(
    f"\nWidth  = "
    f"{estimated_width:.2f} mm"
)

print(
    f"       = "
    f"{estimated_width / 10:.2f} cm"
)


print(
    f"\nHeight = "
    f"{estimated_height:.2f} mm"
)

print(
    f"       = "
    f"{estimated_height / 10:.2f} cm"
)


print("\n================================================")


# ============================================================
# CREATE ANNOTATED RESULT IMAGE
# ============================================================

result_image = image.copy()


integer_points = [
    (
        int(point[0]),
        int(point[1])
    )
    for point in clicked_points
]


# Draw polygon

for i in range(4):

    start = integer_points[i]

    end = integer_points[
        (i + 1) % 4
    ]

    cv2.line(
        result_image,
        start,
        end,
        (0, 255, 0),
        4
    )


# Draw corner points

for i, point in enumerate(
    integer_points
):

    cv2.circle(
        result_image,
        point,
        8,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        result_image,
        str(i + 1),
        (
            point[0] + 10,
            point[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


# Add measurement text

text1 = (
    f"Width: "
    f"{estimated_width / 10:.2f} cm"
)

text2 = (
    f"Height: "
    f"{estimated_height / 10:.2f} cm"
)

text3 = (
    f"Distance: "
    f"{distance_m:.2f} m"
)


cv2.putText(
    result_image,
    text1,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)


cv2.putText(
    result_image,
    text2,
    (30, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)


cv2.putText(
    result_image,
    text3,
    (30, 140),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)


output_filename = (
    Path(image_name).stem
    + "_measurement.jpg"
)


output_path = os.path.join(
    OUTPUT_FOLDER,
    output_filename
)


cv2.imwrite(
    output_path,
    result_image
)


print(
    "\nAnnotated result saved to:"
)

print(output_path)

print(
    "\nStep 2 measurement complete."
)
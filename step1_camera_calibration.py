"""
CSc 8830 - Computer Vision
Module 2 Assignment
Step 1: Smartphone Camera Calibration using OpenCV

This program:
1. Reads checkerboard images from calibration_images/
2. Detects 9 x 6 internal checkerboard corners
3. Performs camera calibration
4. Calculates reprojection errors
5. Saves the camera matrix and distortion coefficients
6. Saves detected-corner images
7. Creates an undistorted sample image

Run:
    python step1_camera_calibration.py
"""

import cv2
import numpy as np
import os
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

# Your checkerboard has 10 squares x 7 squares.
# Therefore, it contains 9 x 6 INTERNAL corners.
CHECKERBOARD = (9, 6)

# Temporary value.
# Replace this later with the actual physical width
# of one checkerboard square on your laptop screen.
SQUARE_SIZE_MM = 25.0

IMAGE_FOLDER = "calibration_images"
OUTPUT_FOLDER = "calibration_output"
CORNERS_FOLDER = os.path.join(OUTPUT_FOLDER, "detected_corners")


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CORNERS_FOLDER, exist_ok=True)


# ============================================================
# LOAD IMAGE FILES
# ============================================================

image_folder_path = Path(IMAGE_FOLDER)

if not image_folder_path.exists():
    print("\nERROR:")
    print(f"Folder '{IMAGE_FOLDER}' does not exist.")
    print("Create the folder and put your calibration images inside.")
    raise SystemExit


# This avoids the previous Windows duplicate-image problem.
images = sorted(
    [
        str(file)
        for file in image_folder_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)


print("\n==============================================")
print("       SMARTPHONE CAMERA CALIBRATION")
print("==============================================")

print(f"\nFound {len(images)} calibration images.\n")


if len(images) == 0:
    print("ERROR: No calibration images were found.")
    print(f"Put your photos inside: {IMAGE_FOLDER}")
    raise SystemExit


# ============================================================
# PREPARE REAL-WORLD CHECKERBOARD POINTS
# ============================================================

# Creates points such as:
#
# (0,0,0)
# (25,0,0)
# (50,0,0)
# ...
#
# Z = 0 because the checkerboard is planar.

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    dtype=np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE_MM


# ============================================================
# STORAGE FOR CALIBRATION POINTS
# ============================================================

objpoints = []
imgpoints = []

successful_images = []
failed_images = []

image_size = None
sample_image = None


# Criteria used for subpixel corner refinement.
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ============================================================
# DETECT CHECKERBOARD CORNERS
# ============================================================

print("Detecting checkerboard corners...\n")


for filename in images:

    image = cv2.imread(filename)

    if image is None:
        print(f"[ERROR] {os.path.basename(filename)} could not be opened.")
        failed_images.append(filename)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    current_size = (gray.shape[1], gray.shape[0])


    # All calibration images should have the same resolution.
    if image_size is None:
        image_size = current_size

    elif current_size != image_size:
        print(
            f"[SKIP] {os.path.basename(filename)} "
            f"- resolution {current_size} does not match {image_size}"
        )

        failed_images.append(filename)
        continue


    # Detect internal checkerboard corners.
    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    )


    if found:

        # Improve corner locations to subpixel accuracy.
        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )


        objpoints.append(objp.copy())
        imgpoints.append(refined_corners)

        successful_images.append(filename)


        if sample_image is None:
            sample_image = image.copy()


        print(f"[OK]   {os.path.basename(filename)}")


        # Draw checkerboard corners for assignment evidence.
        corner_display = image.copy()

        cv2.drawChessboardCorners(
            corner_display,
            CHECKERBOARD,
            refined_corners,
            found
        )


        output_name = (
            "corners_"
            + Path(filename).stem
            + ".jpg"
        )

        output_path = os.path.join(
            CORNERS_FOLDER,
            output_name
        )

        cv2.imwrite(
            output_path,
            corner_display
        )


    else:

        print(f"[MISS] {os.path.basename(filename)}")

        failed_images.append(filename)


# ============================================================
# DETECTION SUMMARY
# ============================================================

print("\n----------------------------------------------")
print(f"Total images:              {len(images)}")
print(f"Successful detections:     {len(successful_images)}")
print(f"Failed/skipped detections: {len(failed_images)}")
print("----------------------------------------------")


if len(successful_images) < 5:

    print("\nERROR:")
    print("Not enough valid checkerboard images.")
    print("At least 5 valid views are required.")
    print("15-25 good views are recommended.")

    raise SystemExit


# ============================================================
# CAMERA CALIBRATION
# ============================================================

print("\nPerforming camera calibration...")


rms, camera_matrix, distortion_coefficients, rvecs, tvecs = (
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
)


# ============================================================
# CALCULATE REPROJECTION ERROR
# ============================================================

per_view_errors = []


for i in range(len(objpoints)):

    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients
    )


    # OpenCV 5 can produce different internal array shapes.
    # Convert both sets explicitly to Nx2 float64 arrays.
    observed = (
        imgpoints[i]
        .reshape(-1, 2)
        .astype(np.float64)
    )

    projected = (
        projected_points
        .reshape(-1, 2)
        .astype(np.float64)
    )


    difference = observed - projected


    # RMS pixel error for this calibration view.
    error = np.sqrt(
        np.mean(
            np.sum(
                difference ** 2,
                axis=1
            )
        )
    )


    per_view_errors.append(float(error))


mean_reprojection_error = float(
    np.mean(per_view_errors)
)

median_reprojection_error = float(
    np.median(per_view_errors)
)

max_reprojection_error = float(
    np.max(per_view_errors)
)

min_reprojection_error = float(
    np.min(per_view_errors)
)


# ============================================================
# EXTRACT INTRINSIC CAMERA PARAMETERS
# ============================================================

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]

cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("======================================================")
print("             CAMERA CALIBRATION COMPLETE")
print("======================================================")

print(f"\nCheckerboard inner corners: {CHECKERBOARD[0]} x {CHECKERBOARD[1]}")

print(
    f"Square size used: "
    f"{SQUARE_SIZE_MM:.3f} mm"
)

print(
    f"Image resolution: "
    f"{image_size[0]} x {image_size[1]} pixels"
)

print(
    f"Valid calibration images: "
    f"{len(successful_images)}"
)


print("\nCamera Intrinsic Matrix K:\n")

print(camera_matrix)


print("\nIntrinsic parameters:")

print(f"fx = {fx:.6f} pixels")
print(f"fy = {fy:.6f} pixels")
print(f"cx = {cx:.6f} pixels")
print(f"cy = {cy:.6f} pixels")


print("\nDistortion coefficients:\n")

print(distortion_coefficients)


print("\nCalibration error statistics:")

print(
    f"OpenCV RMS reprojection error: "
    f"{rms:.6f} pixels"
)

print(
    f"Mean per-view error: "
    f"{mean_reprojection_error:.6f} pixels"
)

print(
    f"Median per-view error: "
    f"{median_reprojection_error:.6f} pixels"
)

print(
    f"Minimum per-view error: "
    f"{min_reprojection_error:.6f} pixels"
)

print(
    f"Maximum per-view error: "
    f"{max_reprojection_error:.6f} pixels"
)


print("\nPer-image reprojection errors:\n")

for filename, error in zip(
    successful_images,
    per_view_errors
):

    print(
        f"{os.path.basename(filename):25s} "
        f"{error:.6f} pixels"
    )


# ============================================================
# SAVE CALIBRATION DATA
# ============================================================

np.savez(
    os.path.join(
        OUTPUT_FOLDER,
        "calibration_results.npz"
    ),

    camera_matrix=camera_matrix,

    distortion_coefficients=distortion_coefficients,

    rvecs=np.array(
        rvecs,
        dtype=object
    ),

    tvecs=np.array(
        tvecs,
        dtype=object
    ),

    rms=rms,

    mean_reprojection_error=mean_reprojection_error,

    per_view_errors=np.array(
        per_view_errors
    ),

    image_size=np.array(
        image_size
    ),

    square_size_mm=SQUARE_SIZE_MM
)


# ============================================================
# SAVE HUMAN-READABLE TEXT RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_FOLDER,
    "calibration_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CSc 8830 COMPUTER VISION\n"
    )

    file.write(
        "CAMERA CALIBRATION RESULTS\n"
    )

    file.write(
        "========================================\n\n"
    )


    file.write(
        f"Checkerboard Inner Corners: "
        f"{CHECKERBOARD[0]} x {CHECKERBOARD[1]}\n"
    )

    file.write(
        f"Square Size Used: "
        f"{SQUARE_SIZE_MM:.3f} mm\n"
    )

    file.write(
        f"Image Resolution: "
        f"{image_size[0]} x "
        f"{image_size[1]} pixels\n"
    )

    file.write(
        f"Total Images: "
        f"{len(images)}\n"
    )

    file.write(
        f"Successful Images: "
        f"{len(successful_images)}\n\n"
    )


    file.write(
        "CAMERA INTRINSIC MATRIX K\n"
    )

    file.write(
        str(camera_matrix)
    )

    file.write("\n\n")


    file.write(
        "INTRINSIC PARAMETERS\n"
    )

    file.write(
        f"fx = {fx:.6f} pixels\n"
    )

    file.write(
        f"fy = {fy:.6f} pixels\n"
    )

    file.write(
        f"cx = {cx:.6f} pixels\n"
    )

    file.write(
        f"cy = {cy:.6f} pixels\n\n"
    )


    file.write(
        "DISTORTION COEFFICIENTS\n"
    )

    file.write(
        str(distortion_coefficients)
    )

    file.write("\n\n")


    file.write(
        "CALIBRATION ERROR STATISTICS\n"
    )

    file.write(
        f"OpenCV RMS Error: "
        f"{rms:.6f} pixels\n"
    )

    file.write(
        f"Mean Error: "
        f"{mean_reprojection_error:.6f} pixels\n"
    )

    file.write(
        f"Median Error: "
        f"{median_reprojection_error:.6f} pixels\n"
    )

    file.write(
        f"Minimum Error: "
        f"{min_reprojection_error:.6f} pixels\n"
    )

    file.write(
        f"Maximum Error: "
        f"{max_reprojection_error:.6f} pixels\n\n"
    )


    file.write(
        "PER-IMAGE REPROJECTION ERROR\n"
    )

    for filename, error in zip(
        successful_images,
        per_view_errors
    ):

        file.write(
            f"{os.path.basename(filename)}: "
            f"{error:.6f} pixels\n"
        )


# ============================================================
# CREATE UNDISTORTED SAMPLE IMAGE
# ============================================================

if sample_image is not None:

    height, width = sample_image.shape[:2]


    new_camera_matrix, roi = (
        cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coefficients,
            (width, height),
            1,
            (width, height)
        )
    )


    undistorted = cv2.undistort(
        sample_image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix
    )


    undistorted_path = os.path.join(
        OUTPUT_FOLDER,
        "undistorted_sample.jpg"
    )


    cv2.imwrite(
        undistorted_path,
        undistorted
    )


# ============================================================
# FINISHED
# ============================================================

print("\n======================================================")

print("Files saved inside:")
print(OUTPUT_FOLDER)

print("\nImportant output files:")

print(
    "1. calibration_results.npz"
)

print(
    "2. calibration_results.txt"
)

print(
    "3. undistorted_sample.jpg"
)

print(
    "4. detected_corners/"
)

print("======================================================\n")
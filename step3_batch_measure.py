import cv2
import numpy as np
import os
import csv
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

CALIBRATION_FILE = os.path.join(
    "calibration_output",
    "calibration_results.npz"
)

IMAGE_FOLDER = "measurement_images"

OUTPUT_FOLDER = "step3_output"

RESULTS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "step3_measurements.csv"
)

DISTANCE_M = 2.5

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD CALIBRATION
# ============================================================

if not os.path.exists(
    CALIBRATION_FILE
):

    print(
        "ERROR: calibration_results.npz not found."
    )

    raise SystemExit


calibration = np.load(
    CALIBRATION_FILE,
    allow_pickle=True
)

camera_matrix = calibration[
    "camera_matrix"
]

dist_coeffs = calibration[
    "distortion_coefficients"
]

calibration_size = calibration[
    "image_size"
]

calibration_width = int(
    calibration_size[0]
)

calibration_height = int(
    calibration_size[1]
)


print("\n============================================")
print("       STEP 3 BATCH MEASUREMENT")
print("============================================")

print("\nCamera calibration loaded.")

print(
    f"Calibration resolution: "
    f"{calibration_width} x "
    f"{calibration_height}"
)

print(
    f"Distance used for all images: "
    f"{DISTANCE_M:.2f} m"
)


# ============================================================
# FIND IMAGES
# ============================================================

image_folder = Path(
    IMAGE_FOLDER
)

images = sorted(
    [
        path
        for path in image_folder.iterdir()
        if path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png"
        }
    ]
)


print(
    f"\nFound {len(images)} images."
)


if len(images) == 0:

    print(
        "\nNo images found in measurement_images."
    )

    raise SystemExit


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

for trial_number, image_path in enumerate(
    images,
    start=1
):

    print("\n")
    print("============================================")
    print(
        f"TRIAL {trial_number} / {len(images)}"
    )
    print(
        f"Image: {image_path.name}"
    )
    print("============================================")


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            "Could not open image. Skipping."
        )

        continue


    image_height, image_width = (
        image.shape[:2]
    )


    print(
        f"Image resolution: "
        f"{image_width} x "
        f"{image_height}"
    )


    # ========================================================
    # SCALE CAMERA MATRIX
    # ========================================================

    K = camera_matrix.copy()


    scale_x = (
        image_width /
        calibration_width
    )

    scale_y = (
        image_height /
        calibration_height
    )


    K[0, 0] *= scale_x
    K[0, 2] *= scale_x

    K[1, 1] *= scale_y
    K[1, 2] *= scale_y


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    MAX_WIDTH = 1000
    MAX_HEIGHT = 750


    display_scale = min(
        MAX_WIDTH / image_width,
        MAX_HEIGHT / image_height,
        1.0
    )


    display_width = int(
        image_width *
        display_scale
    )

    display_height = int(
        image_height *
        display_scale
    )


    display_image = cv2.resize(
        image,
        (
            display_width,
            display_height
        )
    )


    original_display = (
        display_image.copy()
    )


    clicked_points = []


    # ========================================================
    # MOUSE CALLBACK
    # ========================================================

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and
            len(clicked_points) < 4
        ):

            original_x = (
                x /
                display_scale
            )

            original_y = (
                y /
                display_scale
            )


            clicked_points.append(
                (
                    original_x,
                    original_y
                )
            )


            point_number = (
                len(clicked_points)
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
                f"Point {point_number}: "
                f"({original_x:.2f}, "
                f"{original_y:.2f})"
            )


    # ========================================================
    # GET FOUR CORNERS
    # ========================================================

    window_name = (
        f"Trial {trial_number}: "
        f"{image_path.name}"
    )


    cv2.namedWindow(
        window_name
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    print("\nClick corners:")
    print("1 = TOP-LEFT")
    print("2 = TOP-RIGHT")
    print("3 = BOTTOM-RIGHT")
    print("4 = BOTTOM-LEFT")

    print(
        "\nPress R to restart current image."
    )

    print(
        "Press ESC to skip current image."
    )


    skipped = False


    while True:

        cv2.imshow(
            window_name,
            display_image
        )


        key = (
            cv2.waitKey(20)
            & 0xFF
        )


        if len(
            clicked_points
        ) == 4:

            break


        if key == ord(
            "r"
        ):

            clicked_points.clear()

            display_image = (
                original_display.copy()
            )

            print(
                "\nPoints cleared."
            )


        if key == 27:

            skipped = True

            break


    cv2.destroyAllWindows()


    if skipped:

        print(
            "Image skipped."
        )

        continue


    # ========================================================
    # CONVERT POINTS
    # ========================================================

    pixel_points = np.array(
        clicked_points,
        dtype=np.float64
    ).reshape(
        -1,
        1,
        2
    )


    normalized_points = (
        cv2.undistortPoints(
            pixel_points,
            K,
            dist_coeffs
        )
    )


    normalized_points = (
        normalized_points
        .reshape(
            -1,
            2
        )
    )


    # ========================================================
    # CONVERT TO REAL-WORLD COORDINATES
    # ========================================================

    Z = (
        DISTANCE_M *
        1000.0
    )


    world_points = []


    for point in normalized_points:

        xn = point[0]

        yn = point[1]


        X = (
            xn *
            Z
        )

        Y = (
            yn *
            Z
        )


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


    # ========================================================
    # DISTANCE FUNCTION
    # ========================================================

    def point_distance(
        p1,
        p2
    ):

        return (
            np.linalg.norm(
                p1 -
                p2
            )
        )


    # ========================================================
    # CALCULATE WIDTH AND HEIGHT
    # ========================================================

    top_width = (
        point_distance(
            world_points[0],
            world_points[1]
        )
    )


    bottom_width = (
        point_distance(
            world_points[3],
            world_points[2]
        )
    )


    left_height = (
        point_distance(
            world_points[0],
            world_points[3]
        )
    )


    right_height = (
        point_distance(
            world_points[1],
            world_points[2]
        )
    )


    estimated_width_mm = (
        top_width +
        bottom_width
    ) / 2.0


    estimated_height_mm = (
        left_height +
        right_height
    ) / 2.0


    estimated_width_cm = (
        estimated_width_mm /
        10.0
    )

    estimated_height_cm = (
        estimated_height_mm /
        10.0
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        "\nRESULT:"
    )

    print(
        f"Width = "
        f"{estimated_width_cm:.2f} cm"
    )

    print(
        f"Height = "
        f"{estimated_height_cm:.2f} cm"
    )


    # ========================================================
    # SAVE ANNOTATED IMAGE
    # ========================================================

    result_image = (
        image.copy()
    )


    integer_points = [
        (
            int(point[0]),
            int(point[1])
        )
        for point in clicked_points
    ]


    for i in range(
        4
    ):

        start = (
            integer_points[i]
        )

        end = (
            integer_points[
                (i + 1) % 4
            ]
        )


        cv2.line(
            result_image,
            start,
            end,
            (0, 255, 0),
            4
        )


    cv2.putText(
        result_image,
        f"W: {estimated_width_cm:.2f} cm",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )


    cv2.putText(
        result_image,
        f"H: {estimated_height_cm:.2f} cm",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )


    output_name = (
        image_path.stem
        + "_result.jpg"
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )


    cv2.imwrite(
        output_path,
        result_image
    )


    # ========================================================
    # SAVE RESULT IN MEMORY
    # ========================================================

    results.append(
        {
            "trial": trial_number,
            "image": image_path.name,
            "distance_m": DISTANCE_M,
            "estimated_width_cm":
                round(
                    estimated_width_cm,
                    2
                ),
            "estimated_height_cm":
                round(
                    estimated_height_cm,
                    2
                )
        }
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "trial",
        "image",
        "distance_m",
        "estimated_width_cm",
        "estimated_height_cm"
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )


    writer.writeheader()


    writer.writerows(
        results
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("============================================")
print("          STEP 3 COMPLETE")
print("============================================")


print(
    f"\nProcessed images: "
    f"{len(results)}"
)


print(
    f"\nResults saved to:"
)

print(
    RESULTS_FILE
)


print(
    "\nSummary:"
)


for result in results:

    print(
        f"{result['trial']:02d}. "
        f"{result['image']}  "
        f"W={result['estimated_width_cm']:.2f} cm  "
        f"H={result['estimated_height_cm']:.2f} cm"
    )


print(
    "\n============================================"
)
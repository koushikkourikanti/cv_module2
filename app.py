# ============================================================
# CSc 8830 - COMPUTER VISION
# MODULE 2 ASSIGNMENT
#
# VisionMetric
# Interactive Camera Calibration and Object Measurement System
#
# Features:
#   1. Saved camera calibration results
#   2. Live checkerboard calibration test
#   3. Interactive object measurement
#   4. Direct image corner clicking
#   5. Real-world dimension estimation
#   6. Validation analytics
#   7. Optional ground-truth error statistics
#   8. Perspective projection theory
#
# Author: Koushik Kourikanti
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import io
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionMetric | CSc 8830",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN CONSTANTS
# ============================================================

BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
CYAN = "#22D3EE"
GREEN = "#22C55E"
ORANGE = "#F59E0B"
RED = "#EF4444"

TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"


# ============================================================
# PREMIUM CSS
#
# IMPORTANT:
# Only CSS is passed through HTML.
# Visible application content uses native Streamlit elements.
# This avoids the previous raw <div> rendering problem.
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 8% 5%,
                rgba(37, 99, 235, 0.22),
                transparent 29%
            ),
            radial-gradient(
                circle at 92% 12%,
                rgba(139, 92, 246, 0.17),
                transparent 27%
            ),
            radial-gradient(
                circle at 55% 95%,
                rgba(34, 211, 238, 0.07),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #050814 0%,
                #091120 47%,
                #060B16 100%
            );
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0D1629 0%,
                #080E1B 100%
            );

        border-right:
            1px solid rgba(96, 165, 250, 0.15);
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 850 !important;
        letter-spacing: -0.04em !important;
    }

    h2 {
        font-weight: 800 !important;
    }

    h3 {
        font-weight: 720 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                rgba(21, 34, 59, 0.91),
                rgba(10, 19, 35, 0.86)
            );

        border:
            1px solid rgba(96, 165, 250, 0.15) !important;

        border-radius:
            20px !important;

        box-shadow:
            0px 14px 45px rgba(0, 0, 0, 0.24);
    }

    div[data-testid="metric-container"] {
        background:
            linear-gradient(
                145deg,
                rgba(24, 39, 68, 0.93),
                rgba(11, 21, 39, 0.88)
            );

        border:
            1px solid rgba(96, 165, 250, 0.17);

        border-radius:
            18px;

        padding:
            1rem 1.15rem;

        box-shadow:
            0px 10px 35px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }

    .stButton > button {
        border-radius: 13px;

        min-height: 46px;

        font-weight: 750;

        color: white;

        border:
            1px solid rgba(147, 197, 253, 0.35);

        background:
            linear-gradient(
                90deg,
                #2563EB,
                #4F46E5
            );

        transition:
            all 0.20s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 0 28px rgba(59, 130, 246, 0.35);

        border-color:
            #93C5FD;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 13px;

        background:
            linear-gradient(
                90deg,
                #2563EB,
                #7C3AED
            );

        color: white;

        font-weight: 750;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background:
            rgba(15, 23, 42, 0.75);

        border:
            1px dashed rgba(96, 165, 250, 0.45);

        border-radius:
            18px;
    }

    div[data-testid="stAlert"] {
        border-radius: 15px;
    }

    div[data-testid="stDataFrame"] {
        border:
            1px solid rgba(148, 163, 184, 0.12);

        border-radius:
            16px;

        overflow:
            hidden;
    }

    div[data-testid="stCodeBlock"] {
        border-radius: 15px;
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    hr {
        border-color:
            rgba(96, 165, 250, 0.18);
    }

    div[role="radiogroup"] label {
        padding-top: 5px;
        padding-bottom: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

CALIBRATION_FILE = Path(
    "calibration_output/calibration_results.npz"
)

CALIBRATION_IMAGES_FOLDER = Path(
    "calibration_images"
)

CALIBRATION_CORNERS_FOLDER = Path(
    "calibration_output/detected_corners"
)

UNDISTORTED_FILE = Path(
    "calibration_output/undistorted_sample.jpg"
)

MEASUREMENT_FOLDER = Path(
    "measurement_images"
)

STEP2_OUTPUT_FOLDER = Path(
    "step2_output"
)

STEP3_OUTPUT_FOLDER = Path(
    "step3_output"
)

STEP3_RESULTS_FILE = (
    STEP3_OUTPUT_FOLDER /
    "step3_measurements.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):
    """
    Return all JPG/JPEG/PNG images in a folder.
    """

    if not folder.exists():
        return []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in extensions
        ]
    )


def count_images(folder):
    """
    Count supported image files.
    """

    return len(
        get_image_files(folder)
    )


def scale_camera_matrix(
    original_matrix,
    calibration_size,
    target_size
):
    """
    Scale camera intrinsics when the image resolution
    changes but the same camera geometry/aspect ratio
    is maintained.
    """

    calibration_width = float(
        calibration_size[0]
    )

    calibration_height = float(
        calibration_size[1]
    )

    target_width = float(
        target_size[0]
    )

    target_height = float(
        target_size[1]
    )

    scale_x = (
        target_width /
        calibration_width
    )

    scale_y = (
        target_height /
        calibration_height
    )

    scaled_matrix = (
        original_matrix
        .astype(np.float64)
        .copy()
    )

    scaled_matrix[0, 0] *= scale_x
    scaled_matrix[0, 2] *= scale_x

    scaled_matrix[1, 1] *= scale_y
    scaled_matrix[1, 2] *= scale_y

    return scaled_matrix


def calculate_object_dimensions(
    pixel_points,
    distance_m,
    camera_matrix_input,
    distortion_input,
    calibration_size,
    image_size_input
):
    """
    Estimate width and height of a planar object.

    Assumption:
        All four selected points are approximately
        on the same depth plane Z.

    Projection:
        X = x_normalized * Z
        Y = y_normalized * Z
    """

    scaled_K = scale_camera_matrix(
        camera_matrix_input,
        calibration_size,
        image_size_input
    )

    pixel_points = np.array(
        pixel_points,
        dtype=np.float64
    ).reshape(
        -1,
        1,
        2
    )

    normalized_points = (
        cv2.undistortPoints(
            pixel_points,
            scaled_K,
            distortion_input
        )
        .reshape(
            -1,
            2
        )
    )

    Z_mm = (
        float(distance_m) *
        1000.0
    )

    world_points = []

    for point in normalized_points:

        X_mm = (
            point[0] *
            Z_mm
        )

        Y_mm = (
            point[1] *
            Z_mm
        )

        world_points.append(
            [
                X_mm,
                Y_mm
            ]
        )

    world_points = np.array(
        world_points,
        dtype=np.float64
    )

    def physical_distance(
        first,
        second
    ):
        return float(
            np.linalg.norm(
                first -
                second
            )
        )

    # Point order:
    # 0 = Top-Left
    # 1 = Top-Right
    # 2 = Bottom-Right
    # 3 = Bottom-Left

    top_width_mm = physical_distance(
        world_points[0],
        world_points[1]
    )

    bottom_width_mm = physical_distance(
        world_points[3],
        world_points[2]
    )

    left_height_mm = physical_distance(
        world_points[0],
        world_points[3]
    )

    right_height_mm = physical_distance(
        world_points[1],
        world_points[2]
    )

    width_mm = (
        top_width_mm +
        bottom_width_mm
    ) / 2.0

    height_mm = (
        left_height_mm +
        right_height_mm
    ) / 2.0

    return {
        "width_cm": width_mm / 10.0,
        "height_cm": height_mm / 10.0,
        "top_width_cm": top_width_mm / 10.0,
        "bottom_width_cm": bottom_width_mm / 10.0,
        "left_height_cm": left_height_mm / 10.0,
        "right_height_cm": right_height_mm / 10.0
    }


def run_live_calibration(
    uploaded_images,
    checkerboard=(9, 6)
):
    """
    Run an OpenCV calibration directly from uploaded
    checkerboard images.

    A unit checkerboard square size is used because the
    intrinsic camera matrix does not depend on the global
    physical scale of the pattern.
    """

    cols = checkerboard[0]
    rows = checkerboard[1]

    object_template = np.zeros(
        (
            rows * cols,
            3
        ),
        dtype=np.float32
    )

    object_template[:, :2] = (
        np.mgrid[
            0:cols,
            0:rows
        ]
        .T
        .reshape(
            -1,
            2
        )
    )

    object_points = []
    image_points = []

    image_size_local = None

    accepted_names = []
    rejected_names = []

    termination = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    detection_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for uploaded in uploaded_images:

        data = np.frombuffer(
            uploaded.getvalue(),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        if image is None:

            rejected_names.append(
                uploaded.name
            )

            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        current_size = (
            gray.shape[1],
            gray.shape[0]
        )

        if image_size_local is None:

            image_size_local = (
                current_size
            )

        if (
            current_size !=
            image_size_local
        ):

            rejected_names.append(
                uploaded.name
            )

            continue

        found, corners = (
            cv2.findChessboardCorners(
                gray,
                checkerboard,
                detection_flags
            )
        )

        if not found:

            rejected_names.append(
                uploaded.name
            )

            continue

        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            termination
        )

        object_points.append(
            object_template.copy()
        )

        image_points.append(
            refined
        )

        accepted_names.append(
            uploaded.name
        )

    if len(object_points) < 3:

        return {
            "success": False,
            "message": (
                "At least 3 valid checkerboard "
                "images are required."
            ),
            "accepted": accepted_names,
            "rejected": rejected_names
        }

    rms, K, dist, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size_local,
            None,
            None
        )
    )

    per_view_errors = []

    for index in range(
        len(object_points)
    ):

        projected, _ = (
            cv2.projectPoints(
                object_points[index],
                rvecs[index],
                tvecs[index],
                K,
                dist
            )
        )

        observed = (
            image_points[index]
            .reshape(
                -1,
                2
            )
            .astype(np.float64)
        )

        predicted = (
            projected
            .reshape(
                -1,
                2
            )
            .astype(np.float64)
        )

        differences = (
            observed -
            predicted
        )

        view_rms = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        differences ** 2,
                        axis=1
                    )
                )
            )
        )

        per_view_errors.append(
            view_rms
        )

    return {
        "success": True,
        "rms": float(rms),
        "camera_matrix": K,
        "distortion": dist,
        "image_size": image_size_local,
        "accepted": accepted_names,
        "rejected": rejected_names,
        "per_view_errors": (
            per_view_errors
        )
    }



# ============================================================
# MODULE 3 - IMAGE FILTERING HELPERS
# ============================================================

MODULE3_SAMPLE_IMAGE = Path(
    "module3/sample_images/test_image.jpg"
)


def module3_make_box_kernel(kernel_size):
    """
    Create a normalized square box-blur kernel.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float64
    )

    return kernel / np.sum(kernel)


def module3_spatial_filter(
    image_data,
    kernel
):
    """
    Apply the blur directly in the spatial domain.

    For the symmetric box kernel used in this assignment,
    cv2.filter2D produces the same result as convolution.
    """

    return cv2.filter2D(
        image_data,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT
    )


def module3_fourier_filter(
    image_data,
    kernel
):
    """
    Apply equivalent linear convolution by multiplying
    the Fourier transforms of the image and filter.
    """

    image_height, image_width = image_data.shape
    kernel_height, kernel_width = kernel.shape

    fft_height = (
        image_height
        + kernel_height
        - 1
    )

    fft_width = (
        image_width
        + kernel_width
        - 1
    )

    image_fft = np.fft.fft2(
        image_data,
        s=(fft_height, fft_width)
    )

    kernel_fft = np.fft.fft2(
        kernel,
        s=(fft_height, fft_width)
    )

    frequency_product = (
        image_fft
        * kernel_fft
    )

    full_result = np.real(
        np.fft.ifft2(
            frequency_product
        )
    )

    offset_y = kernel_height // 2
    offset_x = kernel_width // 2

    return full_result[
        offset_y:
        offset_y + image_height,
        offset_x:
        offset_x + image_width
    ]


def module3_fourier_spectrum(
    image_data
):
    """
    Create a displayable centered log-magnitude
    Fourier spectrum.
    """

    fft_result = np.fft.fft2(
        image_data
    )

    shifted = np.fft.fftshift(
        fft_result
    )

    magnitude = np.log1p(
        np.abs(shifted)
    )

    normalized = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized.astype(
        np.uint8
    )


def module3_compare_methods(
    image_data,
    kernel_size
):
    """
    Run spatial and Fourier filtering with the same
    box kernel and return numerical validation.
    """

    kernel = module3_make_box_kernel(
        kernel_size
    )

    spatial = module3_spatial_filter(
        image_data,
        kernel
    )

    fourier = module3_fourier_filter(
        image_data,
        kernel
    )

    difference = np.abs(
        spatial - fourier
    )

    return {
        "kernel_size": kernel_size,
        "spatial": spatial,
        "fourier": fourier,
        "difference": difference,
        "mae": float(
            np.mean(difference)
        ),
        "mse": float(
            np.mean(
                difference ** 2
            )
        ),
        "max_difference": float(
            np.max(difference)
        )
    }


# ============================================================
# LOAD SAVED CAMERA CALIBRATION
# ============================================================

calibration_loaded = False

camera_matrix = None
dist_coeffs = None
calibration_image_size = None
calibration_data = None


if CALIBRATION_FILE.exists():

    calibration_data = np.load(
        str(
            CALIBRATION_FILE
        ),
        allow_pickle=True
    )

    camera_matrix = calibration_data[
        "camera_matrix"
    ]

    dist_coeffs = calibration_data[
        "distortion_coefficients"
    ]

    calibration_image_size = (
        calibration_data[
            "image_size"
        ]
    )

    calibration_loaded = True


# ============================================================
# LOAD VALIDATION RESULTS
# ============================================================

validation_df = None


if STEP3_RESULTS_FILE.exists():

    validation_df = pd.read_csv(
        STEP3_RESULTS_FILE
    )


# ============================================================
# CALIBRATION STATISTICS
# ============================================================

rms_error = None
mean_reprojection_error = None
median_reprojection_error = None
valid_calibration_frames = None


if calibration_loaded:

    if (
        "rms" in
        calibration_data.files
    ):

        rms_error = float(
            calibration_data[
                "rms"
            ]
        )

    if (
        "mean_reprojection_error"
        in calibration_data.files
    ):

        mean_reprojection_error = (
            float(
                calibration_data[
                    "mean_reprojection_error"
                ]
            )
        )

    if (
        "per_view_errors"
        in calibration_data.files
    ):

        saved_per_view = np.array(
            calibration_data[
                "per_view_errors"
            ],
            dtype=float
        )

        valid_calibration_frames = len(
            saved_per_view
        )

        median_reprojection_error = (
            float(
                np.median(
                    saved_per_view
                )
            )
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "◉ VisionMetric"
)

st.sidebar.caption(
    "Computer Vision Measurement Lab"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🎯 Camera Calibration",
        "📐 Live Object Measurement",
        "📊 Validation Analytics",
        "🧠 Projection Theory",
        "🌀 Module 3 - Image Filtering"
    ]
)


st.sidebar.divider()


if calibration_loaded:

    st.sidebar.success(
        "● Calibration Ready"
    )

else:

    st.sidebar.error(
        "● Calibration Missing"
    )


if validation_df is not None:

    st.sidebar.success(
        "● Validation Results Ready"
    )


# ============================================================
# GITHUB REPOSITORY LINKS
# ============================================================

st.sidebar.subheader(
    "GitHub Repositories"
)

st.sidebar.link_button(
    "🔗 Module 2 GitHub",
    "https://github.com/koushikkourikanti/cv_module2",
    use_container_width=True
)

st.sidebar.link_button(
    "🔗 Module 3 GitHub",
    "https://github.com/koushikkourikanti/cv_module3",
    use_container_width=True
)


st.sidebar.caption(
    "CSc 8830 · Modules 2 & 3"
)


# ============================================================
# GLOBAL HEADER
# ============================================================

st.title(
    "📷 VisionMetric"
)

st.caption(
    "Camera Calibration  •  "
    "Perspective Projection  •  "
    "Interactive Real-World Dimension Estimation"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    with st.container(
        border=True
    ):

        hero_left, hero_right = (
            st.columns(
                [
                    1.8,
                    1
                ]
            )
        )

        with hero_left:

            st.caption(
                "CSC 8830 • COMPUTER VISION"
            )

            st.header(
                "Smartphone Camera "
                "Measurement System"
            )

            st.write(
                """
                VisionMetric is an interactive
                computer-vision application that
                calibrates a smartphone camera and
                estimates real-world object dimensions
                using perspective projection.
                """
            )

            st.write(
                """
                The system supports live testing:
                upload an image, select four object
                corners and calculate the estimated
                physical width and height.
                """
            )

            st.success(
                "✓ Interactive Module 2 system ready"
            )

        with hero_right:

            completed_modules = 0

            if calibration_loaded:
                completed_modules += 1

            completed_modules += 1

            if validation_df is not None:
                completed_modules += 1

            completed_modules += 1

            completion_percentage = (
                completed_modules /
                4.0 *
                100.0
            )

            donut = go.Figure(
                go.Pie(
                    values=[
                        completion_percentage,
                        (
                            100 -
                            completion_percentage
                        )
                    ],
                    labels=[
                        "Complete",
                        "Remaining"
                    ],
                    hole=0.73,
                    marker=dict(
                        colors=[
                            BLUE,
                            "rgba(148,163,184,0.12)"
                        ]
                    ),
                    textinfo="none",
                    hoverinfo="label+value"
                )
            )

            donut.add_annotation(
                text=(
                    f"<b>"
                    f"{completion_percentage:.0f}%"
                    f"</b>"
                    "<br>"
                    "<span style='font-size:12px'>"
                    "System Ready"
                    "</span>"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    size=24,
                    color=TEXT_PRIMARY
                )
            )

            donut.update_layout(
                height=275,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                showlegend=False,
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                )
            )

            st.plotly_chart(
                donut,
                use_container_width=True,
                config={
                    "displayModeBar":
                    False
                }
            )

    st.write("")

    calibration_count = (
        valid_calibration_frames
        if valid_calibration_frames
        is not None
        else count_images(
            CALIBRATION_IMAGES_FOLDER
        )
    )

    validation_count = (
        len(validation_df)
        if validation_df is not None
        else 0
    )

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Camera",
        (
            "Calibrated"
            if calibration_loaded
            else "Missing"
        )
    )

    d2.metric(
        "Valid Calibration Frames",
        calibration_count
    )

    d3.metric(
        "Validation Images",
        validation_count
    )

    d4.metric(
        "Dimension Outputs",
        validation_count * 2
    )

    st.divider()

    st.subheader(
        "System Workflow"
    )

    w1, w2, w3, w4 = st.columns(
        4
    )

    with w1:

        with st.container(
            border=True
        ):

            st.caption(
                "STEP 01"
            )

            st.header(
                "📱"
            )

            st.subheader(
                "Capture"
            )

            st.write(
                """
                Capture checkerboard and
                object images using a
                smartphone camera.
                """
            )

    with w2:

        with st.container(
            border=True
        ):

            st.caption(
                "STEP 02"
            )

            st.header(
                "🎯"
            )

            st.subheader(
                "Calibrate"
            )

            st.write(
                """
                Estimate the camera
                intrinsic matrix and
                lens distortion.
                """
            )

    with w3:

        with st.container(
            border=True
        ):

            st.caption(
                "STEP 03"
            )

            st.header(
                "📐"
            )

            st.subheader(
                "Measure"
            )

            st.write(
                """
                Click four corners and
                estimate real-world
                width and height.
                """
            )

    with w4:

        with st.container(
            border=True
        ):

            st.caption(
                "STEP 04"
            )

            st.header(
                "📊"
            )

            st.subheader(
                "Validate"
            )

            st.write(
                """
                Analyze results from
                multiple test objects
                and error statistics.
                """
            )

    st.divider()

    st.subheader(
        "Professor Test Mode"
    )

    with st.container(
        border=True
    ):

        st.write(
            """
            The live measurement page allows the
            application to be tested directly.
            """
        )

        st.markdown(
            """
            **Test workflow**

            Upload/select image → enter distance →
            click Top-Left → Top-Right →
            Bottom-Right → Bottom-Left →
            calculate dimensions.
            """
        )

        st.info(
            "For the most reliable measurements, "
            "use images captured using the same "
            "smartphone/lens configuration used "
            "during calibration."
        )


# ============================================================
# CAMERA CALIBRATION PAGE
# ============================================================

elif page == "🎯 Camera Calibration":

    st.header(
        "🎯 Camera Calibration"
    )

    st.caption(
        "Saved calibration results and "
        "live OpenCV calibration test"
    )

    tab_saved, tab_live = st.tabs(
        [
            "📊 Saved Calibration",
            "🧪 Live Calibration Test"
        ]
    )

    # ========================================================
    # SAVED CALIBRATION
    # ========================================================

    with tab_saved:

        if not calibration_loaded:

            st.error(
                "Saved calibration results "
                "were not found."
            )

        else:

            st.success(
                "✓ Camera calibration "
                "loaded successfully."
            )

            fx = float(
                camera_matrix[
                    0,
                    0
                ]
            )

            fy = float(
                camera_matrix[
                    1,
                    1
                ]
            )

            cx = float(
                camera_matrix[
                    0,
                    2
                ]
            )

            cy = float(
                camera_matrix[
                    1,
                    2
                ]
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "Focal Length fx",
                f"{fx:.2f} px"
            )

            c2.metric(
                "Focal Length fy",
                f"{fy:.2f} px"
            )

            c3.metric(
                "Principal Point cx",
                f"{cx:.2f} px"
            )

            c4.metric(
                "Principal Point cy",
                f"{cy:.2f} px"
            )

            st.divider()

            left, right = st.columns(
                [
                    1,
                    1.35
                ]
            )

            with left:

                st.subheader(
                    "Calibration Quality"
                )

                if rms_error is not None:

                    gauge = go.Figure(
                        go.Indicator(
                            mode=(
                                "gauge+number"
                            ),
                            value=rms_error,
                            number=dict(
                                suffix=" px",
                                font=dict(
                                    color=(
                                        TEXT_PRIMARY
                                    )
                                )
                            ),
                            title=dict(
                                text=(
                                    "RMS "
                                    "Reprojection "
                                    "Error"
                                ),
                                font=dict(
                                    color=(
                                        TEXT_SECONDARY
                                    )
                                )
                            ),
                            gauge=dict(
                                axis=dict(
                                    range=[
                                        0,
                                        3
                                    ]
                                ),
                                bar=dict(
                                    color=GREEN
                                ),
                                bgcolor=(
                                    "rgba(0,0,0,0)"
                                ),
                                steps=[
                                    dict(
                                        range=[
                                            0,
                                            1
                                        ],
                                        color=(
                                            "rgba("
                                            "34,197,"
                                            "94,0.20)"
                                        )
                                    ),
                                    dict(
                                        range=[
                                            1,
                                            2
                                        ],
                                        color=(
                                            "rgba("
                                            "245,158,"
                                            "11,0.20)"
                                        )
                                    ),
                                    dict(
                                        range=[
                                            2,
                                            3
                                        ],
                                        color=(
                                            "rgba("
                                            "239,68,"
                                            "68,0.20)"
                                        )
                                    )
                                ]
                            )
                        )
                    )

                    gauge.update_layout(
                        height=310,
                        margin=dict(
                            l=20,
                            r=20,
                            t=55,
                            b=15
                        ),
                        paper_bgcolor=(
                            "rgba(0,0,0,0)"
                        ),
                        font=dict(
                            color=TEXT_PRIMARY
                        )
                    )

                    st.plotly_chart(
                        gauge,
                        use_container_width=True,
                        config={
                            "displayModeBar":
                            False
                        }
                    )

            with right:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Intrinsic Matrix K"
                    )

                    st.code(
                        np.array2string(
                            camera_matrix,
                            precision=5
                        ),
                        language="text"
                    )

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Distortion Coefficients"
                    )

                    st.code(
                        np.array2string(
                            dist_coeffs,
                            precision=7
                        ),
                        language="text"
                    )

            if (
                rms_error is not None
                or
                mean_reprojection_error
                is not None
            ):

                e1, e2, e3, e4 = (
                    st.columns(4)
                )

                e1.metric(
                    "RMS Error",
                    (
                        f"{rms_error:.4f} px"
                        if rms_error
                        is not None
                        else "N/A"
                    )
                )

                e2.metric(
                    "Mean Error",
                    (
                        f"{mean_reprojection_error:.4f} px"
                        if mean_reprojection_error
                        is not None
                        else "N/A"
                    )
                )

                e3.metric(
                    "Median Error",
                    (
                        f"{median_reprojection_error:.4f} px"
                        if median_reprojection_error
                        is not None
                        else "N/A"
                    )
                )

                e4.metric(
                    "Valid Frames",
                    (
                        valid_calibration_frames
                        if valid_calibration_frames
                        is not None
                        else "N/A"
                    )
                )

            if (
                calibration_image_size
                is not None
            ):

                st.info(
                    (
                        "Calibration resolution: "
                        f"{int(calibration_image_size[0])}"
                        " × "
                        f"{int(calibration_image_size[1])}"
                        " pixels | "
                        "Checkerboard: 9 × 6 "
                        "internal corners"
                    )
                )

            if UNDISTORTED_FILE.exists():

                st.divider()

                st.subheader(
                    "Lens Distortion Correction"
                )

                st.image(
                    str(
                        UNDISTORTED_FILE
                    ),
                    caption=(
                        "Example image after "
                        "distortion correction"
                    ),
                    use_container_width=True
                )

            corner_images = (
                get_image_files(
                    CALIBRATION_CORNERS_FOLDER
                )
            )

            if corner_images:

                st.divider()

                st.subheader(
                    "Checkerboard "
                    "Corner Detection"
                )

                selected_corner = (
                    st.selectbox(
                        "Select detected frame",
                        corner_images,
                        format_func=(
                            lambda x:
                            x.name
                        )
                    )
                )

                st.image(
                    str(
                        selected_corner
                    ),
                    caption=(
                        "Detected 9 × 6 "
                        "checkerboard corners"
                    ),
                    use_container_width=True
                )

    # ========================================================
    # LIVE CALIBRATION TEST
    # ========================================================

    with tab_live:

        st.subheader(
            "Live Checkerboard "
            "Calibration Test"
        )

        st.write(
            """
            Upload several checkerboard images.
            The application will detect the
            **9 × 6 internal corners** and run
            OpenCV camera calibration directly
            in the web application.
            """
        )

        live_files = st.file_uploader(
            "Upload checkerboard images",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            accept_multiple_files=True,
            key="live_calibration_files"
        )

        st.caption(
            "Use images from the same camera "
            "and the same image resolution."
        )

        if live_files:

            st.metric(
                "Images Uploaded",
                len(live_files)
            )

            if st.button(
                "🎯 Run Live Calibration",
                use_container_width=True
            ):

                with st.spinner(
                    "Detecting checkerboard "
                    "corners and calibrating..."
                ):

                    live_result = (
                        run_live_calibration(
                            live_files,
                            checkerboard=(
                                9,
                                6
                            )
                        )
                    )

                    st.session_state[
                        "live_calibration_result"
                    ] = live_result

        live_result = (
            st.session_state.get(
                "live_calibration_result"
            )
        )

        if live_result:

            if not live_result[
                "success"
            ]:

                st.error(
                    live_result[
                        "message"
                    ]
                )

            else:

                st.success(
                    "✓ Live calibration "
                    "completed successfully."
                )

                lc1, lc2, lc3 = (
                    st.columns(3)
                )

                lc1.metric(
                    "Accepted Frames",
                    len(
                        live_result[
                            "accepted"
                        ]
                    )
                )

                lc2.metric(
                    "Rejected Frames",
                    len(
                        live_result[
                            "rejected"
                        ]
                    )
                )

                lc3.metric(
                    "RMS Error",
                    (
                        f"{live_result['rms']:.4f}"
                        " px"
                    )
                )

                st.subheader(
                    "Live Camera Matrix"
                )

                st.code(
                    np.array2string(
                        live_result[
                            "camera_matrix"
                        ],
                        precision=5
                    ),
                    language="text"
                )

                st.subheader(
                    "Live Distortion "
                    "Coefficients"
                )

                st.code(
                    np.array2string(
                        live_result[
                            "distortion"
                        ],
                        precision=7
                    ),
                    language="text"
                )

                error_df = pd.DataFrame(
                    {
                        "Frame": (
                            live_result[
                                "accepted"
                            ]
                        ),
                        "Reprojection Error (px)": (
                            live_result[
                                "per_view_errors"
                            ]
                        )
                    }
                )

                error_chart = px.bar(
                    error_df,
                    x="Frame",
                    y=(
                        "Reprojection "
                        "Error (px)"
                    ),
                    title=(
                        "Per-Frame "
                        "Reprojection Error"
                    ),
                    color=(
                        "Reprojection "
                        "Error (px)"
                    ),
                    color_continuous_scale=(
                        "Blues"
                    )
                )

                error_chart.update_layout(
                    paper_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    plot_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    font=dict(
                        color=TEXT_PRIMARY
                    )
                )

                st.plotly_chart(
                    error_chart,
                    use_container_width=True
                )

                if live_result[
                    "rejected"
                ]:

                    with st.expander(
                        "Rejected Images"
                    ):

                        for name in (
                            live_result[
                                "rejected"
                            ]
                        ):

                            st.write(
                                f"• {name}"
                            )


# ============================================================
# LIVE OBJECT MEASUREMENT
# ============================================================

elif page == "📐 Live Object Measurement":

    st.header(
        "📐 Interactive Object Measurement"
    )

    st.caption(
        "Select an image, click four corners "
        "and calculate real-world dimensions"
    )

    if not calibration_loaded:

        st.error(
            "Saved camera calibration "
            "is required before measurement."
        )

        st.stop()

    with st.container(
        border=True
    ):

        st.subheader(
            "🧪 Professor Test Mode"
        )

        st.write(
            """
            The application performs a real
            measurement from the selected image.
            """
        )

        st.info(
            "For best accuracy, use an image "
            "captured using the same calibrated "
            "smartphone/lens configuration."
        )

    input_mode = st.radio(
        "Image Source",
        [
            "Use Project Test Image",
            "Upload New Image"
        ],
        horizontal=True
    )

    image_bytes = None
    image_name = None

    # ========================================================
    # PROJECT IMAGE
    # ========================================================

    if (
        input_mode ==
        "Use Project Test Image"
    ):

        project_images = (
            get_image_files(
                MEASUREMENT_FOLDER
            )
        )

        if not project_images:

            st.warning(
                "No images were found "
                "in measurement_images."
            )

        else:

            selected_project_image = (
                st.selectbox(
                    "Choose a test image",
                    project_images,
                    format_func=(
                        lambda path:
                        path.name
                    )
                )
            )

            image_name = (
                selected_project_image.name
            )

            image_bytes = (
                selected_project_image
                .read_bytes()
            )

    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    else:

        uploaded_object = (
            st.file_uploader(
                "Upload object image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="object_image_upload"
            )
        )

        if (
            uploaded_object
            is not None
        ):

            image_bytes = (
                uploaded_object
                .getvalue()
            )

            image_name = (
                uploaded_object.name
            )

    # ========================================================
    # NO IMAGE YET
    # ========================================================

    if image_bytes is None:

        with st.container(
            border=True
        ):

            st.subheader(
                "How to Test"
            )

            st.markdown(
                """
                **1.** Select or upload an image.

                **2.** Enter camera-to-object distance.

                **3.** Click Top-Left.

                **4.** Click Top-Right.

                **5.** Click Bottom-Right.

                **6.** Click Bottom-Left.

                **7.** Press Calculate Dimensions.
                """
            )

    # ========================================================
    # PROCESS SELECTED IMAGE
    # ========================================================

    else:

        try:

            original_pil = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            original_pil = (
                ImageOps.exif_transpose(
                    original_pil
                )
                .convert(
                    "RGB"
                )
            )

        except Exception as error:

            st.error(
                (
                    "Unable to read image: "
                    f"{error}"
                )
            )

            st.stop()

        original_width, original_height = (
            original_pil.size
        )

        image_hash = hashlib.md5(
            image_bytes
        ).hexdigest()[:12]

        # Reset points when image changes.

        if (
            st.session_state.get(
                "active_measurement_image"
            )
            != image_hash
        ):

            st.session_state[
                "active_measurement_image"
            ] = image_hash

            st.session_state[
                "measurement_points"
            ] = []

            st.session_state[
                "measurement_result"
            ] = None

        if (
            "measurement_points"
            not in st.session_state
        ):

            st.session_state[
                "measurement_points"
            ] = []

        if (
            "measurement_result"
            not in st.session_state
        ):

            st.session_state[
                "measurement_result"
            ] = None

        # ====================================================
        # DISTANCE INPUT
        # ====================================================

        distance_m = st.number_input(
            "Camera-to-object distance (meters)",
            min_value=0.10,
            max_value=30.0,
            value=2.50,
            step=0.10,
            format="%.2f"
        )

        point_count = len(
            st.session_state[
                "measurement_points"
            ]
        )

        m1, m2, m3, m4 = st.columns(
            4
        )

        m1.metric(
            "Image",
            image_name
        )

        m2.metric(
            "Resolution",
            (
                f"{original_width}"
                " × "
                f"{original_height}"
            )
        )

        m3.metric(
            "Distance",
            f"{distance_m:.2f} m"
        )

        m4.metric(
            "Corners Selected",
            f"{point_count}/4"
        )

        # ====================================================
        # ASPECT RATIO CHECK
        # ====================================================

        calibration_width = int(
            calibration_image_size[
                0
            ]
        )

        calibration_height = int(
            calibration_image_size[
                1
            ]
        )

        calibration_ratio = (
            calibration_width /
            calibration_height
        )

        current_ratio = (
            original_width /
            original_height
        )

        relative_ratio_difference = abs(
            current_ratio -
            calibration_ratio
        ) / calibration_ratio

        if (
            relative_ratio_difference
            <= 0.03
        ):

            st.success(
                "✓ Image aspect ratio is "
                "compatible with the saved "
                "camera calibration."
            )

        else:

            st.warning(
                "This image has a different "
                "aspect ratio from the calibration "
                "images. A result can still be "
                "computed, but accuracy may be lower."
            )

        st.divider()

        # ====================================================
        # CLICK ORDER
        # ====================================================

        corner_names = [
            "Top-Left",
            "Top-Right",
            "Bottom-Right",
            "Bottom-Left"
        ]

        st.subheader(
            "Select Object Corners"
        )

        if point_count < 4:

            st.info(
                (
                    f"Click #{point_count + 1}: "
                    f"**{corner_names[point_count]}**"
                )
            )

        else:

            st.success(
                "✓ All four corners selected. "
                "Press Calculate Dimensions."
            )

        # ====================================================
        # DISPLAY IMAGE SIZE
        # ====================================================

        MAX_DISPLAY_WIDTH = 760

        display_scale = min(
            1.0,
            (
                MAX_DISPLAY_WIDTH /
                original_width
            )
        )

        display_width = max(
            1,
            int(
                original_width *
                display_scale
            )
        )

        display_height = max(
            1,
            int(
                original_height *
                display_scale
            )
        )

        display_image = (
            original_pil.resize(
                (
                    display_width,
                    display_height
                )
            )
        )

        # ====================================================
        # DRAW SELECTED POINTS
        # ====================================================

        overlay = display_image.copy()

        overlay_draw = ImageDraw.Draw(
            overlay
        )

        display_points = []

        for index, point in enumerate(
            st.session_state[
                "measurement_points"
            ]
        ):

            original_x = point[0]
            original_y = point[1]

            display_x = int(
                original_x *
                display_scale
            )

            display_y = int(
                original_y *
                display_scale
            )

            display_points.append(
                (
                    display_x,
                    display_y
                )
            )

            radius = 8

            overlay_draw.ellipse(
                (
                    display_x - radius,
                    display_y - radius,
                    display_x + radius,
                    display_y + radius
                ),
                fill="red",
                outline="white",
                width=2
            )

            overlay_draw.text(
                (
                    display_x + 10,
                    display_y - 18
                ),
                str(
                    index + 1
                ),
                fill="white"
            )

        if (
            len(display_points)
            >= 2
        ):

            for index in range(
                len(display_points) - 1
            ):

                overlay_draw.line(
                    (
                        display_points[
                            index
                        ],
                        display_points[
                            index + 1
                        ]
                    ),
                    fill="lime",
                    width=4
                )

        if (
            len(display_points)
            == 4
        ):

            overlay_draw.line(
                (
                    display_points[3],
                    display_points[0]
                ),
                fill="lime",
                width=4
            )

        # ====================================================
        # INTERACTIVE CLICK COMPONENT
        # ====================================================

        click_column, info_column = (
            st.columns(
                [
                    1.5,
                    1
                ]
            )
        )

        with click_column:

            if point_count < 4:

                click_value = (
                    streamlit_image_coordinates(
                        overlay,
                        key=(
                            "object_click_"
                            f"{image_hash}_"
                            f"{point_count}"
                        )
                    )
                )

                if (
                    click_value
                    is not None
                ):

                    x_display = float(
                        click_value[
                            "x"
                        ]
                    )

                    y_display = float(
                        click_value[
                            "y"
                        ]
                    )

                    x_original = (
                        x_display /
                        display_scale
                    )

                    y_original = (
                        y_display /
                        display_scale
                    )

                    x_original = float(
                        np.clip(
                            x_original,
                            0,
                            original_width - 1
                        )
                    )

                    y_original = float(
                        np.clip(
                            y_original,
                            0,
                            original_height - 1
                        )
                    )

                    st.session_state[
                        "measurement_points"
                    ].append(
                        (
                            x_original,
                            y_original
                        )
                    )

                    st.session_state[
                        "measurement_result"
                    ] = None

                    st.rerun()

            else:

                st.image(
                    overlay,
                    caption=(
                        "Selected object boundary"
                    ),
                    use_container_width=False
                )

        with info_column:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Click Order"
                )

                for index, name in enumerate(
                    corner_names
                ):

                    if index < point_count:

                        st.write(
                            f"✅ {index + 1}. "
                            f"{name}"
                        )

                    elif index == point_count:

                        st.write(
                            f"👉 {index + 1}. "
                            f"{name}"
                        )

                    else:

                        st.write(
                            f"○ {index + 1}. "
                            f"{name}"
                        )

        # ====================================================
        # COORDINATE TABLE
        # ====================================================

        if (
            st.session_state[
                "measurement_points"
            ]
        ):

            coordinate_rows = []

            for index, point in enumerate(
                st.session_state[
                    "measurement_points"
                ]
            ):

                coordinate_rows.append(
                    {
                        "Point":
                            index + 1,

                        "Corner":
                            corner_names[
                                index
                            ],

                        "X (px)":
                            round(
                                point[0],
                                1
                            ),

                        "Y (px)":
                            round(
                                point[1],
                                1
                            )
                    }
                )

            st.subheader(
                "Selected Coordinates"
            )

            st.dataframe(
                pd.DataFrame(
                    coordinate_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # CONTROL BUTTONS
        # ====================================================

        reset_col, calc_col = (
            st.columns(
                [
                    1,
                    2
                ]
            )
        )

        with reset_col:

            if st.button(
                "↻ Reset Corners",
                use_container_width=True
            ):

                st.session_state[
                    "measurement_points"
                ] = []

                st.session_state[
                    "measurement_result"
                ] = None

                st.rerun()

        with calc_col:

            calculate_clicked = (
                st.button(
                    "🚀 Calculate Dimensions",
                    disabled=(
                        len(
                            st.session_state[
                                "measurement_points"
                            ]
                        )
                        != 4
                    ),
                    use_container_width=True
                )
            )

        # ====================================================
        # DIMENSION CALCULATION
        # ====================================================

        if calculate_clicked:

            try:

                result = (
                    calculate_object_dimensions(
                        pixel_points=(
                            st.session_state[
                                "measurement_points"
                            ]
                        ),
                        distance_m=(
                            distance_m
                        ),
                        camera_matrix_input=(
                            camera_matrix
                        ),
                        distortion_input=(
                            dist_coeffs
                        ),
                        calibration_size=(
                            calibration_image_size
                        ),
                        image_size_input=(
                            (
                                original_width,
                                original_height
                            )
                        )
                    )
                )

                result[
                    "distance_m"
                ] = distance_m

                result[
                    "image_name"
                ] = image_name

                st.session_state[
                    "measurement_result"
                ] = result

            except Exception as error:

                st.error(
                    (
                        "Measurement calculation "
                        f"failed: {error}"
                    )
                )

        # ====================================================
        # SHOW RESULT
        # ====================================================

        measurement_result = (
            st.session_state.get(
                "measurement_result"
            )
        )

        if (
            measurement_result
            is not None
        ):

            st.divider()

            st.header(
                "✅ Measurement Result"
            )

            result1, result2, result3 = (
                st.columns(3)
            )

            result1.metric(
                "Estimated Width",
                (
                    f"{measurement_result['width_cm']:.2f}"
                    " cm"
                )
            )

            result2.metric(
                "Estimated Height",
                (
                    f"{measurement_result['height_cm']:.2f}"
                    " cm"
                )
            )

            result3.metric(
                "Camera Distance",
                (
                    f"{measurement_result['distance_m']:.2f}"
                    " m"
                )
            )

            st.success(
                "✓ Real-world dimension "
                "estimation completed."
            )

            # ================================================
            # FINAL ANNOTATED IMAGE
            # ================================================

            final_image = (
                original_pil.copy()
            )

            final_draw = (
                ImageDraw.Draw(
                    final_image
                )
            )

            final_points = [
                (
                    int(point[0]),
                    int(point[1])
                )
                for point in (
                    st.session_state[
                        "measurement_points"
                    ]
                )
            ]

            line_width = max(
                4,
                original_width // 180
            )

            final_draw.line(
                [
                    final_points[0],
                    final_points[1],
                    final_points[2],
                    final_points[3],
                    final_points[0]
                ],
                fill="lime",
                width=line_width
            )

            point_radius = max(
                7,
                original_width // 110
            )

            for point in final_points:

                x_point, y_point = point

                final_draw.ellipse(
                    (
                        x_point -
                        point_radius,

                        y_point -
                        point_radius,

                        x_point +
                        point_radius,

                        y_point +
                        point_radius
                    ),
                    fill="red",
                    outline="white",
                    width=3
                )

            st.image(
                final_image,
                caption=(
                    "Estimated Width: "
                    f"{measurement_result['width_cm']:.2f}"
                    " cm  |  "
                    "Estimated Height: "
                    f"{measurement_result['height_cm']:.2f}"
                    " cm"
                ),
                use_container_width=True
            )

            detail1, detail2 = (
                st.columns(2)
            )

            detail1.metric(
                "Top Width",
                (
                    f"{measurement_result['top_width_cm']:.2f}"
                    " cm"
                )
            )

            detail1.metric(
                "Bottom Width",
                (
                    f"{measurement_result['bottom_width_cm']:.2f}"
                    " cm"
                )
            )

            detail2.metric(
                "Left Height",
                (
                    f"{measurement_result['left_height_cm']:.2f}"
                    " cm"
                )
            )

            detail2.metric(
                "Right Height",
                (
                    f"{measurement_result['right_height_cm']:.2f}"
                    " cm"
                )
            )

            result_table = pd.DataFrame(
                [
                    {
                        "Image":
                            image_name,

                        "Distance (m)":
                            round(
                                distance_m,
                                2
                            ),

                        "Estimated Width (cm)":
                            round(
                                measurement_result[
                                    "width_cm"
                                ],
                                2
                            ),

                        "Estimated Height (cm)":
                            round(
                                measurement_result[
                                    "height_cm"
                                ],
                                2
                            )
                    }
                ]
            )

            st.dataframe(
                result_table,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        with st.expander(
            "📘 How does this calculation work?"
        ):

            st.write(
                """
                First, lens distortion is
                corrected using the camera
                calibration parameters.
                """
            )

            equation1, equation2 = (
                st.columns(2)
            )

            with equation1:

                st.latex(
                    r"""
                    X =
                    \frac{(u-c_x)Z}{f_x}
                    """
                )

            with equation2:

                st.latex(
                    r"""
                    Y =
                    \frac{(v-c_y)Z}{f_y}
                    """
                )

            st.write(
                """
                After each image point is
                converted into a real-world
                coordinate on the known depth
                plane, Euclidean distance is
                used between the four selected
                corners to obtain width and height.
                """
            )


# ============================================================
# VALIDATION ANALYTICS
# ============================================================

elif page == "📊 Validation Analytics":

    st.header(
        "📊 Validation Analytics"
    )

    st.caption(
        "Multi-object measurement results "
        "and optional ground-truth error analysis"
    )

    if validation_df is None:

        st.error(
            "Step 3 results CSV "
            "was not found."
        )

        st.stop()

    df = validation_df.copy()

    total_dimension_outputs = (
        len(df) *
        2
    )

    st.success(
        (
            f"✓ {len(df)} images processed • "
            f"{total_dimension_outputs} "
            "dimension outputs"
        )
    )

    v1, v2, v3, v4 = st.columns(
        4
    )

    v1.metric(
        "Objects Tested",
        len(df)
    )

    v2.metric(
        "Dimension Outputs",
        total_dimension_outputs
    )

    v3.metric(
        "Average Width",
        (
            f"{df['estimated_width_cm'].mean():.2f}"
            " cm"
        )
    )

    v4.metric(
        "Average Height",
        (
            f"{df['estimated_height_cm'].mean():.2f}"
            " cm"
        )
    )

    st.divider()

    tab_results, tab_errors = st.tabs(
        [
            "📊 Measurement Results",
            "🎯 Error Statistics"
        ]
    )

    # ========================================================
    # RESULT VISUALIZATION
    # ========================================================

    with tab_results:

        st.subheader(
            "Measurement Dataset"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Estimated Object Dimensions"
        )

        dimension_chart = px.bar(
            df,
            x="image",
            y=[
                "estimated_width_cm",
                "estimated_height_cm"
            ],
            barmode="group",
            labels={
                "image":
                    "Object",

                "value":
                    "Dimension (cm)",

                "variable":
                    "Measurement"
            },
            color_discrete_sequence=[
                BLUE,
                PURPLE
            ]
        )

        dimension_chart.update_layout(
            height=510,
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            font=dict(
                color=TEXT_PRIMARY
            ),
            legend_title_text="",
            xaxis_tickangle=-35
        )

        dimension_chart.update_xaxes(
            gridcolor=(
                "rgba(148,163,184,0.08)"
            )
        )

        dimension_chart.update_yaxes(
            gridcolor=(
                "rgba(148,163,184,0.12)"
            )
        )

        st.plotly_chart(
            dimension_chart,
            use_container_width=True
        )

        st.subheader(
            "Measurement Space"
        )

        scatter = px.scatter(
            df,
            x="estimated_width_cm",
            y="estimated_height_cm",
            hover_name="image",
            size="estimated_width_cm",
            size_max=42,
            color="estimated_height_cm",
            color_continuous_scale=(
                "Blues"
            ),
            labels={
                "estimated_width_cm":
                    "Width (cm)",

                "estimated_height_cm":
                    "Height (cm)"
            }
        )

        scatter.update_layout(
            height=470,
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            font=dict(
                color=TEXT_PRIMARY
            )
        )

        st.plotly_chart(
            scatter,
            use_container_width=True
        )

        result_images = sorted(
            STEP3_OUTPUT_FOLDER.glob(
                "*_result.jpg"
            )
        )

        if result_images:

            st.divider()

            st.subheader(
                "Annotated Result Viewer"
            )

            selected_result_image = (
                st.selectbox(
                    "Select result image",
                    result_images,
                    format_func=(
                        lambda path:
                        path.name
                    )
                )
            )

            st.image(
                str(
                    selected_result_image
                ),
                caption=(
                    "Detected object boundary "
                    "and estimated dimensions"
                ),
                use_container_width=True
            )

    # ========================================================
    # OPTIONAL ERROR STATISTICS
    # ========================================================

    with tab_errors:

        st.subheader(
            "Ground-Truth Validation"
        )

        st.write(
            """
            Enter actual measured dimensions
            where available. The application
            automatically calculates error
            statistics. Blank values are ignored.
            """
        )

        error_table = pd.DataFrame(
            {
                "image":
                    df[
                        "image"
                    ],

                "estimated_width_cm":
                    df[
                        "estimated_width_cm"
                    ],

                "actual_width_cm":
                    np.nan,

                "estimated_height_cm":
                    df[
                        "estimated_height_cm"
                    ],

                "actual_height_cm":
                    np.nan
            }
        )

        edited_errors = st.data_editor(
            error_table,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "image":
                    st.column_config.TextColumn(
                        "Image",
                        disabled=True
                    ),

                "estimated_width_cm":
                    st.column_config.NumberColumn(
                        "Estimated Width (cm)",
                        disabled=True,
                        format="%.2f"
                    ),

                "actual_width_cm":
                    st.column_config.NumberColumn(
                        "Actual Width (cm)",
                        min_value=0.01,
                        format="%.2f"
                    ),

                "estimated_height_cm":
                    st.column_config.NumberColumn(
                        "Estimated Height (cm)",
                        disabled=True,
                        format="%.2f"
                    ),

                "actual_height_cm":
                    st.column_config.NumberColumn(
                        "Actual Height (cm)",
                        min_value=0.01,
                        format="%.2f"
                    )
            }
        )

        absolute_errors = []
        percentage_errors = []

        # Width errors.

        for _, row in (
            edited_errors.iterrows()
        ):

            actual_width = row[
                "actual_width_cm"
            ]

            if (
                pd.notna(
                    actual_width
                )
                and
                actual_width > 0
            ):

                estimated_width = row[
                    "estimated_width_cm"
                ]

                absolute = abs(
                    estimated_width -
                    actual_width
                )

                percentage = (
                    absolute /
                    actual_width *
                    100.0
                )

                absolute_errors.append(
                    absolute
                )

                percentage_errors.append(
                    percentage
                )

            actual_height = row[
                "actual_height_cm"
            ]

            if (
                pd.notna(
                    actual_height
                )
                and
                actual_height > 0
            ):

                estimated_height = row[
                    "estimated_height_cm"
                ]

                absolute = abs(
                    estimated_height -
                    actual_height
                )

                percentage = (
                    absolute /
                    actual_height *
                    100.0
                )

                absolute_errors.append(
                    absolute
                )

                percentage_errors.append(
                    percentage
                )

        if absolute_errors:

            absolute_array = np.array(
                absolute_errors,
                dtype=float
            )

            percentage_array = np.array(
                percentage_errors,
                dtype=float
            )

            mae = float(
                np.mean(
                    absolute_array
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        absolute_array ** 2
                    )
                )
            )

            mape = float(
                np.mean(
                    percentage_array
                )
            )

            error_std = float(
                np.std(
                    absolute_array
                )
            )

            er1, er2, er3, er4 = (
                st.columns(4)
            )

            er1.metric(
                "Validated Dimensions",
                len(
                    absolute_errors
                )
            )

            er2.metric(
                "MAE",
                f"{mae:.2f} cm"
            )

            er3.metric(
                "RMSE",
                f"{rmse:.2f} cm"
            )

            er4.metric(
                "Mean % Error",
                f"{mape:.2f}%"
            )

            st.metric(
                "Error Standard Deviation",
                f"{error_std:.2f} cm"
            )

            st.success(
                "✓ Error statistics calculated "
                "from entered ground-truth values."
            )

        else:

            st.info(
                "Enter at least one actual "
                "width or height to calculate "
                "error statistics."
            )


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "🧠 Projection Theory":

    st.header(
        "🧠 Perspective Projection Theory"
    )

    st.caption(
        "Single-camera projection and "
        "two-camera transformation relationship"
    )

    # ========================================================
    # SINGLE CAMERA
    # ========================================================

    with st.container(
        border=True
    ):

        st.subheader(
            "1. Single-Camera Projection"
        )

        eq1, eq2 = st.columns(
            2
        )

        with eq1:

            st.latex(
                r"""
                u =
                f_x
                \frac{X}{Z}
                +
                c_x
                """
            )

        with eq2:

            st.latex(
                r"""
                v =
                f_y
                \frac{Y}{Z}
                +
                c_y
                """
            )

        st.markdown(
            """
            **Where**

            - \(X,Y,Z\) are 3D coordinates in the camera frame.
            - \(u,v\) are image pixel coordinates.
            - \(f_x,f_y\) are focal lengths in pixels.
            - \(c_x,c_y\) are principal-point coordinates.
            """
        )

    st.divider()

    # ========================================================
    # INVERSE PROJECTION
    # ========================================================

    with st.container(
        border=True
    ):

        st.subheader(
            "2. Recovering Coordinates "
            "at Known Depth"
        )

        inv1, inv2 = st.columns(
            2
        )

        with inv1:

            st.latex(
                r"""
                X =
                \frac{(u-c_x)Z}{f_x}
                """
            )

        with inv2:

            st.latex(
                r"""
                Y =
                \frac{(v-c_y)Z}{f_y}
                """
            )

        st.write(
            """
            Therefore, if the object's depth
            \(Z\) is known, image coordinates
            can be converted into physical
            coordinates on the assumed planar
            depth surface.
            """
        )

    st.divider()

    # ========================================================
    # CAMERA 1
    # ========================================================

    with st.container(
        border=True
    ):

        st.subheader(
            "3. Camera 1 - Static Camera"
        )

        st.write(
            """
            Let \(P_1=[X,Y,Z]^T\) represent
            the 3D point in Camera 1 coordinates.
            """
        )

        st.latex(
            r"""
            \lambda_1
            \tilde{p}_1
            =
            K_1 P_1
            """
        )

        st.write(
            "Therefore:"
        )

        st.latex(
            r"""
            P_1 =
            \lambda_1
            K_1^{-1}
            \tilde{p}_1
            """
        )

    st.divider()

    # ========================================================
    # CAMERA 2
    # ========================================================

    with st.container(
        border=True
    ):

        st.subheader(
            "4. Camera 2 - "
            "Translated and Oblique"
        )

        st.write(
            """
            Camera 2 is related to Camera 1
            through rotation \(R\) and
            translation \(t\).
            """
        )

        st.latex(
            r"""
            P_2 =
            R P_1 + t
            """
        )

        st.write(
            """
            If the Camera 2 center is \(C_2\)
            expressed in Camera 1 coordinates:
            """
        )

        st.latex(
            r"""
            t = -RC_2
            """
        )

        st.write(
            "Camera 2 projection is:"
        )

        st.latex(
            r"""
            \lambda_2
            \tilde{p}_2
            =
            K_2
            \left(
            RP_1+t
            \right)
            """
        )

    st.divider()

    # ========================================================
    # COMBINED RELATIONSHIP
    # ========================================================

    with st.container(
        border=True
    ):

        st.subheader(
            "5. Relationship Between "
            "Image 1 and Image 2"
        )

        st.write(
            """
            Substitute the Camera 1 expression
            for \(P_1\) into Camera 2:
            """
        )

        st.latex(
            r"""
            \lambda_2
            \tilde{p}_2
            =
            K_2
            \left[
            R
            \left(
            \lambda_1
            K_1^{-1}
            \tilde{p}_1
            \right)
            +t
            \right]
            """
        )

        st.write(
            """
            This equation describes how a
            3D point observed by the static
            first camera appears in a second
            translated and obliquely oriented
            camera.
            """
        )

    st.divider()

    # ========================================================
    # EPIPOLAR GEOMETRY
    # ========================================================

    with st.container(
        border=True
    ):

        st.subheader(
            "6. Unknown Depth Case"
        )

        st.write(
            """
            If depth is unknown, a unique
            corresponding image point cannot
            generally be recovered from one
            pixel alone. Instead, the two
            image points satisfy the epipolar
            constraint:
            """
        )

        st.latex(
            r"""
            \tilde{p}_2^T
            F
            \tilde{p}_1
            =
            0
            """
        )

        st.write(
            "where:"
        )

        st.latex(
            r"""
            F =
            K_2^{-T}
            [t]_{\times}
            R
            K_1^{-1}
            """
        )

    st.divider()

    # ========================================================
    # VARIABLES AND ASSUMPTIONS
    # ========================================================

    theory_left, theory_right = (
        st.columns(2)
    )

    with theory_left:

        with st.container(
            border=True
        ):

            st.subheader(
                "Parameters"
            )

            st.markdown(
                """
                - \(K_1,K_2\): camera intrinsic matrices
                - \(R\): relative rotation matrix
                - \(t\): relative translation vector
                - \(C_2\): Camera 2 center
                - \(f_x,f_y\): focal lengths
                - \(c_x,c_y\): principal point
                """
            )

    with theory_right:

        with st.container(
            border=True
        ):

            st.subheader(
                "Variables"
            )

            st.markdown(
                """
                - \(P\): 3D scene point
                - \(p_1\): pixel in Camera 1
                - \(p_2\): pixel in Camera 2
                - \(Z\): object depth
                - \(\lambda_1,\lambda_2\): depth scale factors
                """
            )

    st.divider()

    with st.container(
        border=True
    ):

        st.subheader(
            "Measurement Assumptions"
        )

        st.markdown(
            """
            - The camera is calibrated.
            - Lens distortion is corrected before measurement.
            - The object is approximately planar.
            - The four selected corners are approximately at the same depth.
            - Camera-to-object distance is known.
            - The image is captured with the same camera/lens configuration as the calibration images.
            - Camera 1 is static in the two-camera derivation.
            - Camera 2 undergoes rigid translation and rotation.
            """
        )



# ============================================================
# MODULE 3 - IMAGE FILTERING
# ============================================================

elif page == "🌀 Module 3 - Image Filtering":

    st.header(
        "🌀 Module 3 — Image Filtering"
    )

    st.caption(
        "Spatial-domain convolution • "
        "Fourier-domain multiplication • "
        "Experimental validation"
    )

    with st.container(
        border=True
    ):

        st.subheader(
            "Assignment Objective"
        )

        st.write(
            """
            This experiment implements image blurring
            with a spatial filter and then performs the
            equivalent operation in the Fourier domain.

            The same normalized box-blur kernel is used
            in both approaches so the two numerical
            outputs can be compared directly.
            """
        )

        st.latex(
            r"""
            g(x,y)
            =
            f(x,y) * h(x,y)
            """
        )

        st.latex(
            r"""
            G(u,v)
            =
            F(u,v)\,H(u,v)
            """
        )

        st.info(
            "The convolution theorem predicts that "
            "spatial convolution and Fourier-domain "
            "multiplication produce the same result, "
            "apart from floating-point numerical precision."
        )

    st.divider()

    # --------------------------------------------------------
    # IMAGE SOURCE
    # --------------------------------------------------------

    st.subheader(
        "1. Select or Upload an Image"
    )

    st.write(
        """
        Test the filtering system using the included project
        image or upload a completely new image. The same
        spatial-domain and Fourier-domain processing will be
        performed on either input.
        """
    )

    source_mode = st.radio(
        "Choose Image Source",
        [
            "Use Project Test Image",
            "Upload and Test New Image"
        ],
        horizontal=True,
        key="module3_source_mode"
    )

    module3_pil = None
    module3_image_name = None

    if source_mode == "Use Project Test Image":

        st.info(
            "Using the included Module 3 test image."
        )

        if MODULE3_SAMPLE_IMAGE.exists():

            module3_pil = (
                Image.open(
                    MODULE3_SAMPLE_IMAGE
                )
                .convert(
                    "RGB"
                )
            )

            module3_image_name = (
                MODULE3_SAMPLE_IMAGE.name
            )

        else:

            st.error(
                "The Module 3 sample image was not found."
            )

            st.code(
                "module3/sample_images/test_image.jpg"
            )

    else:

        with st.container(
            border=True
        ):

            st.subheader(
                "📤 Upload Image for Live Test"
            )

            st.write(
                """
                Upload any JPG, JPEG or PNG image.
                The application will process the uploaded
                image directly and compare spatial filtering
                with Fourier-domain filtering.
                """
            )

            module3_uploaded = st.file_uploader(
                "Choose an image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="module3_image_upload"
            )

            if module3_uploaded is not None:

                try:

                    module3_pil = (
                        Image.open(
                            io.BytesIO(
                                module3_uploaded.getvalue()
                            )
                        )
                    )

                    module3_pil = (
                        ImageOps.exif_transpose(
                            module3_pil
                        )
                        .convert(
                            "RGB"
                        )
                    )

                    module3_image_name = (
                        module3_uploaded.name
                    )

                    st.success(
                        "✓ Image uploaded successfully. "
                        "Live filtering results are shown below."
                    )

                except Exception as error:

                    st.error(
                        f"Unable to read image: {error}"
                    )

    if module3_pil is None:

        if (
            source_mode ==
            "Upload and Test New Image"
        ):

            st.warning(
                "Upload an image above to run the "
                "Module 3 experiment."
            )

        else:

            st.warning(
                "The project test image could not be loaded."
            )

    else:

        # Convert RGB image to grayscale for the experiment.
        module3_rgb = np.array(
            module3_pil
        )

        module3_gray = cv2.cvtColor(
            module3_rgb,
            cv2.COLOR_RGB2GRAY
        )

        module3_gray_float = (
            module3_gray.astype(
                np.float64
            )
        )

        image_height, image_width = (
            module3_gray.shape
        )

        i1, i2, i3 = st.columns(
            3
        )

        i1.metric(
            "Image",
            module3_image_name
        )

        i2.metric(
            "Resolution",
            f"{image_width} × {image_height}"
        )

        i3.metric(
            "Processing",
            "Grayscale"
        )

        preview_left, preview_right = (
            st.columns(2)
        )

        with preview_left:

            st.image(
                module3_pil,
                caption="Original RGB image",
                use_container_width=True
            )

        with preview_right:

            st.image(
                module3_gray,
                caption="Grayscale image used for filtering",
                use_container_width=True,
                clamp=True
            )

        st.divider()

        # ----------------------------------------------------
        # LIVE FILTERING EXPERIMENT
        # ----------------------------------------------------

        st.subheader(
            "2. Live Filtering Experiment"
        )

        kernel_size = st.select_slider(
            "Select box-filter kernel size",
            options=[
                3,
                5,
                9,
                15,
                25
            ],
            value=15,
            format_func=lambda value:
                f"{value} × {value}",
            key="module3_kernel_size"
        )

        st.caption(
            "A larger kernel averages a wider "
            "neighborhood and therefore produces "
            "stronger blurring."
        )

        result = module3_compare_methods(
            module3_gray_float,
            kernel_size
        )

        spatial_uint8 = np.clip(
            result["spatial"],
            0,
            255
        ).astype(
            np.uint8
        )

        fourier_uint8 = np.clip(
            result["fourier"],
            0,
            255
        ).astype(
            np.uint8
        )

        difference = result[
            "difference"
        ]

        difference_threshold = 1e-9

        if (
            np.max(difference)
            > difference_threshold
        ):

            difference_visual = (
                cv2.normalize(
                    difference,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )
                .astype(
                    np.uint8
                )
            )

        else:

            difference_visual = (
                np.zeros_like(
                    module3_gray,
                    dtype=np.uint8
                )
            )

        spectrum = (
            module3_fourier_spectrum(
                module3_gray_float
            )
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        with result_col1:

            st.image(
                spatial_uint8,
                caption=(
                    "Spatial-domain blur "
                    f"({kernel_size} × {kernel_size})"
                ),
                use_container_width=True,
                clamp=True
            )

        with result_col2:

            st.image(
                fourier_uint8,
                caption=(
                    "Fourier-domain blur "
                    f"({kernel_size} × {kernel_size})"
                ),
                use_container_width=True,
                clamp=True
            )

        st.subheader(
            "Difference and Frequency Spectrum"
        )

        diff_col, spectrum_col = (
            st.columns(2)
        )

        with diff_col:

            st.image(
                difference_visual,
                caption=(
                    "Difference image "
                    "(black = no measurable difference)"
                ),
                use_container_width=True,
                clamp=True
            )

        with spectrum_col:

            st.image(
                spectrum,
                caption=(
                    "Centered log-magnitude "
                    "Fourier spectrum"
                ),
                use_container_width=True,
                clamp=True
            )

        st.divider()

        # ----------------------------------------------------
        # NUMERICAL VALIDATION
        # ----------------------------------------------------

        st.subheader(
            "3. Numerical Validation"
        )

        v1, v2, v3 = st.columns(
            3
        )

        v1.metric(
            "Mean Absolute Error",
            f"{result['mae']:.12f}"
        )

        v2.metric(
            "Mean Squared Error",
            f"{result['mse']:.12f}"
        )

        v3.metric(
            "Maximum Difference",
            f"{result['max_difference']:.12f}"
        )

        if (
            result["max_difference"]
            < 1e-9
        ):

            st.success(
                "✓ The spatial and Fourier results "
                "match to floating-point precision."
            )

        else:

            st.info(
                "The methods are numerically very close. "
                "Any remaining difference is reported "
                "above and can arise from floating-point "
                "precision or boundary handling."
            )

        st.divider()

        # ----------------------------------------------------
        # MULTI-KERNEL EXPERIMENT
        # ----------------------------------------------------

        st.subheader(
            "4. Multi-Kernel Validation"
        )

        validation_rows = []

        for test_kernel_size in [
            3,
            5,
            9,
            15,
            25
        ]:

            test_result = (
                module3_compare_methods(
                    module3_gray_float,
                    test_kernel_size
                )
            )

            validation_rows.append(
                {
                    "Kernel":
                        (
                            f"{test_kernel_size}"
                            " × "
                            f"{test_kernel_size}"
                        ),

                    "MAE":
                        test_result[
                            "mae"
                        ],

                    "MSE":
                        test_result[
                            "mse"
                        ],

                    "Maximum Difference":
                        test_result[
                            "max_difference"
                        ]
                }
            )

        validation_table = pd.DataFrame(
            validation_rows
        )

        st.dataframe(
            validation_table.style.format(
                {
                    "MAE": "{:.12f}",
                    "MSE": "{:.12f}",
                    "Maximum Difference":
                        "{:.12f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.write(
            """
            Testing several filter sizes demonstrates
            that the equivalence is not limited to one
            particular blur kernel size.
            """
        )

        st.divider()

        # ----------------------------------------------------
        # THEORY
        # ----------------------------------------------------

        st.subheader(
            "5. Convolution Theorem"
        )

        with st.container(
            border=True
        ):

            st.write(
                """
                Let the image be \(f(x,y)\) and the
                spatial blur kernel be \(h(x,y)\).
                Direct spatial filtering gives:
                """
            )

            st.latex(
                r"""
                g(x,y)
                =
                f(x,y) * h(x,y)
                """
            )

            st.write(
                """
                Taking the two-dimensional Fourier
                transform and applying the convolution
                theorem gives:
                """
            )

            st.latex(
                r"""
                \mathcal{F}
                \{
                f*h
                \}
                =
                F(u,v)H(u,v)
                """
            )

            st.write(
                """
                Therefore the same filtered image can
                be recovered with an inverse Fourier
                transform:
                """
            )

            st.latex(
                r"""
                g(x,y)
                =
                \mathcal{F}^{-1}
                \{
                F(u,v)H(u,v)
                \}
                """
            )

            st.success(
                "The live numerical results above "
                "provide experimental validation of "
                "this theorem for the implemented "
                "image-blurring filters."
            )

        st.divider()

        # ----------------------------------------------------
        # PROFESSOR TEST WORKFLOW
        # ----------------------------------------------------

        st.subheader(
            "Professor Test Workflow"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                """
                **1.** Open the Module 3 page.

                **2.** Use the included test image or
                upload a new JPG/PNG image.

                **3.** Select a blur kernel size.

                **4.** Compare the spatial-domain and
                Fourier-domain blurred outputs.

                **5.** Inspect the difference image and
                Fourier magnitude spectrum.

                **6.** Review MAE, MSE and maximum
                difference.

                **7.** Review the multi-kernel validation
                table and convolution-theorem derivation.
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CSc 8830 Computer Vision • "
    "Module 2: Camera Calibration & Measurement • "
    "Module 3: Spatial & Fourier Image Filtering"
)
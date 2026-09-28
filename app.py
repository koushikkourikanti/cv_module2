# ============================================================
# CSc 8830 - COMPUTER VISION
# MODULE 4 WEB APPLICATION
#
# Human Boundary Detection
# RGB + Thermal + Fourier/Frequency Domain Analysis
#
# Author: Koushik Kourikanti
# ============================================================

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Module 4 | CSc 8830",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

MODULE4_FOLDER = Path("module4")
OUTPUT_FOLDER = MODULE4_FOLDER / "outputs"
FREQUENCY_FOLDER = OUTPUT_FOLDER / "frequency_domain"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(37,99,235,0.18), transparent 28%),
            radial-gradient(circle at 92% 10%, rgba(139,92,246,0.14), transparent 25%),
            linear-gradient(135deg, #050814 0%, #091120 48%, #060B16 100%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1629 0%, #080E1B 100%);
        border-right: 1px solid rgba(96,165,250,0.15);
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #CBD5E1 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #E5E7EB !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(
            145deg,
            rgba(21,34,59,0.91),
            rgba(10,19,35,0.86)
        );
        border: 1px solid rgba(96,165,250,0.15) !important;
        border-radius: 18px !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(96,165,250,0.15);
        border-radius: 16px;
        padding: 0.8rem 1rem;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def uploaded_image_to_bgr(uploaded_file):
    """Decode an uploaded image and return OpenCV BGR format."""

    pil_image = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    )

    pil_image = ImageOps.exif_transpose(
        pil_image
    ).convert("RGB")

    rgb_image = np.array(pil_image)

    return cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )


def bgr_to_rgb(image):
    """Convert an OpenCV BGR image for Streamlit display."""

    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def resize_for_processing(
    image,
    max_height=900,
    max_width=1200,
):
    """Resize large images while preserving aspect ratio."""

    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_height / float(height),
        max_width / float(width),
    )

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


def find_output(*names):
    """Return the first existing saved Module 4 output."""

    for name in names:
        candidate = OUTPUT_FOLDER / name

        if candidate.exists():
            return candidate

    return None


def find_frequency_output(*names):
    """Return the first existing saved frequency-domain output."""

    for name in names:
        candidate = FREQUENCY_FOLDER / name

        if candidate.exists():
            return candidate

    return None


def show_saved_image(
    image_path,
    caption,
):
    """Display a saved image when it exists."""

    if (
        image_path is not None
        and Path(image_path).exists()
    ):
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
        return True

    return False


# ============================================================
# Q1 - RGB HUMAN BOUNDARY
# ============================================================

def rgb_human_segmentation(
    image,
    x_percent=(12, 88),
    y_percent=(3, 97),
):
    """
    Classical human segmentation using GrabCut.

    The user supplies an approximate bounding rectangle using
    percentage sliders. No machine-learning/deep-learning model
    is used in this live classical implementation.
    """

    working = resize_for_processing(
        image.copy()
    )

    original = working.copy()
    height, width = original.shape[:2]

    x1 = int(
        np.clip(
            width * x_percent[0] / 100.0,
            0,
            width - 2,
        )
    )

    x2 = int(
        np.clip(
            width * x_percent[1] / 100.0,
            x1 + 2,
            width,
        )
    )

    y1 = int(
        np.clip(
            height * y_percent[0] / 100.0,
            0,
            height - 2,
        )
    )

    y2 = int(
        np.clip(
            height * y_percent[1] / 100.0,
            y1 + 2,
            height,
        )
    )

    rectangle = (
        x1,
        y1,
        max(2, x2 - x1),
        max(2, y2 - y1),
    )

    rectangle_preview = original.copy()

    cv2.rectangle(
        rectangle_preview,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 255),
        3,
    )

    grabcut_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    cv2.grabCut(
        working,
        grabcut_mask,
        rectangle,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )

    binary_mask = np.where(
        (
            (grabcut_mask == cv2.GC_FGD)
            | (grabcut_mask == cv2.GC_PR_FGD)
        ),
        255,
        0,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    clean_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    clean_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise RuntimeError(
            "No foreground contour was detected. "
            "Adjust the person-region sliders and try again."
        )

    human_contour = max(
        contours,
        key=cv2.contourArea,
    )

    contour_area = float(
        cv2.contourArea(human_contour)
    )

    human_mask = np.zeros_like(
        clean_mask
    )

    cv2.drawContours(
        human_mask,
        [human_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    segmented = cv2.bitwise_and(
        original,
        original,
        mask=human_mask,
    )

    boundary = original.copy()

    cv2.drawContours(
        boundary,
        [human_contour],
        -1,
        (0, 0, 255),
        3,
    )

    boundary_only = np.zeros_like(
        original
    )

    cv2.drawContours(
        boundary_only,
        [human_contour],
        -1,
        (255, 255, 255),
        2,
    )

    foreground_pixels = int(
        np.count_nonzero(human_mask)
    )

    foreground_percent = (
        foreground_pixels
        / float(height * width)
        * 100.0
    )

    return {
        "original": original,
        "rectangle_preview": rectangle_preview,
        "mask": human_mask,
        "segmented": segmented,
        "boundary": boundary,
        "boundary_only": boundary_only,
        "contour_area": contour_area,
        "foreground_pixels": foreground_pixels,
        "foreground_percent": foreground_percent,
        "size": (width, height),
    }


# ============================================================
# Q2 - THERMAL HUMAN BOUNDARY
# ============================================================

def thermal_human_segmentation(
    image,
    hot_is_bright=True,
):
    """
    Classical thermal segmentation using grayscale intensity,
    Otsu thresholding, morphology and contour selection.
    """

    working = resize_for_processing(
        image.copy()
    )

    original = working.copy()
    height, width = original.shape[:2]

    gray = cv2.cvtColor(
        working,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    threshold_type = (
        cv2.THRESH_BINARY
        if hot_is_bright
        else cv2.THRESH_BINARY_INV
    )

    otsu_threshold, binary_mask = cv2.threshold(
        blurred,
        0,
        255,
        threshold_type | cv2.THRESH_OTSU,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    clean_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    clean_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise RuntimeError(
            "No thermal foreground contour was detected. "
            "Try the opposite hot/cold polarity."
        )

    image_center = np.array(
        [
            width / 2.0,
            height / 2.0,
        ],
        dtype=np.float64,
    )

    diagonal = max(
        1.0,
        float(np.hypot(width, height)),
    )

    candidates = []

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )

        if area < 0.0025 * width * height:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            center = np.array(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ],
                dtype=np.float64,
            )
        else:
            x, y, w, h = cv2.boundingRect(
                contour
            )

            center = np.array(
                [
                    x + w / 2.0,
                    y + h / 2.0,
                ],
                dtype=np.float64,
            )

        center_distance = float(
            np.linalg.norm(
                center - image_center
            )
            / diagonal
        )

        score = area * max(
            0.25,
            1.0 - center_distance,
        )

        candidates.append(
            (
                score,
                area,
                contour,
            )
        )

    if candidates:
        _, contour_area, human_contour = max(
            candidates,
            key=lambda item: item[0],
        )
    else:
        human_contour = max(
            contours,
            key=cv2.contourArea,
        )

        contour_area = float(
            cv2.contourArea(
                human_contour
            )
        )

    human_mask = np.zeros_like(
        clean_mask
    )

    cv2.drawContours(
        human_mask,
        [human_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    segmented = cv2.bitwise_and(
        original,
        original,
        mask=human_mask,
    )

    boundary = original.copy()

    cv2.drawContours(
        boundary,
        [human_contour],
        -1,
        (0, 0, 255),
        3,
    )

    foreground_pixels = int(
        np.count_nonzero(human_mask)
    )

    foreground_percent = (
        foreground_pixels
        / float(height * width)
        * 100.0
    )

    return {
        "original": original,
        "gray": gray,
        "mask": human_mask,
        "segmented": segmented,
        "boundary": boundary,
        "otsu_threshold": float(otsu_threshold),
        "contour_area": float(contour_area),
        "foreground_pixels": foreground_pixels,
        "foreground_percent": foreground_percent,
        "size": (width, height),
    }


# ============================================================
# Q3 - FREQUENCY DOMAIN
# ============================================================

def frequency_domain_analysis(
    image,
    cutoff_fraction=0.08,
):
    """
    Fourier-domain experiment using NumPy FFT.

    Creates:
    - centered magnitude spectrum
    - ideal low-pass filter
    - ideal high-pass filter
    - inverse FFT results
    - frequency-derived edge map
    - low-frequency segmentation result
    """

    working = resize_for_processing(
        image.copy()
    )

    original = working.copy()

    gray = cv2.cvtColor(
        working,
        cv2.COLOR_BGR2GRAY,
    )

    gray_float = gray.astype(
        np.float64
    )

    height, width = gray.shape

    spectrum = np.fft.fftshift(
        np.fft.fft2(gray_float)
    )

    magnitude = np.log1p(
        np.abs(spectrum)
    )

    magnitude_display = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    y_grid, x_grid = np.ogrid[
        :height,
        :width,
    ]

    center_y = height / 2.0
    center_x = width / 2.0

    distance = np.sqrt(
        (y_grid - center_y) ** 2
        + (x_grid - center_x) ** 2
    )

    cutoff = max(
        5,
        int(
            min(height, width)
            * cutoff_fraction
        ),
    )

    low_pass = (
        distance <= cutoff
    ).astype(np.float64)

    high_pass = 1.0 - low_pass

    low_frequency = (
        spectrum * low_pass
    )

    high_frequency = (
        spectrum * high_pass
    )

    low_spatial = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(
                low_frequency
            )
        )
    )

    high_spatial = np.abs(
        np.fft.ifft2(
            np.fft.ifftshift(
                high_frequency
            )
        )
    )

    low_display = cv2.normalize(
        low_spatial,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    high_display = cv2.normalize(
        high_spatial,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    edge_threshold, edge_map = cv2.threshold(
        high_display,
        0,
        255,
        cv2.THRESH_BINARY
        | cv2.THRESH_OTSU,
    )

    edge_map = cv2.morphologyEx(
        edge_map,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    segmentation_threshold, segmentation_mask = cv2.threshold(
        low_display,
        0,
        255,
        cv2.THRESH_BINARY
        | cv2.THRESH_OTSU,
    )

    segmentation_mask = cv2.morphologyEx(
        segmentation_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
        iterations=2,
    )

    segmentation_mask = cv2.morphologyEx(
        segmentation_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    segmented = cv2.bitwise_and(
        original,
        original,
        mask=segmentation_mask,
    )

    edge_overlay = original.copy()
    edge_overlay[
        edge_map > 0
    ] = (
        0,
        0,
        255,
    )

    return {
        "original": original,
        "gray": gray,
        "magnitude": magnitude_display,
        "low_pass_mask": (
            low_pass * 255
        ).astype(np.uint8),
        "high_pass_mask": (
            high_pass * 255
        ).astype(np.uint8),
        "low_result": low_display,
        "high_result": high_display,
        "edge_map": edge_map,
        "edge_overlay": edge_overlay,
        "segmentation_mask": segmentation_mask,
        "segmented": segmented,
        "cutoff": int(cutoff),
        "edge_threshold": float(
            edge_threshold
        ),
        "segmentation_threshold": float(
            segmentation_threshold
        ),
        "size": (width, height),
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧍 Module 4"
)

st.sidebar.caption(
    "CSc 8830 · Computer Vision"
)

st.sidebar.divider()

st.sidebar.success(
    "● Module 4 Web Application"
)

st.sidebar.markdown(
    """
    **Q1** — RGB Human Boundary  
    **Q2** — Thermal Human Boundary  
    **Q3** — Frequency-Domain Analysis
    """
)

st.sidebar.divider()

st.sidebar.link_button(
    "🔗 Module 4 GitHub",
    "https://github.com/koushikkourikanti/cv_module4",
    use_container_width=True,
)

st.sidebar.caption(
    "Live testing is available directly in this webpage."
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧍 Module 4 — Human Boundary Detection"
)

st.caption(
    "Classical Computer Vision • "
    "SAM2.1 Saved Comparison • "
    "Thermal Imaging • "
    "Fourier/Frequency-Domain Analysis"
)

with st.container(
    border=True
):
    st.subheader(
        "Assignment Overview"
    )

    st.write(
        """
        Module 4 studies human-boundary detection in regular RGB
        and thermal images. Questions 1 and 2 use classical OpenCV
        processing for the live web demonstration. Saved SAM2.1
        results can be shown for comparison when those output files
        are present. Question 3 performs Fourier-domain edge and
        segmentation experiments using NumPy FFT and OpenCV.
        """
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Question 1",
        "RGB Human Boundary",
    )

    m2.metric(
        "Question 2",
        "Thermal Human Boundary",
    )

    m3.metric(
        "Question 3",
        "Frequency Domain",
    )


# ============================================================
# TABS
# ============================================================

rgb_tab, thermal_tab, frequency_tab = st.tabs(
    [
        "🧍 Q1 — RGB Human",
        "🌡️ Q2 — Thermal Human",
        "〰️ Q3 — Frequency Domain",
    ]
)


# ============================================================
# Q1 UI
# ============================================================

with rgb_tab:

    st.header(
        "Q1 — Human Boundary from RGB Image"
    )

    st.write(
        """
        Upload a regular RGB image containing a person.
        The live implementation uses GrabCut, morphology and
        contour extraction to isolate the human region and draw
        its boundary.
        """
    )

    rgb_upload = st.file_uploader(
        "Upload RGB human image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="rgb_upload",
    )

    if rgb_upload is not None:

        rgb_image = uploaded_image_to_bgr(
            rgb_upload
        )

        preview = resize_for_processing(
            rgb_image.copy()
        )

        st.image(
            bgr_to_rgb(preview),
            caption="Uploaded RGB image",
            use_container_width=True,
        )

        st.subheader(
            "Person Region"
        )

        c1, c2 = st.columns(2)

        with c1:
            x_range = st.slider(
                "Horizontal range (%)",
                min_value=0,
                max_value=100,
                value=(12, 88),
                key="rgb_x_range",
            )

        with c2:
            y_range = st.slider(
                "Vertical range (%)",
                min_value=0,
                max_value=100,
                value=(3, 97),
                key="rgb_y_range",
            )

        if st.button(
            "Run RGB Human Boundary Detection",
            type="primary",
            use_container_width=True,
        ):
            try:
                result = rgb_human_segmentation(
                    rgb_image,
                    x_percent=x_range,
                    y_percent=y_range,
                )

                st.success(
                    "RGB human boundary detection completed."
                )

                r1, r2 = st.columns(2)

                with r1:
                    st.image(
                        bgr_to_rgb(
                            result[
                                "rectangle_preview"
                            ]
                        ),
                        caption=(
                            "GrabCut initialization rectangle"
                        ),
                        use_container_width=True,
                    )

                with r2:
                    st.image(
                        result["mask"],
                        caption="Final binary human mask",
                        use_container_width=True,
                        clamp=True,
                    )

                r3, r4 = st.columns(2)

                with r3:
                    st.image(
                        bgr_to_rgb(
                            result["segmented"]
                        ),
                        caption="Segmented person",
                        use_container_width=True,
                    )

                with r4:
                    st.image(
                        bgr_to_rgb(
                            result["boundary"]
                        ),
                        caption="Detected human boundary",
                        use_container_width=True,
                    )

                a, b, c = st.columns(3)

                a.metric(
                    "Contour Area",
                    f"{result['contour_area']:.0f} px²",
                )

                b.metric(
                    "Foreground Pixels",
                    f"{result['foreground_pixels']:,}",
                )

                c.metric(
                    "Foreground",
                    f"{result['foreground_percent']:.2f}%",
                )

            except Exception as exc:
                st.error(
                    f"RGB processing failed: {exc}"
                )

    else:
        st.info(
            "Upload an RGB image to run the live Q1 demonstration."
        )

    st.divider()

    st.subheader(
        "Saved SAM2.1 Comparison"
    )

    st.caption(
        "These results are displayed only when the saved SAM2.1 "
        "output files are available in module4/outputs."
    )

    sam_mask = find_output(
        "rgb_sam2_mask.jpg",
        "sam2_mask.jpg",
    )

    sam_boundary = find_output(
        "rgb_sam2_boundary.jpg",
        "sam2_boundary.jpg",
    )

    sam_segmented = find_output(
        "rgb_sam2_segmented_person.jpg",
        "sam2_segmented_person.jpg",
    )

    s1, s2 = st.columns(2)

    with s1:
        found_mask = show_saved_image(
            sam_mask,
            "SAM2.1 RGB mask",
        )

    with s2:
        found_boundary = show_saved_image(
            sam_boundary,
            "SAM2.1 RGB boundary",
        )

    if sam_segmented is not None:
        show_saved_image(
            sam_segmented,
            "SAM2.1 segmented person",
        )

    if not (
        found_mask
        or found_boundary
        or sam_segmented is not None
    ):
        st.info(
            "No saved SAM2.1 RGB outputs were found. "
            "The live classical Q1 test still works."
        )


# ============================================================
# Q2 UI
# ============================================================

with thermal_tab:

    st.header(
        "Q2 — Human Boundary from Thermal Image"
    )

    st.write(
        """
        Upload a thermal image. The image is converted to
        grayscale, automatically thresholded with Otsu's method,
        cleaned with morphology, and reduced to the strongest
        plausible human foreground region.
        """
    )

    thermal_upload = st.file_uploader(
        "Upload thermal image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="thermal_upload",
    )

    hot_mode = st.radio(
        "Thermal polarity",
        [
            "Hot person appears bright",
            "Hot person appears dark",
        ],
        horizontal=True,
    )

    if thermal_upload is not None:

        thermal_image = uploaded_image_to_bgr(
            thermal_upload
        )

        st.image(
            bgr_to_rgb(
                resize_for_processing(
                    thermal_image.copy()
                )
            ),
            caption="Uploaded thermal image",
            use_container_width=True,
        )

        if st.button(
            "Run Thermal Human Boundary Detection",
            type="primary",
            use_container_width=True,
        ):
            try:
                result = thermal_human_segmentation(
                    thermal_image,
                    hot_is_bright=(
                        hot_mode
                        == "Hot person appears bright"
                    ),
                )

                st.success(
                    "Thermal human boundary detection completed."
                )

                t1, t2 = st.columns(2)

                with t1:
                    st.image(
                        result["gray"],
                        caption="Thermal grayscale",
                        use_container_width=True,
                        clamp=True,
                    )

                with t2:
                    st.image(
                        result["mask"],
                        caption="Thermal human mask",
                        use_container_width=True,
                        clamp=True,
                    )

                t3, t4 = st.columns(2)

                with t3:
                    st.image(
                        bgr_to_rgb(
                            result["segmented"]
                        ),
                        caption="Segmented thermal person",
                        use_container_width=True,
                    )

                with t4:
                    st.image(
                        bgr_to_rgb(
                            result["boundary"]
                        ),
                        caption="Detected thermal boundary",
                        use_container_width=True,
                    )

                a, b, c = st.columns(3)

                a.metric(
                    "Otsu Threshold",
                    f"{result['otsu_threshold']:.2f}",
                )

                b.metric(
                    "Contour Area",
                    f"{result['contour_area']:.0f} px²",
                )

                c.metric(
                    "Foreground",
                    f"{result['foreground_percent']:.2f}%",
                )

            except Exception as exc:
                st.error(
                    f"Thermal processing failed: {exc}"
                )

    else:
        st.info(
            "Upload a thermal image to run the live Q2 demonstration."
        )


# ============================================================
# Q3 UI
# ============================================================

with frequency_tab:

    st.header(
        "Q3 — Fourier/Frequency-Domain Analysis"
    )

    st.write(
        """
        Upload an image and select a frequency cutoff.
        The application computes the 2-D Fourier transform,
        builds ideal low-pass and high-pass filters, performs
        inverse transforms, extracts a high-frequency edge map,
        and produces a low-frequency segmentation result.
        """
    )

    frequency_upload = st.file_uploader(
        "Upload image for frequency analysis",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="frequency_upload",
    )

    cutoff_fraction = st.slider(
        "Cutoff frequency fraction",
        min_value=0.02,
        max_value=0.30,
        value=0.08,
        step=0.01,
    )

    if frequency_upload is not None:

        frequency_image = uploaded_image_to_bgr(
            frequency_upload
        )

        st.image(
            bgr_to_rgb(
                resize_for_processing(
                    frequency_image.copy()
                )
            ),
            caption="Input image",
            use_container_width=True,
        )

        if st.button(
            "Run Frequency-Domain Analysis",
            type="primary",
            use_container_width=True,
        ):
            try:
                result = frequency_domain_analysis(
                    frequency_image,
                    cutoff_fraction=cutoff_fraction,
                )

                st.success(
                    "Frequency-domain analysis completed."
                )

                f1, f2 = st.columns(2)

                with f1:
                    st.image(
                        result["gray"],
                        caption="Grayscale image",
                        use_container_width=True,
                        clamp=True,
                    )

                with f2:
                    st.image(
                        result["magnitude"],
                        caption="Centered log Fourier magnitude",
                        use_container_width=True,
                        clamp=True,
                    )

                f3, f4 = st.columns(2)

                with f3:
                    st.image(
                        result["low_pass_mask"],
                        caption="Ideal low-pass mask",
                        use_container_width=True,
                        clamp=True,
                    )

                with f4:
                    st.image(
                        result["high_pass_mask"],
                        caption="Ideal high-pass mask",
                        use_container_width=True,
                        clamp=True,
                    )

                f5, f6 = st.columns(2)

                with f5:
                    st.image(
                        result["low_result"],
                        caption="Low-frequency reconstruction",
                        use_container_width=True,
                        clamp=True,
                    )

                with f6:
                    st.image(
                        result["high_result"],
                        caption="High-frequency reconstruction",
                        use_container_width=True,
                        clamp=True,
                    )

                f7, f8 = st.columns(2)

                with f7:
                    st.image(
                        result["edge_map"],
                        caption="Frequency-derived edge map",
                        use_container_width=True,
                        clamp=True,
                    )

                with f8:
                    st.image(
                        bgr_to_rgb(
                            result["edge_overlay"]
                        ),
                        caption="Edges over original image",
                        use_container_width=True,
                    )

                f9, f10 = st.columns(2)

                with f9:
                    st.image(
                        result[
                            "segmentation_mask"
                        ],
                        caption=(
                            "Low-frequency segmentation mask"
                        ),
                        use_container_width=True,
                        clamp=True,
                    )

                with f10:
                    st.image(
                        bgr_to_rgb(
                            result["segmented"]
                        ),
                        caption=(
                            "Frequency-domain segmentation result"
                        ),
                        use_container_width=True,
                    )

                a, b, c = st.columns(3)

                a.metric(
                    "Cutoff Radius",
                    f"{result['cutoff']} px",
                )

                b.metric(
                    "Edge Otsu Threshold",
                    f"{result['edge_threshold']:.2f}",
                )

                c.metric(
                    "Segmentation Threshold",
                    f"{result['segmentation_threshold']:.2f}",
                )

                st.divider()

                st.subheader(
                    "Theory"
                )

                st.latex(
                    r"F(u,v)=\sum_x\sum_y f(x,y)"
                    r"e^{-j2\pi(ux/M+vy/N)}"
                )

                st.latex(
                    r"G(u,v)=H(u,v)F(u,v)"
                )

                st.latex(
                    r"g(x,y)=\mathcal{F}^{-1}\{G(u,v)\}"
                )

                st.write(
                    """
                    The low-pass filter preserves slowly varying
                    image structure, while the high-pass filter
                    emphasizes rapid intensity changes associated
                    with boundaries and edges.
                    """
                )

            except Exception as exc:
                st.error(
                    f"Frequency processing failed: {exc}"
                )

    else:
        st.info(
            "Upload an image to run the live Q3 demonstration."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CSc 8830 Computer Vision • "
    "Module 4 • Human Boundary & Frequency-Domain Analysis"
)

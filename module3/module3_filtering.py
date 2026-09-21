import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CSc 8830 - Computer Vision
# Module 3 Assignment
#
# Image Blurring:
# Spatial Domain Convolution vs Fourier Domain Multiplication
#
# This file supports:
# 1. Standalone desktop execution
# 2. Import from the existing Streamlit web application
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "sample_images"
    / "test_image.jpg"
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
# CREATE NORMALIZED BOX KERNEL
# ============================================================

def create_box_kernel(kernel_size):
    """
    Create a normalized square box-blur kernel.

    The sum of all kernel values is 1.0.
    """

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        dtype=np.float64
    )

    kernel = (
        kernel
        / np.sum(kernel)
    )

    return kernel


# ============================================================
# SPATIAL DOMAIN FILTERING
# ============================================================

def spatial_filter(
    image_data,
    kernel
):
    """
    Apply image blurring directly in the spatial domain.

    For the symmetric box kernel used here,
    cv2.filter2D gives the same result as convolution.
    """

    result = cv2.filter2D(
        image_data,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT
    )

    return result


# ============================================================
# FOURIER DOMAIN FILTERING
# ============================================================

def fourier_filter(
    image_data,
    kernel
):
    """
    Apply equivalent linear convolution in the
    Fourier domain.

    Spatial convolution:
        f(x,y) * h(x,y)

    Fourier equivalent:
        F(u,v) H(u,v)
    """

    image_height, image_width = (
        image_data.shape
    )

    kernel_height, kernel_width = (
        kernel.shape
    )


    # --------------------------------------------------------
    # FULL SIZE FOR LINEAR CONVOLUTION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FFT OF IMAGE
    # --------------------------------------------------------

    image_fft = np.fft.fft2(
        image_data,
        s=(
            fft_height,
            fft_width
        )
    )


    # --------------------------------------------------------
    # FFT OF KERNEL
    # --------------------------------------------------------

    kernel_fft = np.fft.fft2(
        kernel,
        s=(
            fft_height,
            fft_width
        )
    )


    # --------------------------------------------------------
    # MULTIPLICATION IN FREQUENCY DOMAIN
    # --------------------------------------------------------

    frequency_product = (
        image_fft
        * kernel_fft
    )


    # --------------------------------------------------------
    # INVERSE FFT
    # --------------------------------------------------------

    full_result = np.fft.ifft2(
        frequency_product
    )

    full_result = np.real(
        full_result
    )


    # --------------------------------------------------------
    # CROP BACK TO ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    offset_y = (
        kernel_height // 2
    )

    offset_x = (
        kernel_width // 2
    )

    cropped_result = full_result[
        offset_y:
        offset_y + image_height,

        offset_x:
        offset_x + image_width
    ]

    return cropped_result


# ============================================================
# FOURIER MAGNITUDE SPECTRUM
# ============================================================

def get_fourier_spectrum(
    image_data
):
    """
    Compute centered logarithmic Fourier magnitude spectrum.
    """

    fft_result = np.fft.fft2(
        image_data
    )

    shifted = np.fft.fftshift(
        fft_result
    )

    magnitude = np.log1p(
        np.abs(
            shifted
        )
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


# ============================================================
# COMPARE BOTH METHODS
# ============================================================

def compare_methods(
    image_data,
    kernel_size
):
    """
    Compare spatial-domain filtering and
    Fourier-domain filtering.

    Returns results and error metrics.
    """

    kernel = create_box_kernel(
        kernel_size
    )

    spatial_result = spatial_filter(
        image_data,
        kernel
    )

    fourier_result = fourier_filter(
        image_data,
        kernel
    )

    difference = np.abs(
        spatial_result
        - fourier_result
    )

    mae = float(
        np.mean(
            difference
        )
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    max_difference = float(
        np.max(
            difference
        )
    )

    return {
        "kernel_size":
            kernel_size,

        "kernel":
            kernel,

        "spatial":
            spatial_result,

        "fourier":
            fourier_result,

        "difference":
            difference,

        "mae":
            mae,

        "mse":
            mse,

        "max_difference":
            max_difference
    }


# ============================================================
# SAVE OUTPUT IMAGES
# ============================================================

def save_outputs(
    image,
    result
):
    """
    Save the representative Module 3 outputs.
    """

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


    # --------------------------------------------------------
    # DIFFERENCE IMAGE
    # --------------------------------------------------------

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
                image,
                dtype=np.uint8
            )
        )


    # --------------------------------------------------------
    # FOURIER SPECTRUM
    # --------------------------------------------------------

    spectrum = get_fourier_spectrum(
        image.astype(
            np.float64
        )
    )


    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "original.jpg"
        ),
        image
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "spatial_blur_15x15.jpg"
        ),
        spatial_uint8
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "fourier_blur_15x15.jpg"
        ),
        fourier_uint8
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "difference_15x15.jpg"
        ),
        difference_visual
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "fourier_spectrum.jpg"
        ),
        spectrum
    )

    return {
        "spatial_uint8":
            spatial_uint8,

        "fourier_uint8":
            fourier_uint8,

        "difference_visual":
            difference_visual,

        "spectrum":
            spectrum
    }


# ============================================================
# STANDALONE DESKTOP DEMO
# ============================================================

def run_desktop_demo():
    """
    Run the complete Module 3 experiment locally.

    This function is not automatically executed
    when imported by Streamlit.
    """

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    image_float = image.astype(
        np.float64
    )

    print(
        "\nImage loaded successfully."
    )

    print(
        "Image path:",
        IMAGE_PATH
    )

    print(
        "Image shape:",
        image.shape
    )


    # ========================================================
    # MULTI-KERNEL VALIDATION
    # ========================================================

    kernel_sizes = [
        3,
        5,
        9,
        15,
        25
    ]

    print(
        "\n=============================================="
    )

    print(
        "MULTI-KERNEL SPATIAL vs FOURIER VALIDATION"
    )

    print(
        "=============================================="
    )

    print(
        f"{'Kernel':<12}"
        f"{'MAE':<20}"
        f"{'MSE':<20}"
        f"{'Max Difference':<20}"
    )

    print(
        "-" * 72
    )

    representative_result = None


    for kernel_size in kernel_sizes:

        result = compare_methods(
            image_float,
            kernel_size
        )

        print(
            f"{kernel_size}x{kernel_size:<8}"
            f"{result['mae']:<20.12f}"
            f"{result['mse']:<20.12f}"
            f"{result['max_difference']:<20.12f}"
        )

        if kernel_size == 15:

            representative_result = (
                result
            )


    print(
        "\nMulti-kernel validation completed."
    )


    # ========================================================
    # SAVE 15x15 REPRESENTATIVE OUTPUT
    # ========================================================

    saved = save_outputs(
        image,
        representative_result
    )

    print(
        "\nSaved output files:"
    )

    print(
        "1. original.jpg"
    )

    print(
        "2. spatial_blur_15x15.jpg"
    )

    print(
        "3. fourier_blur_15x15.jpg"
    )

    print(
        "4. difference_15x15.jpg"
    )

    print(
        "5. fourier_spectrum.jpg"
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    cv2.imshow(
        "Original Image",
        image
    )

    cv2.imshow(
        "Spatial Domain Blur - 15x15",
        saved[
            "spatial_uint8"
        ]
    )

    cv2.imshow(
        "Fourier Domain Blur - 15x15",
        saved[
            "fourier_uint8"
        ]
    )

    cv2.imshow(
        "Difference - Spatial vs Fourier",
        saved[
            "difference_visual"
        ]
    )

    cv2.imshow(
        "Fourier Magnitude Spectrum",
        saved[
            "spectrum"
        ]
    )

    print(
        "\nPress any key inside an image window to close."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# RUN ONLY WHEN EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_desktop_demo()
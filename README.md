# CSc 8830 Computer Vision - Module 2

## VisionMetric

VisionMetric is an interactive computer vision web application developed for the CSc 8830 Computer Vision Module 2 assignment.

## Features

- Smartphone camera calibration using OpenCV
- Checkerboard corner detection
- Camera intrinsic matrix estimation
- Lens distortion estimation
- Reprojection error analysis
- Perspective-based real-world object measurement
- Interactive four-corner object selection
- Validation using multiple object images
- Measurement analytics and visualizations
- Optional error statistics using ground-truth dimensions
- Two-camera perspective projection theory
- Streamlit web application for interactive testing

## Author

Koushik Kourikanti

## Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Streamlit
- Plotly
- Pillow
- streamlit-image-coordinates

## Project Structure

CSc8830_Module2/

- app.py
- step1_camera_calibration.py
- step2_measure_dimensions.py
- step3_batch_measure.py
- requirements.txt
- README.md
- calibration_images/
- calibration_output/
- measurement_images/
- step2_output/
- step3_output/

## Installation

Create a Python virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

## Run the Web Application

Run:

    python -m streamlit run app.py

Streamlit normally opens the application at:

    http://localhost:8501

## Step 1 - Camera Calibration

Run:

    python step1_camera_calibration.py

The calibration uses a checkerboard with 9 x 6 internal corners.

The program calculates:

- Camera intrinsic matrix
- Focal lengths
- Principal point
- Lens distortion coefficients
- Reprojection error
- Undistorted sample image

Calibration outputs are saved inside:

    calibration_output/

## Step 2 - Object Measurement

Run:

    python step2_measure_dimensions.py

For interactive measurement, the user selects four corners in this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

The known camera-to-object distance is used with the calibrated camera model to estimate the physical width and height.

The same measurement process is also available through the Streamlit web application.

## Step 3 - Validation

Run:

    python step3_batch_measure.py

The validation script processes multiple object images and saves measurements to:

    step3_output/step3_measurements.csv

Annotated result images are also saved in the Step 3 output directory.

## Interactive Web Application

The Streamlit application provides:

- Dashboard
- Saved Camera Calibration Results
- Live Camera Calibration Test
- Interactive Object Measurement
- Direct Four-Corner Image Clicking
- Validation Analytics
- Error Statistics
- Perspective Projection Theory

A user can select or upload an object image, enter the camera-to-object distance, click four object corners, and calculate estimated real-world width and height directly in the browser.

## Perspective Projection

For a calibrated camera:

    u = fx(X/Z) + cx

    v = fy(Y/Z) + cy

Therefore:

    X = ((u-cx)Z)/fx

    Y = ((v-cy)Z)/fy

Lens distortion is corrected before calculating the real-world coordinates.

## Two-Camera Geometry

If Camera 2 has rotation R and translation t relative to Camera 1:

    P2 = R P1 + t

The corresponding projection relationship is:

    lambda2 p2 = K2 [R(lambda1 K1^-1 p1) + t]

This describes how a 3D point observed by Camera 1 appears in a translated and obliquely oriented Camera 2.

## Notes

For best measurement accuracy, object images should be captured using the same smartphone camera and lens configuration used during calibration.

The web application is designed so the assignment can be tested interactively instead of only displaying precomputed results.

## GitHub Repository

https://github.com/koushikkourikanti/cv_module2

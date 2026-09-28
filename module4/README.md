# CSc 8830 Computer Vision — Module 4

## Human Boundary Detection, Thermal Segmentation, and Frequency-Domain Analysis

**Student:** Koushik Kourikanti  
**Course:** CSc 8830 — Computer Vision  
**Institution:** Georgia State University  
**Language:** Python  
**Libraries:** OpenCV, NumPy, Streamlit  
**Comparison Model:** SAM2.1  

## Overview

This project contains the implementation for Module 4 of CSc 8830 Computer Vision.

The assignment contains three parts:

1. Human boundary detection from a regular RGB image using classical computer vision.
2. Human boundary detection from a thermal image using classical computer vision.
3. Edge detection and segmentation using Fourier/frequency-domain processing.

The classical RGB and thermal methods do not use machine learning or deep learning. SAM2.1 is used only for comparison.

## Project Structure

```text
module4/
│
├── README.md
├── rgb_human_boundary.py
├── thermal_human_boundary.py
├── sam2_comparison.py
├── thermal_sam2_comparison.py
├── frequency_domain_analysis.py
│
├── images/
│   ├── rgb/
│   │   └── person.jpg
│   └── thermal/
│       └── person_thermal.jpg
│
└── outputs/
    ├── RGB OpenCV outputs
    ├── RGB SAM2 outputs
    ├── Thermal OpenCV outputs
    ├── Thermal SAM2 outputs
    └── frequency_domain/
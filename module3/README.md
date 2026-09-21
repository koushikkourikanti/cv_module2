# CSc 8830 - Computer Vision
## Module 3 Assignment

### Image Blurring using Spatial and Fourier Domain Filtering

**Student:** Koushik Kourikanti  
**Course:** CSc 8830 - Computer Vision  
**University:** Georgia State University  
**Semester:** Fall 2026  

---

## 1. Overview

This project implements image blurring using two equivalent filtering approaches:

1. **Spatial-domain convolution**
2. **Fourier-domain multiplication**

The purpose of the assignment is to experimentally verify the convolution theorem:

\[
f(x,y) * h(x,y)
\longleftrightarrow
F(u,v)H(u,v)
\]

where:

- \(f(x,y)\) is the input image
- \(h(x,y)\) is the blur filter
- \(F(u,v)\) is the Fourier transform of the image
- \(H(u,v)\) is the Fourier transform of the filter

The implementation demonstrates that applying convolution directly in the spatial domain produces the same output as multiplying the Fourier transforms of the image and filter and then applying the inverse Fourier transform.

---

## 2. Assignment Requirements Addressed

This project includes:

- Image blurring using a filtering approach
- Spatial-domain implementation
- Fourier-domain implementation
- Experimental comparison of both methods
- Numerical validation
- Multiple blur-kernel sizes
- Fourier magnitude spectrum visualization
- Difference-image visualization
- Streamlit web application integration
- Live image-upload testing
- GitHub source-code management

---

## 3. Project Structure

```text
module3/
│
├── module3_filtering.py
├── __init__.py
│
├── sample_images/
│   └── test_image.jpg
│
├── outputs/
│   ├── original.jpg
│   ├── spatial_blur_15x15.jpg
│   ├── fourier_blur_15x15.jpg
│   ├── difference_15x15.jpg
│   └── fourier_spectrum.jpg
│
└── README.md
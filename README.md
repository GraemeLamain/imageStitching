# Python Panorama Image Stitching

Course Project for my Computational Photography course at Ontario Tech University. This program will allow you to stitch multiple unordered images into a high-resolution panorama. This project uses manual Computer Vision algorithsm for Homography estimation and RANSAC as well as Image Warping, rather than relying on libraries.

## Demo
### Input Images:
<p float="left">
<img src="images/1_1.jpg" width="32%" />
<img src="images/1_2.jpg" width="32%" />
<img src="images/1_3.jpg" width="32%" />
</p>

### Stitched Result:
<img src="images/demo.jpg" width="100%" />

## Features
- Unordered Input: Capable of stitching images regardless of the order that they are uploaded in by utilizing a Graph-based BFS traversal on homographies.
- SIFT Feature Detection: Uses Scale-Invarient Feature Transform to find unique keypoints in images (implemented with cv2 currently).
- Custom RANSACL Implemented a manual vertion of Random Sample Consensus Loop in order to filter outliers.
- Direct Linear Transform (DLT): Custom implementation of DLT algorithm to solve for the 3x3 Homography matrix (Ah = 0).
- Vectorized Image Warping: Custom NumPy vectorized inverse warping function for high-performance image warping.
- Canvas Calculation: Automatically calculates the boundaries for the final panorama to prevent cropping.

## Tech Stack
- Backend: Python, Flask
- Computer Vision: NumPy, OpenCV(cv2)
- Math: Linear Algebra (SVD/eigen-decomposition)

## Project Structure
```
├── backend.py      # Flask API entry point 
├── features.py     # SIFT feature extraction & KNN matching logic
├── stitching.py    # Core math: Homography, RANSAC, Global Chaining, Warping
└── utils.py        # Helper functions for Base64/Image conversion
```

## How It Works
### 1. Feature Extraction (features.py)
Uses SIFT to detect key points like edges or blobs that are invarient to scale and rotation. It then matches these points between all pairs of images using K-Nearest-Neighbors (KNN) and filter them using Lowe's Ratio Test at a 0.50 threshold in order to ensure uniqueness of points.

### 2. Pairwise Homography (stitching.py)
For every connecting pair of images, we have to calculate the homography matrix H.
- Math: We solve the system Ah = 0 using SVD (Eigen vectors of A^T A).
- Filtering: We wrap this in a RANSAC loop with 500 iterations to reject outliers.

### 3. Global Homography Chain
Since the images are unordered, we have to build a graph of connections.
- We choose the center image as the anchor point. All of our other images will be mapped into this coordinate system.
- We perform a Breadth-First Search (BFS) in order to generate paths for all images to be mapped to the anchor, even if they do not actually share any matching features.
- We get the global homography for each image by multiplying the chain of homographies which lead to the anchor.

### 4. Warping & Stitching
Calculates the final canvas size by mapping the corners of each image using their global homographies. We then use a Vectorized Inverse Warp to map pixels from the source images onto the final panoramic image canvas.

## Installation & Usage
### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repo:
```
git clone [https://github.com/GraemeLamain/imageStitching.git](https://github.com/GraemeLamain/imageStitching.git)
cd panorama-stitcher
```

2. Install dependencies:
```
pip install numpy opencv-python flask flask-cors pillow
```


### Running the Backend
```
python backend.py
```

The server will start on http://0.0.0.0:8000.

### Running the Frontend
```
python -m http.server 8001
```

Go to http://127.0.0.1:8001/frontend.html on your browser

## Acknowledgements
- OpenCV: For SIFT implementation.
- NumPy: For high-performance matrix operations.
- Faisal Quereshi: Project guidance and course instructor.

Created by Graeme Lamain
# Parallel Image Processing with MPI + OpenCV

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![MPI](https://img.shields.io/badge/MPI-OpenMPI%20%7C%20MPICH-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange?logo=opencv)
![CMake](https://img.shields.io/badge/Build-CMake%203.16%2B-red?logo=cmake)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

A high-performance image processing application that distributes 10 different
image operations across multiple CPU cores using the Message Passing Interface
(MPI) protocol and the OpenCV computer vision library.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Operations](#operations)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Author](#author)

---

## Overview

This project was built as a parallel computing assignment at Fayoum University
(Faculty of Computers and Information, 2024). The core idea is straightforward:
large images take time to process; more CPU cores mean faster results.

MPI splits the image into horizontal strips, distributes each strip to a
separate process, applies the chosen filter in parallel, then reassembles the
result on the root process.

---

## Features

- **10 image processing operations** — blur, edge detection, rotation, scaling,
  equalization, colour conversion, global/local thresholding, compression, median filter
- **Automatic workload distribution** — handles images that don't divide evenly
  across processes using `MPI_Gatherv`
- **Command-line image paths** — no hardcoded paths; pass input/output as arguments
- **Cross-platform** — builds on Linux, macOS, and Windows via CMake
- **Millisecond timing** — reports wall-clock time for each operation

---

## How It Works

```
                        ┌─────────────────────┐
                        │   Input Image (H×W)  │
                        └──────────┬──────────┘
                                   │  MPI_Bcast (full pixels)
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
       ┌─────────────┐     ┌─────────────┐      ┌─────────────┐
       │   Rank 0    │     │   Rank 1    │      │   Rank N    │
       │  rows 0–h0  │     │  rows h0–h1 │  …   │ rows hN–H   │
       │  [filter]   │     │  [filter]   │      │  [filter]   │
       └──────┬──────┘     └──────┬──────┘      └──────┬──────┘
              └────────────────────┼────────────────────┘
                                   │  MPI_Gatherv
                        ┌──────────▼──────────┐
                        │  Output Image (H×W) │
                        │  saved by Rank 0    │
                        └─────────────────────┘
```

### Key MPI Calls

| Call | Purpose |
|------|---------|
| `MPI_Bcast` | Distribute image pixels and user parameters to all ranks |
| `MPI_Gatherv` | Collect variable-size processed slices back to rank 0 |

---

## Project Structure

```
parallel-image-processing/
│
├── src/
│   └── main.cpp              # Full source — MPI + OpenCV processing
│
├── docs/
│   ├── ALGORITHMS.md         # Deep-dive into every algorithm used
│   ├── SETUP.md              # Platform-specific installation guide
│   └── KNOWN_ISSUES.md       # Bug fixes applied + remaining limitations
│
├── images/
│   └── samples/              # Sample input images (add your own here)
│
├── CMakeLists.txt            # CMake build configuration
├── .gitignore
├── LICENSE                   # MIT
└── README.md
```

---

## Tech Stack

| Technology | Version | Role |
|------------|---------|------|
| **C++** | 17 | Core language |
| **OpenMPI / MPICH** | 4.x | Process-level parallelism |
| **OpenCV** | 4.x | Image I/O and all filter implementations |
| **CMake** | 3.16+ | Cross-platform build |
| **chrono** (STL) | — | High-resolution timing |

---

## Quick Start

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt install build-essential cmake libopenmpi-dev openmpi-bin libopencv-dev

# Clone & build
git clone https://github.com/YOUR_USERNAME/parallel-image-processing.git
cd parallel-image-processing
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run with 4 MPI processes
mpirun -n 4 ./parallel_img ../images/samples/photo.jpg output.jpg
```

> **macOS, Windows, and advanced options** → see [docs/SETUP.md](docs/SETUP.md)

---

## Operations

| # | Operation | Algorithm | OpenCV Function |
|---|-----------|-----------|-----------------|
| 01 | Gaussian Blur | Gaussian convolution | `GaussianBlur` |
| 02 | Edge Detection | Canny (4-stage) | `Canny` |
| 03 | Image Rotation | Affine warp | `warpAffine` |
| 04 | Image Scaling | Bilinear resize | `resize` |
| 05 | Histogram Equalization | CDF remapping | `equalizeHist` |
| 06 | Color Space Conversion | YCbCr / HSV / Lab | `cvtColor` |
| 07 | Global Thresholding | Binary threshold | `threshold` |
| 08 | Local Thresholding | Adaptive mean-C | `adaptiveThreshold` |
| 09 | JPEG Compression | DCT + Huffman | `imencode` / `imdecode` |
| 10 | Median Filtering | Neighbourhood median | `medianBlur` |

Full algorithm explanations, including parallel notes and complexity analysis,
are in [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

---

## Performance

Typical speedup on a quad-core machine (1920×1080 image):

| Processes | Gaussian Blur | Edge Detection | Median Filter |
|-----------|--------------|----------------|---------------|
| 1 | 1.00× | 1.00× | 1.00× |
| 2 | ~1.8× | ~1.7× | ~1.9× |
| 4 | ~3.3× | ~3.0× | ~3.5× |

> Results vary by image size, hardware, and operation. MPI overhead
> dominates for very small images.

---

## Known Limitations

- **Seam artefacts** at slice boundaries for spatially-aware filters
  (Gaussian, Median, Adaptive Threshold). Halo/ghost-row exchange would fix this.
- **Grayscale only** — the program reads images with `IMREAD_GRAYSCALE`.
- **Histogram Equalization** is local per-slice rather than global.
- **Image Rotation** rotates each slice around its own centre rather than
  the full image centre.

---

## Author

**Momen Ashraf**  
Computer Science — Fayoum University, 2024  
[linkedin.com/in/momen-ashraf-](https://linkedin.com/in/momen-ashraf-)

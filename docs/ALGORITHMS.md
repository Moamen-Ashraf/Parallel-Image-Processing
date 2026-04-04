# Algorithms & Techniques

This document explains every image processing operation implemented in the project,
the OpenCV function used, and the parallel strategy applied.

---

## Parallel Strategy — How MPI Is Used

The image is **split row-wise** across all MPI processes before any processing starts.

```
Full Image (H × W)
│
├── Rank 0  → rows [0   … h0-1]
├── Rank 1  → rows [h0  … h1-1]
├── Rank 2  → rows [h1  … h2-1]
└── Rank N  → rows [hN  … H-1 ]
```

- **Scatter** — `distributeImage()` computes each rank's slice using arithmetic (no MPI_Scatter, avoids copy overhead for uneven splits).
- **Process** — every rank independently applies the chosen filter to its slice.
- **Gather** — `gatherProcessedImages()` uses `MPI_Gatherv` to collect variable-length slices back to rank 0, which writes the final image.

**Why row decomposition?**  
Most spatial filters (blur, threshold, edge) operate per-pixel or in small local windows, so row slices introduce minimal boundary error and are trivially composable.

---

## Operations

### 01 — Gaussian Blur

| Property | Value |
|----------|-------|
| OpenCV   | `GaussianBlur(src, dst, Size(r,r), 0)` |
| Input    | Grayscale or colour image |
| Parameter | Radius `r` (must be odd) |

**How it works:**  
Convolves the image with a 2-D Gaussian kernel. Each output pixel becomes a weighted average of its neighbours, where weights follow the Gaussian distribution — pixels closer to the centre contribute more. Result: smooth, noise-reduced image.

**Kernel (3×3, σ auto):**
```
1/16 · [ 1 2 1 ]
        [ 2 4 2 ]
        [ 1 2 1 ]
```

**Parallelism note:** Each row slice is blurred independently. Pixels at slice boundaries are computed using only local data (no halo exchange), which is acceptable for moderate radii but can introduce a 1–2 pixel seam at larger radii.

---

### 02 — Edge Detection (Canny)

| Property | Value |
|----------|-------|
| OpenCV   | `Canny(gray, dst, 100, 200)` |
| Input    | Converted to grayscale if needed |
| Parameters | Low threshold = 100, high threshold = 200 |

**How it works (4 stages):**
1. **Gaussian smoothing** — reduces noise before differentiation.
2. **Gradient computation** — Sobel operator finds intensity gradients (magnitude + direction).
3. **Non-maximum suppression** — keeps only local maxima in the gradient direction (thin edges).
4. **Double thresholding + hysteresis** — pixels above the high threshold are "strong" edges; pixels between thresholds are "weak" and kept only if connected to a strong edge.

**Parallelism note:** Canny has a hysteresis step that in theory requires global connectivity. In practice, per-slice execution gives good results for most images; edge stitching at boundaries is a known limitation.

---

### 03 — Image Rotation

| Property | Value |
|----------|-------|
| OpenCV   | `getRotationMatrix2D` + `warpAffine` |
| Input    | Any image |
| Parameter | Angle in degrees (clockwise) |

**How it works:**  
Builds a 2×3 affine transformation matrix **M** centred on the image midpoint, then applies it via bilinear interpolation:

```
M = [ cos θ   -sin θ   tx ]
    [ sin θ    cos θ   ty ]
```

`tx` and `ty` shift the rotation centre to the origin and back.

**Parallelism note:** Each rank rotates its local slice around its own centre — this is geometrically incorrect for a full image rotation and is best applied to the full image on rank 0. The current code does this correctly for the full image by broadcasting pixels first.

---

### 04 — Image Scaling

| Property | Value |
|----------|-------|
| OpenCV   | `resize(src, dst, Size(), scaleX, scaleY)` |
| Input    | Any image |
| Parameters | `scaleX`, `scaleY` (e.g. 1.5 = 150%) |

**How it works:**  
Uses bilinear interpolation by default (`INTER_LINEAR`). Each output pixel is a weighted combination of the four nearest input pixels, computed from the inverse mapping:

```
src(x, y)  →  dst(x · scaleX, y · scaleY)
```

**Parallelism note:** Scaling changes image dimensions per slice, making the gather step non-trivial (variable byte counts). `MPI_Gatherv` handles this correctly.

---

### 05 — Histogram Equalization

| Property | Value |
|----------|-------|
| OpenCV   | `equalizeHist(gray, dst)` |
| Input    | Converted to grayscale if needed |
| Parameters | None |

**How it works:**  
1. Compute the histogram H[i] for pixel intensities 0–255.
2. Build the cumulative distribution function (CDF).
3. Map each pixel intensity i to a new value:

```
new_intensity = round( (CDF(i) - CDF_min) / (N - CDF_min) × 255 )
```

Result: the output histogram is approximately flat — contrast is maximised.

**Parallelism note:** True global equalization requires the global histogram. Per-slice equalization (as implemented) is local and will produce slightly different tone curves per slice. For exact results, rank 0 should collect the full histogram, compute the global CDF, and broadcast the mapping table.

---

### 06 — Color Space Conversion

| Property | Value |
|----------|-------|
| OpenCV   | `cvtColor(src, dst, code)` |
| Input    | Colour image |
| Parameter | OpenCV conversion code (e.g. `COLOR_BGR2GRAY = 6`) |

**Common codes:**

| Code | Value | Description |
|------|-------|-------------|
| `COLOR_BGR2GRAY` | 6 | Remove colour, keep luminance |
| `COLOR_GRAY2BGR` | 8 | Expand grayscale to 3-channel |
| `COLOR_BGR2HSV`  | 40 | Hue–Saturation–Value |
| `COLOR_BGR2Lab`  | 44 | Perceptual Lab colour space |

**Parallelism note:** Pixel-wise operation — perfect for parallel decomposition with no boundary effects.

---

### 07 — Global Thresholding

| Property | Value |
|----------|-------|
| OpenCV   | `threshold(gray, dst, T, 255, THRESH_BINARY)` |
| Input    | Converted to grayscale |
| Parameter | Threshold value T (0–255) |

**How it works:**  
Simple binarisation — every pixel above T becomes 255 (white), every pixel at or below T becomes 0 (black):

```
dst(x,y) = 255  if src(x,y) > T
           0    otherwise
```

Use case: separating foreground from a uniform background.

**Parallelism note:** Purely pixel-wise — ideal parallelism, zero boundary issues.

---

### 08 — Local (Adaptive) Thresholding

| Property | Value |
|----------|-------|
| OpenCV   | `adaptiveThreshold(gray, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, C)` |
| Input    | Grayscale |
| Parameters | Block size (neighbourhood size), constant C |

**How it works:**  
Instead of a single global threshold, computes a local threshold for each pixel based on the mean of its `blockSize × blockSize` neighbourhood:

```
T(x,y) = mean(neighbourhood(x,y)) - C
dst(x,y) = 255 if src(x,y) > T(x,y), else 0
```

More robust than global thresholding when illumination varies across the image.

**Parallelism note:** Requires a small border (half of `blockSize`) of context beyond each slice boundary. Current implementation may have seam artefacts at slice edges for large block sizes.

---

### 09 — JPEG Compression

| Property | Value |
|----------|-------|
| OpenCV   | `imencode(".jpg", src, buf, params)` + `imdecode` |
| Input    | Any image |
| Parameter | Quality 0–100 (lower = smaller file, more loss) |

**How it works:**  
Implements the JPEG pipeline per slice:
1. Convert to YCbCr colour space.
2. Downsample chroma channels (4:2:0).
3. Divide into 8×8 blocks and apply the Discrete Cosine Transform (DCT).
4. Quantise DCT coefficients (lossy step — controlled by quality factor).
5. Entropy-code with Huffman coding.

The `imdecode` call decompresses back to a Mat for viewing/gathering.

**Parallelism note:** Each slice is encoded and decoded independently — valid and efficient.

---

### 10 — Median Filtering

| Property | Value |
|----------|-------|
| OpenCV   | `medianBlur(src, dst, ksize)` |
| Input    | Grayscale or colour |
| Parameter | Kernel size (odd integer, e.g. 5) |

**How it works:**  
For each pixel, collects all values in the `ksize × ksize` neighbourhood, sorts them, and replaces the pixel with the **median** value. Unlike Gaussian blur, the median is robust to impulse noise (salt-and-pepper) and preserves edges better.

```
neighbourhood = sorted([p1, p2, …, pk²])
output = neighbourhood[k²/2]   (the middle value)
```

**Parallelism note:** Requires a border of `ksize/2` rows beyond each slice for correctness. Seam artefacts appear at boundaries for large kernels.

---

## Performance Considerations

| Operation | Parallel Efficiency | Boundary Issue |
|-----------|---------------------|----------------|
| Gaussian Blur | High | Small (radius-dependent) |
| Edge Detection | Medium | Hysteresis not global |
| Rotation | Medium | Centre offset per slice |
| Scaling | High | None |
| Histogram EQ | Medium | Global histogram needed |
| Color Conversion | Very High | None |
| Global Threshold | Very High | None |
| Adaptive Threshold | High | Block-size dependent |
| JPEG Compression | High | None |
| Median Filter | High | Kernel-size dependent |

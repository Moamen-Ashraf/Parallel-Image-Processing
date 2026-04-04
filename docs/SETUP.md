# Setup & Installation

## Prerequisites

| Dependency | Minimum Version | Purpose |
|------------|-----------------|---------|
| C++ compiler | C++17 | GCC 9+, Clang 10+, or MSVC 2019+ |
| CMake | 3.16 | Build system |
| OpenMPI / MPICH | 4.0+ | MPI runtime & headers |
| OpenCV | 4.x | Image processing |

---

## Linux (Ubuntu / Debian)

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake libopenmpi-dev openmpi-bin libopencv-dev

# 2. Verify versions
mpirun --version
opencv_version 2>/dev/null || pkg-config --modversion opencv4

# 3. Clone and build
git clone https://github.com/YOUR_USERNAME/parallel-image-processing.git
cd parallel-image-processing

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. Run with 4 processes
mpirun -n 4 ./parallel_img ../images/samples/sample.jpg output.jpg
```

---

## macOS (Homebrew)

```bash
brew install cmake open-mpi opencv

git clone https://github.com/YOUR_USERNAME/parallel-image-processing.git
cd parallel-image-processing

mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu)

mpirun -n 4 ./parallel_img ../images/samples/sample.jpg output.jpg
```

---

## Windows (MSVC + vcpkg)

```powershell
# 1. Install vcpkg (if not installed)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && .\bootstrap-vcpkg.bat

# 2. Install dependencies
.\vcpkg install opencv4:x64-windows msmpi:x64-windows

# 3. Build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release

# 4. Run
mpiexec -n 4 parallel_img.exe ..\images\samples\sample.jpg output.jpg
```

> **Note:** Microsoft MPI (MS-MPI) is required on Windows. Download from:
> https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

---

## Running the Program

```
Usage: mpirun -n <processes> ./parallel_img <input_image> <output_image>

Arguments:
  <processes>    Number of MPI processes (e.g. 2, 4, 8)
  <input_image>  Path to input JPEG/PNG image
  <output_image> Path for the processed output image (default: output.jpg)
```

**Example session:**
```
$ mpirun -n 4 ./parallel_img photo.jpg result.jpg

  ╔══════════════════════════════════════════════════╗
  ║   Parallel Image Processing with MPI + OpenCV   ║
  ╚══════════════════════════════════════════════════╝

  01 - Gaussian Blur
  02 - Edge Detection (Canny)
  ...
  Enter your choice (1-10): 1

  Blur radius (odd integer, e.g. 5): 7

  Done in 0.023 s  →  saved to result.jpg
```

---

## Choosing the Number of Processes

| Image Size | Recommended `-n` |
|------------|-----------------|
| < 1 MP | 2–4 |
| 1–8 MP | 4–8 |
| > 8 MP | 8–16 |

More processes help for large images; for small images, MPI overhead can outweigh the speedup.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `mpirun: command not found` | Install OpenMPI: `sudo apt install openmpi-bin` |
| `cannot open image` | Check the image path and that OpenCV can decode the format |
| Seam artefacts in output | Use fewer processes or increase image size |
| `MPI_ERR_TRUNCATE` on gather | Ensure all ranks process equal-typed images (same channels) |

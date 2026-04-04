#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// ─────────────────────────────────────────────────────────────────────────────
// distributeImage
//   Splits the full image row-wise across all MPI processes.
//   Handles uneven splits by giving the first (totalRows % world_size)
//   processes one extra row.
// ─────────────────────────────────────────────────────────────────────────────
void distributeImage(const Mat& inputImage, Mat& localImage,
                     int world_rank, int world_size) {
    int totalRows      = inputImage.rows;
    int rowsPerProcess = totalRows / world_size;
    int remainingRows  = totalRows % world_size;

    int startRow   = world_rank * rowsPerProcess + min(world_rank, remainingRows);
    int rowsToSend = (world_rank < remainingRows) ? rowsPerProcess + 1 : rowsPerProcess;

    Rect roi(0, startRow, inputImage.cols, rowsToSend);
    localImage = inputImage(roi).clone();
}

// ─────────────────────────────────────────────────────────────────────────────
// gatherProcessedImages
//   Collects processed row-slices from all processes back to rank 0.
//   Uses MPI_Gatherv to correctly handle variable row counts per process.
// ─────────────────────────────────────────────────────────────────────────────
void gatherProcessedImages(const Mat& localImage, Mat& gatheredImage,
                           int world_rank, int world_size) {
    int localSize = (int)(localImage.total() * localImage.elemSize());

    vector<int> recvCounts(world_size);
    vector<int> displacements(world_size);

    MPI_Gather(&localSize, 1, MPI_INT,
               recvCounts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int totalBytes = 0;
        for (int i = 0; i < world_size; i++) {
            displacements[i] = totalBytes;
            totalBytes += recvCounts[i];
        }
        // Reconstruct as a flat byte buffer then reshape
        gatheredImage.create(
            localImage.rows * world_size,   // approximate; reshape after gather
            localImage.cols,
            localImage.type()
        );
    }

    MPI_Gatherv(
        localImage.data,  localSize,        MPI_BYTE,
        (world_rank == 0 ? gatheredImage.data : nullptr),
        recvCounts.data(), displacements.data(), MPI_BYTE,
        0, MPI_COMM_WORLD
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// printMenu  –  shown only on rank 0
// ─────────────────────────────────────────────────────────────────────────────
void printMenu() {
    cout << "\n";
    cout << "  ╔══════════════════════════════════════════════════╗\n";
    cout << "  ║   Parallel Image Processing with MPI + OpenCV   ║\n";
    cout << "  ╚══════════════════════════════════════════════════╝\n\n";
    cout << "  01 - Gaussian Blur\n";
    cout << "  02 - Edge Detection (Canny)\n";
    cout << "  03 - Image Rotation\n";
    cout << "  04 - Image Scaling\n";
    cout << "  05 - Histogram Equalization\n";
    cout << "  06 - Color Space Conversion\n";
    cout << "  07 - Global Thresholding\n";
    cout << "  08 - Local (Adaptive) Thresholding\n";
    cout << "  09 - Image Compression (JPEG)\n";
    cout << "  10 - Median Filtering\n\n";
    cout << "  Enter your choice (1-10): ";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // ── Image paths (override via argv[1] and argv[2]) ──────────────────────
    string imagePath = (argc > 1) ? argv[1] : "input.jpg";
    string savedPath = (argc > 2) ? argv[2] : "output.jpg";

    // ── Load image on rank 0, broadcast dimensions, scatter data ────────────
    Mat inputImage;
    if (world_rank == 0) {
        inputImage = imread(imagePath, IMREAD_GRAYSCALE);
        if (inputImage.empty()) {
            cerr << "[ERROR] Cannot open image: " << imagePath << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions so all ranks can prepare buffers
    int imgRows = 0, imgCols = 0;
    if (world_rank == 0) { imgRows = inputImage.rows; imgCols = inputImage.cols; }
    MPI_Bcast(&imgRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0)
        inputImage.create(imgRows, imgCols, CV_8UC1);

    // Broadcast the full image pixels to all ranks before slicing
    MPI_Bcast(inputImage.data, imgRows * imgCols, MPI_BYTE, 0, MPI_COMM_WORLD);

    Mat localImage;
    distributeImage(inputImage, localImage, world_rank, world_size);

    // ── User menu (rank 0 only) ──────────────────────────────────────────────
    int choice = 0;
    if (world_rank == 0) {
        printMenu();
        cout.flush();
        cin >> choice;
        if (cin.fail() || choice < 1 || choice > 10) {
            cerr << "[ERROR] Invalid choice. Must be 1-10.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ── Per-operation parameter collection + broadcast ───────────────────────
    Mat resultImage;
    auto start = chrono::steady_clock::now();

    // ── 01  Gaussian Blur ─────────────────────────────────────────────────────
    if (choice == 1) {
        int blurRadius = 9;
        if (world_rank == 0) {
            cout << "\nBlur radius (odd integer, e.g. 5): ";
            cin >> blurRadius;
            if (blurRadius % 2 == 0) blurRadius++;   // must be odd
        }
        MPI_Bcast(&blurRadius, 1, MPI_INT, 0, MPI_COMM_WORLD);

        GaussianBlur(localImage, resultImage, Size(blurRadius, blurRadius), 0);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 02  Edge Detection ────────────────────────────────────────────────────
    else if (choice == 2) {
        Mat gray = (localImage.channels() > 1)
                   ? [&]{ Mat g; cvtColor(localImage, g, COLOR_BGR2GRAY); return g; }()
                   : localImage.clone();
        Canny(gray, resultImage, 100, 200);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 03  Image Rotation ────────────────────────────────────────────────────
    else if (choice == 3) {
        double angle = 45.0;
        if (world_rank == 0) {
            cout << "\nRotation angle (degrees): ";
            cin >> angle;
        }
        MPI_Bcast(&angle, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        Point2f center(localImage.cols / 2.0f, localImage.rows / 2.0f);
        Mat M = getRotationMatrix2D(center, angle, 1.0);
        warpAffine(localImage, resultImage, M, localImage.size());
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 04  Image Scaling ─────────────────────────────────────────────────────
    else if (choice == 4) {
        double scaleX = 1.5, scaleY = 1.5;
        if (world_rank == 0) {
            cout << "\nScale X: "; cin >> scaleX;
            cout << "Scale Y: "; cin >> scaleY;
        }
        MPI_Bcast(&scaleX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&scaleY, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        resize(localImage, resultImage, Size(), scaleX, scaleY);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 05  Histogram Equalization ────────────────────────────────────────────
    else if (choice == 5) {
        Mat gray = (localImage.channels() > 1)
                   ? [&]{ Mat g; cvtColor(localImage, g, COLOR_BGR2GRAY); return g; }()
                   : localImage.clone();
        equalizeHist(gray, resultImage);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 06  Color Space Conversion ────────────────────────────────────────────
    else if (choice == 6) {
        int code = COLOR_BGR2GRAY;
        if (world_rank == 0) {
            cout << "\nConversion code (e.g. 6=BGR2GRAY, 8=GRAY2BGR): ";
            cin >> code;
        }
        MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (localImage.channels() > 1)
            cvtColor(localImage, resultImage, code);
        else
            resultImage = localImage.clone();
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 07  Global Thresholding ───────────────────────────────────────────────
    else if (choice == 7) {
        int threshVal = 128;
        if (world_rank == 0) {
            cout << "\nThreshold value (0-255): ";
            cin >> threshVal;
        }
        MPI_Bcast(&threshVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

        Mat gray = (localImage.channels() > 1)
                   ? [&]{ Mat g; cvtColor(localImage, g, COLOR_BGR2GRAY); return g; }()
                   : localImage.clone();
        threshold(gray, resultImage, threshVal, 255, THRESH_BINARY);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 08  Local (Adaptive) Thresholding ────────────────────────────────────
    else if (choice == 8) {
        int blockSize = 11;
        double C = 2.0;
        if (world_rank == 0) {
            cout << "\nBlock size (odd, e.g. 11): "; cin >> blockSize;
            cout << "Constant C: ";                  cin >> C;
            if (blockSize % 2 == 0) blockSize++;
        }
        MPI_Bcast(&blockSize, 1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&C,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        Mat gray = (localImage.channels() > 1)
                   ? [&]{ Mat g; cvtColor(localImage, g, COLOR_BGR2GRAY); return g; }()
                   : localImage.clone();
        adaptiveThreshold(gray, resultImage, 255,
                          ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, C);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 09  JPEG Compression ──────────────────────────────────────────────────
    else if (choice == 9) {
        int quality = 70;
        if (world_rank == 0) {
            cout << "\nJPEG quality (0-100, lower = more compressed): ";
            cin >> quality;
        }
        MPI_Bcast(&quality, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> params = { IMWRITE_JPEG_QUALITY, quality };
        vector<uchar> buf;
        imencode(".jpg", localImage, buf, params);
        resultImage = imdecode(buf, IMREAD_GRAYSCALE);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    // ── 10  Median Filtering ──────────────────────────────────────────────────
    else if (choice == 10) {
        int ksize = 5;
        if (world_rank == 0) {
            cout << "\nKernel size (odd, e.g. 5): ";
            cin >> ksize;
            if (ksize % 2 == 0) ksize++;
        }
        MPI_Bcast(&ksize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        medianBlur(localImage, resultImage, ksize);
        gatherProcessedImages(resultImage, inputImage, world_rank, world_size);
    }

    else {
        if (world_rank == 0)
            cerr << "[ERROR] Invalid choice.\n";
        MPI_Finalize();
        return 1;
    }

    // ── Timing + save (rank 0 only) ──────────────────────────────────────────
    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    if (world_rank == 0) {
        imwrite(savedPath, inputImage);
        cout << "\n  Done in " << elapsed << " s  →  saved to " << savedPath << "\n";
        cout << "  Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    MPI_Finalize();
    return 0;
}

cmake_minimum_required(VERSION 3.16)
project(ParallelImageProcessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(MPI   REQUIRED)
find_package(OpenCV REQUIRED)

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(parallel_img src/main.cpp)

target_include_directories(parallel_img PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(parallel_img PRIVATE
    MPI::MPI_CXX
    ${OpenCV_LIBS}
)

# ── Compile flags ─────────────────────────────────────────────────────────────
target_compile_options(parallel_img PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

message(STATUS "MPI version   : ${MPI_CXX_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Directory where Eigen should be installed
EIGEN_DIR=/usr/local/include/eigen3
EIGEN_VER="5.0.1"
EIGEN_ARCHIVE="eigen-${EIGEN_VER}.zip"
EIGEN_SRC_DIR="eigen-${EIGEN_VER}"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/${EIGEN_ARCHIVE}"
MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 1)}"

# Check if Eigen is already installed
if [ ! -d "$EIGEN_DIR" ]; then
  # Download Eigen
  curl -OL "$EIGEN_URL"

  # Unzip Eigen
  unzip "$EIGEN_ARCHIVE"

  # Create build directory
  mkdir "$EIGEN_SRC_DIR/build"
  cd "$EIGEN_SRC_DIR/build"

  # Configure
  cmake ..

  # Install
  make -j"$MAKE_JOBS" install
else
  echo "Eigen is already installed at $EIGEN_DIR."
fi


# Get the Python interpreter path
PYTHON_PATH=$(which python)

# Install pybind11 using the correct Python interpreter
$PYTHON_PATH -m pip install pybind11

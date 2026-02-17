cmake_minimum_required(VERSION 3.16)
project(racplusplus VERSION "0.0.5")
message(STATUS "CC: $ENV{CC}")
message(STATUS "CXX: $ENV{CXX}")
option(RACPP_SYMDIST_USE_FLOAT "Store SymDistMatrix values as float32 instead of float64" ON)
option(RACPP_ENABLE_MARCH_NATIVE "Enable -march=native for host-specific CPU optimizations" ON)
option(RACPP_ENABLE_FAST_MATH "Enable -ffast-math (may change floating-point numerics)" ON)
option(RACPP_USE_BLAS_BACKEND "Use external BLAS backend for Eigen GEMM (OpenBLAS/MKL/BLIS)" OFF)
option(RACPP_SPLIT_STORE_NN "Split dense-init off-diagonal store and NN update into two passes" OFF)
option(RACPP_SIMD_DISSIM_TAIL_UPDATE "Enable AVX2/FMA SIMD tail fast path in dissim update loop" OFF)
option(RACPP_SIMD_NN_TAIL_UPDATE "Enable AVX2/FMA SIMD tail fast path in nn_dist tail scan" OFF)
option(RACPP_SIMD_TAIL_UPDATE "Legacy umbrella option: enable both SIMD tail paths" OFF)

# Set the OpenMP flags here
if(DEFINED ENV{OpenMP_LIBRARY})
  message(STATUS "OpenMP_LIBRARY = $ENV{OpenMP_LIBRARY}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
  include_directories("/usr/local/include")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "The C++ compiler is g++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Set Python3.11 directories
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${Python_SITEARCH})
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

message(STATUS "-- -- Found Python ${Python3_VERSION}")
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# Set compilers
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

#Set Opt, Warning Flags, Linker
message(STATUS "\n-- Setting C++ Warning and Optimization Flags, Setting Linker to lld...\n")

if (MSVC)
  message(STATUS "MSVC detected, setting flags...")
  add_compile_options(/openmp)

  # Get the Python version
  execute_process(COMMAND python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
  OUTPUT_VARIABLE PYTHON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")

  list(APPEND CMAKE_PREFIX_PATH "C:/hostedtoolcache/windows/Python/${PYTHON_VERSION}/x64/lib/site-packages/pybind11")
  add_compile_options(/O2 /W4 /EHsc)
  add_compile_options(/std:c++latest)

else()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wsign-compare -Wunreachable-code -fwrapv -Wall")
  if (RACPP_ENABLE_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    message(STATUS "Extra optimization: -march=native enabled")
  endif()
  if (RACPP_ENABLE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    message(STATUS "Extra optimization: -ffast-math enabled")
  endif()
  set(CMAKE_CXX_STANDARD 17)
endif()

#-------------------required packages-------------------------
find_package(Python3 REQUIRED Development.Module)
find_package(Python3 REQUIRED Interpreter)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
if (RACPP_USE_BLAS_BACKEND)
  find_package(BLAS REQUIRED)
endif()
#-------------------------------------------------------------

#-------------------------------------Include Directories----------------------------
#include python directory for Python.h header file
include_directories(${PYTHON311_INCLUDE_DIRS})
# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#include eigen directories
include_directories(${EIGEN3_INCLUDE_DIRS} )
#------------------------------------------------------------------------------------

#-----------------------------------Make Library---------------------------------------
set(python_module_name _racplusplus)
pybind11_add_module(${python_module_name} MODULE
     src/racplusplus/_racplusplus.cpp
    )

if (RACPP_SYMDIST_USE_FLOAT)
  target_compile_definitions(${python_module_name} PRIVATE RACPP_SYMDIST_USE_FLOAT=1)
  message(STATUS "SymDistMatrix storage precision: float32")
else()
  target_compile_definitions(${python_module_name} PRIVATE RACPP_SYMDIST_USE_FLOAT=0)
  message(STATUS "SymDistMatrix storage precision: float64")
endif()

if (RACPP_SPLIT_STORE_NN)
  target_compile_definitions(${python_module_name} PRIVATE RACPP_SPLIT_STORE_NN=1)
  message(STATUS "Dense-init off-diagonal store/NN split: enabled")
else()
  message(STATUS "Dense-init off-diagonal store/NN split: disabled")
endif()

set(_racpp_simd_dissim_tail OFF)
set(_racpp_simd_nn_tail OFF)
if (RACPP_SIMD_TAIL_UPDATE)
  set(_racpp_simd_dissim_tail ON)
  set(_racpp_simd_nn_tail ON)
endif()
if (RACPP_SIMD_DISSIM_TAIL_UPDATE)
  set(_racpp_simd_dissim_tail ON)
endif()
if (RACPP_SIMD_NN_TAIL_UPDATE)
  set(_racpp_simd_nn_tail ON)
endif()

if (_racpp_simd_dissim_tail)
  target_compile_definitions(${python_module_name} PRIVATE RACPP_SIMD_DISSIM_TAIL_UPDATE=1)
  message(STATUS "Dissim update SIMD tail fast path: enabled")
else()
  message(STATUS "Dissim update SIMD tail fast path: disabled")
endif()

if (_racpp_simd_nn_tail)
  target_compile_definitions(${python_module_name} PRIVATE RACPP_SIMD_NN_TAIL_UPDATE=1)
  message(STATUS "NN rescan SIMD tail fast path: enabled")
else()
  message(STATUS "NN rescan SIMD tail fast path: disabled")
endif()

if (RACPP_USE_BLAS_BACKEND)
  target_compile_definitions(${python_module_name} PRIVATE EIGEN_USE_BLAS=1)
  if (TARGET BLAS::BLAS)
    target_link_libraries(${python_module_name} PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(${python_module_name} PRIVATE ${BLAS_LIBRARIES})
  endif()
  message(STATUS "Eigen GEMM backend: external BLAS")
else()
  message(STATUS "Eigen GEMM backend: Eigen built-in")
endif()

#-----------------------------------Link Libraries-------------------------------------
if (MSVC)
  target_link_libraries(${python_module_name} PRIVATE
      -L${PYTHON311_LIBRARIES}
      ${pybind11_LIBRARIES}
      Eigen3::Eigen)
else()
  target_link_libraries(${python_module_name} PRIVATE
      -L${PYTHON311_LIBRARIES}
      ${pybind11_LIBRARIES}
      c++
      Eigen3::Eigen)
endif()
#----------------------------------------------------------------------------------------

#Install the Cmake Target in src directory specified in setup.py
install(TARGETS ${python_module_name}
        DESTINATION .)

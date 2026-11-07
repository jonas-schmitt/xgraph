cmake_minimum_required(VERSION 3.26...4.2)
project(xgraph LANGUAGES CXX)

# The GPU engine needs a CUDA compiler. Without one, xgraph builds as pure
# Python and the engine tests skip on machines that have no NVIDIA GPU.
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "No CUDA compiler found: building xgraph without the GPU engine")
  return()
endif()

# Portable code for Hopper, Blackwell datacenter, and Blackwell consumer and
# GB10 GPUs, plus PTX for newer ones (docs/cuda-kernels-via-nanobind-adr.md).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90-real 100-real 120-real 121)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_engine NB_STATIC csrc/engine.cu)
install(TARGETS _engine LIBRARY DESTINATION xgraph)

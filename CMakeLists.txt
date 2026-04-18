cmake_minimum_required(VERSION 3.25)
project(magenta_realtime_mlx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MRT_BUILD_TESTS "Build C++ unit tests"               ON)

# MLX is consumed from the project's `uv` venv, which ships libmlx.dylib,
# mlx.metallib, headers, and an MLXConfig.cmake. Caller may override
# MLX_ROOT to point at a different MLX install (e.g. a from-source build).
if(NOT DEFINED MLX_ROOT)
  file(GLOB _py_dirs "${CMAKE_SOURCE_DIR}/.venv/lib/python3.*")
  list(GET _py_dirs 0 _py_dir)
  set(MLX_ROOT "${_py_dir}/site-packages/mlx"
      CACHE PATH "Directory containing share/cmake/MLX/MLXConfig.cmake")
endif()

list(APPEND CMAKE_PREFIX_PATH "${MLX_ROOT}")
find_package(MLX CONFIG REQUIRED)

message(STATUS "MLX include dirs: ${MLX_INCLUDE_DIRS}")
message(STATUS "MLX libraries:    ${MLX_LIBRARIES}")

# RPATH used by every runtime artifact so they can find libmlx.dylib without
# DYLD_LIBRARY_PATH or `install_name_tool` gymnastics during development.
set(MRT_RPATH "${MLX_ROOT}/lib" CACHE INTERNAL "")

add_subdirectory(src/magenta_realtime_mlx)
add_subdirectory(cli)

if(MRT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

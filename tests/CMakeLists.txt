# Catch2 v3 (`brew install catch2`). Each test is a standalone executable
# linked against the core library; ctest discovers them automatically.
#
# This distribution ships only the playback path; the full parity test
# suite is not part of this release.

find_package(Catch2 3 REQUIRED)
include(Catch)

function(add_mrt_test NAME)
  set(_src "${ARGN}")
  if(NOT _src)
    set(_src "${NAME}.cpp")
  endif()
  add_executable(${NAME} ${_src})
  target_include_directories(${NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${NAME}
    PRIVATE
      magenta_realtime_mlx
      Catch2::Catch2WithMain
  )
  set_target_properties(${NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
    INSTALL_RPATH "${MRT_RPATH}"
    BUILD_WITH_INSTALL_RPATH ON
  )
  catch_discover_tests(${NAME})
endfunction()

add_mrt_test(test_dtype)

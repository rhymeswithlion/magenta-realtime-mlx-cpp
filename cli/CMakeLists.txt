# ``mlx-stream`` binary -- live MagentaRT music streaming on Apple Silicon.
# Links against the core magenta_realtime_mlx library which pulls in MLX,
# SentencePiece, and PortAudio transitively.

add_executable(mlx-stream main.cpp)
target_link_libraries(mlx-stream PRIVATE magenta_realtime_mlx)
set_target_properties(mlx-stream PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  INSTALL_RPATH "${MRT_RPATH}"
  BUILD_WITH_INSTALL_RPATH ON
)

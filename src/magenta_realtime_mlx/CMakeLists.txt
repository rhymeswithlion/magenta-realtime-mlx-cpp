# Core C++ library: every module that the CLI links against lives here.
# Keep this list alphabetised.

add_library(magenta_realtime_mlx STATIC
  audio.cpp
  depthformer/decode.cpp
  depthformer/model.cpp
  depthformer/modules.cpp
  dtype.cpp
  musiccoca.cpp
  playback.cpp
  rvq.cpp
  spectrostream.cpp
  style_model.cpp
  system.cpp
  weights.cpp
)

target_include_directories(magenta_realtime_mlx PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(magenta_realtime_mlx PUBLIC mlx)

# SentencePiece is needed by the style_model module (text tokenizer).
find_library(SENTENCEPIECE_LIBRARY sentencepiece HINTS /opt/homebrew/lib)
find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h
          HINTS /opt/homebrew/include)
if(NOT SENTENCEPIECE_LIBRARY OR NOT SENTENCEPIECE_INCLUDE_DIR)
  message(FATAL_ERROR
    "SentencePiece is required for the C++ runtime. "
    "Install with: brew install sentencepiece")
endif()
target_include_directories(magenta_realtime_mlx PUBLIC
  "${SENTENCEPIECE_INCLUDE_DIR}")
target_link_libraries(magenta_realtime_mlx PUBLIC "${SENTENCEPIECE_LIBRARY}")

# PortAudio is needed by the playback module (CLI-facing live output).
find_library(PORTAUDIO_LIBRARY portaudio HINTS /opt/homebrew/lib)
find_path(PORTAUDIO_INCLUDE_DIR portaudio.h HINTS /opt/homebrew/include)
if(NOT PORTAUDIO_LIBRARY OR NOT PORTAUDIO_INCLUDE_DIR)
  message(FATAL_ERROR
    "PortAudio is required for the C++ runtime. "
    "Install with: brew install portaudio")
endif()
target_include_directories(magenta_realtime_mlx PUBLIC
  "${PORTAUDIO_INCLUDE_DIR}")
target_link_libraries(magenta_realtime_mlx PUBLIC "${PORTAUDIO_LIBRARY}")

target_compile_features(magenta_realtime_mlx PUBLIC cxx_std_20)

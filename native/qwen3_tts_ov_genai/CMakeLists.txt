cmake_minimum_required(VERSION 3.20)
project(qwen3_tts_ov_genai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT OPENVINO_INCLUDE_DIR)
  message(FATAL_ERROR "OPENVINO_INCLUDE_DIR is required")
endif()
if(NOT OPENVINO_LIB)
  message(FATAL_ERROR "OPENVINO_LIB is required")
endif()
if(NOT OPENVINO_GENAI_INCLUDE_DIR)
  message(FATAL_ERROR "OPENVINO_GENAI_INCLUDE_DIR is required")
endif()
if(NOT OPENVINO_GENAI_LIB)
  message(FATAL_ERROR "OPENVINO_GENAI_LIB is required")
endif()

add_library(qwen3_tts_ov_genai SHARED qwen3_tts_codegen.cpp)
target_include_directories(
  qwen3_tts_ov_genai
  PRIVATE
    "${OPENVINO_INCLUDE_DIR}"
    "${OPENVINO_GENAI_INCLUDE_DIR}"
)
target_link_libraries(qwen3_tts_ov_genai PRIVATE "${OPENVINO_LIB}" "${OPENVINO_GENAI_LIB}")

if(MSVC)
  target_compile_options(qwen3_tts_ov_genai PRIVATE /O2 /DNDEBUG)
else()
  target_compile_options(qwen3_tts_ov_genai PRIVATE -O3 -DNDEBUG)
endif()

set(BUILD_NATIVE_CLI ON CACHE BOOL "Build native smoke CLI")
if(BUILD_NATIVE_CLI)
  add_executable(qwen3_tts_ov_native_cli qwen3_tts_cli.cpp)
  target_include_directories(
    qwen3_tts_ov_native_cli
    PRIVATE
      "${OPENVINO_INCLUDE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  if(OPENVINO_TOKENIZERS_PATH)
    set(OPENVINO_TOKENIZERS_PATH_CPP "${OPENVINO_TOKENIZERS_PATH}")
    file(TO_CMAKE_PATH "${OPENVINO_TOKENIZERS_PATH_CPP}" OPENVINO_TOKENIZERS_PATH_CPP)
    target_compile_definitions(
      qwen3_tts_ov_native_cli
      PRIVATE
        DEFAULT_OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH_CPP}"
    )
  endif()
  target_link_libraries(qwen3_tts_ov_native_cli PRIVATE qwen3_tts_ov_genai)
endif()

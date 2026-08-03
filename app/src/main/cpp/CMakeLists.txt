cmake_minimum_required(VERSION 3.22.1)

project(llamacpp_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp checkout")
set(LLAMA_CPP_REVISION "unknown" CACHE STRING "Pinned llama.cpp Git revision")
set(SPIRV_HEADERS_INCLUDE_DIR "" CACHE PATH "Path to SPIR-V Headers include directory")
set(VULKAN_HEADERS_INCLUDE_DIR "" CACHE PATH "Path to Vulkan-Headers include directory")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found. Clone it to third_party/llama.cpp or pass -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

if(GGML_VULKAN AND VULKAN_HEADERS_INCLUDE_DIR AND TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE "${VULKAN_HEADERS_INCLUDE_DIR}")
endif()

if(GGML_VULKAN AND SPIRV_HEADERS_INCLUDE_DIR AND TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE "${SPIRV_HEADERS_INCLUDE_DIR}")
endif()

add_library(llamacpp_jni SHARED llama_jni.cpp)

set(LLAMACPP_GEMMA_NATIVE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(NOT LLAMACPP_GEMMA_NATIVE_BUILD_TYPE)
    set(LLAMACPP_GEMMA_NATIVE_BUILD_TYPE "Unknown")
endif()
target_compile_definitions(
    llamacpp_jni
    PRIVATE
    "LLAMACPP_GEMMA_BUILD_TYPE=\"${LLAMACPP_GEMMA_NATIVE_BUILD_TYPE}\""
    "LLAMACPP_GEMMA_LLAMA_REVISION=\"${LLAMA_CPP_REVISION}\""
)

if(GGML_VULKAN)
    target_compile_definitions(llamacpp_jni PRIVATE LLAMACPP_GEMMA_VULKAN_ENABLED=1)
endif()

target_include_directories(llamacpp_jni PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
)

if(GGML_VULKAN AND VULKAN_HEADERS_INCLUDE_DIR)
    target_include_directories(llamacpp_jni PRIVATE "${VULKAN_HEADERS_INCLUDE_DIR}")
endif()

if(GGML_VULKAN AND SPIRV_HEADERS_INCLUDE_DIR)
    target_include_directories(llamacpp_jni PRIVATE "${SPIRV_HEADERS_INCLUDE_DIR}")
endif()

target_link_libraries(llamacpp_jni
    llama
    android
    log
)

cmake_minimum_required(VERSION 3.22.1)

project(llamacpp_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp checkout")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found. Clone it to third_party/llama.cpp or pass -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

add_library(llamacpp_jni SHARED llama_jni.cpp)

if(GGML_VULKAN)
    target_compile_definitions(llamacpp_jni PRIVATE LLAMACPP_GEMMA_VULKAN_ENABLED=1)
endif()

target_include_directories(llamacpp_jni PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
)

target_link_libraries(llamacpp_jni
    llama
    android
    log
)

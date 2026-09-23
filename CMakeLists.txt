cmake_minimum_required(VERSION 3.20)

project(vdb
    VERSION 0.1.0
    DESCRIPTION "Embedded vector database for RAG (segments + HNSW), C ABI-ready"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Per-compiler warning contract. MSVC and GCC/Clang disagree on flag syntax,
# so this cannot live in a single add_compile_options.
if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(cmake/simd.cmake)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)

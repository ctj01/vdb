add_library(vdb STATIC
    kernels/distance_scalar.cpp
    kernels/distance_avx2.cpp
    kernels/dispatch.cpp
    hnsw/builder.cpp
    hnsw/search.cpp
    io/file.cpp
    segment/crc32c.cpp
    segment/writer.cpp
    segment/reader.cpp
)

target_include_directories(vdb
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# The only TU allowed to contain AVX2 instructions (runtime-dispatched).
vdb_mark_avx2(kernels/distance_avx2.cpp)

# The static core gets linked into the shared C-ABI library.
set_target_properties(vdb PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C ABI shared library (the P/Invoke surface for .NET).
add_library(vdb_c SHARED capi/c_api.cpp)
target_link_libraries(vdb_c PRIVATE vdb)
target_compile_definitions(vdb_c PRIVATE VDB_C_BUILD)

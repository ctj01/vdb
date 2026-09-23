add_executable(vdb_bench bench_hnsw.cpp)
target_link_libraries(vdb_bench PRIVATE vdb)
target_include_directories(vdb_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

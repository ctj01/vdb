function(vdb_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE vdb)
    # Tests may reach into internal headers (kernels, search core).
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

vdb_test(test_kernels)
vdb_test(test_hnsw)
vdb_test(test_segment)

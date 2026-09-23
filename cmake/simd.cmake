# Marks a single translation unit as AVX2+FMA. Isolating the flags per-TU is
# load-bearing: applied globally, the compiler may emit AVX2 anywhere in the
# binary and crash (SIGILL) on older CPUs regardless of runtime dispatch.
# On non-x86 targets (macOS arm64) this is a no-op: the TU compiles its
# scalar-only preprocessor branch.
function(vdb_mark_avx2 source_file)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        if(MSVC)
            set_source_files_properties(${source_file} PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(${source_file} PROPERTIES
                COMPILE_OPTIONS "-mavx2;-mfma")
        endif()
    endif()
endfunction()

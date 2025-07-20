set(CHECKER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/check-vpdpbssd.c)

add_executable(check-vpdpbssd ${CHECKER_SRC})
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(check-vpdpbssd PRIVATE -mavxvnni -O2)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(check-vpdpbssd PRIVATE -mavx2 -mavxvnni -O2)
endif()

# Optional: ensure it outputs to a known build dir
set_target_properties(check-vpdpbssd PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Export the full path to parent scope
set(CHECKER_BIN ${CMAKE_BINARY_DIR}/bin/check-vpdpbssd PARENT_SCOPE)


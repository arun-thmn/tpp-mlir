set(CHECKER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/check-vpdpbssd.cpp)

add_executable(check-vpdpbssd ${CHECKER_SRC})
target_compile_options(check-vpdpbssd PRIVATE -mavxvnni -O2)

# Optional: ensure it outputs to a known build dir
set_target_properties(check-vpdpbssd PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Export the full path to parent scope
set(CHECKER_BIN ${CMAKE_BINARY_DIR}/bin/check-vpdpbssd PARENT_SCOPE)


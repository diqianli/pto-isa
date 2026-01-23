cmake_minimum_required(VERSION 3.15)

# Add example subdirectories
add_subdirectory(basic)

# Add Python example if Python bindings are enabled
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(basic_python)
endif()

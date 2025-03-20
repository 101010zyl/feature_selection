cmake_minimum_required(VERSION 3.14)
project(feature_selection VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Enable testing
include(CTest)
enable_testing()

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create feature selection library
add_library(feature_selection_lib
    src/data_loader.cpp
    # src/nearest_neighbor.cpp
    # src/feature_selection.cpp
)

# Set include directories for the library
target_include_directories(feature_selection_lib 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP to the library
target_link_libraries(feature_selection_lib
    PUBLIC
        OpenMP::OpenMP_CXX
)

# Create executable
add_executable(feature_selection src/main.cpp)

# Link library with executable
target_link_libraries(feature_selection
    PRIVATE
        feature_selection_lib
)

# Add tests subdirectory if testing is enabled
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS feature_selection
    RUNTIME DESTINATION bin
)

# Install library
install(TARGETS feature_selection_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
)
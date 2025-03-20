# Find required packages
find_package(GTest QUIET)

# If GTest is not found, use FetchContent to download it
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Create test executables
add_executable(test_data_loader test_data_loader.cpp)
target_link_libraries(test_data_loader
    PRIVATE
        feature_selection_lib
        GTest::gtest
        GTest::gtest_main
)

# add_executable(test_nearest_neighbor test_nearest_neighbor.cpp)
# target_link_libraries(test_nearest_neighbor
#     PRIVATE
#         feature_selection_lib
#         gtest
#         gtest_main
# )

# add_executable(test_feature_selection test_feature_selection.cpp)
# target_link_libraries(test_feature_selection
#     PRIVATE
#         feature_selection_lib
#         gtest
#         gtest_main
# )

# Add tests to CTest
add_test(NAME DataLoaderTests COMMAND test_data_loader)
add_test(NAME NearestNeighborTests COMMAND test_nearest_neighbor)
add_test(NAME FeatureSelectionTests COMMAND test_feature_selection)
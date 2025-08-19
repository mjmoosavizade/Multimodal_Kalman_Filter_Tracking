# Test CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

# Find Google Test (prefer static libraries)
find_package(GTest QUIET)

if(GTest_FOUND)
    # Only include existing test sources
    set(TEST_SOURCES
        test_kalman_filter.cpp
    )

    # Create test executable
    add_executable(run_tests ${TEST_SOURCES})

    # Check if static libraries are available
    if(EXISTS "/usr/local/lib/libgtest.a" AND EXISTS "/usr/local/lib/libgtest_main.a")
        # Use static libraries to avoid ABI issues
        target_link_libraries(run_tests
            PRIVATE
            tracking_lib
            /usr/local/lib/libgtest.a
            /usr/local/lib/libgtest_main.a
            ${OpenCV_LIBS}
            Eigen3::Eigen
            pthread
        )
        message(STATUS "Using static GTest libraries")
    else()
        # Fall back to dynamic libraries
        target_link_libraries(run_tests
            PRIVATE
            tracking_lib
            GTest::GTest
            GTest::Main
            ${OpenCV_LIBS}
            Eigen3::Eigen
        )
        message(STATUS "Using dynamic GTest libraries")
    endif()

    # Include directories
    target_include_directories(run_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )

    # Register tests
    include(GoogleTest)
    gtest_discover_tests(run_tests)

    message(STATUS "Tests enabled with Google Test")
else()
    message(STATUS "Google Test not found - tests disabled")
endif()

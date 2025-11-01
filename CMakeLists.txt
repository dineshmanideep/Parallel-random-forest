cmake_minimum_required(VERSION 3.16)
project(parallel-random-forest LANGUAGES CXX)

# --- Compiler setup ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Find Packages ---
find_package(OpenMP REQUIRED) # OpenMP

# --- Add include directory ---
include_directories(${PROJECT_SOURCE_DIR}/include)

# --- Collect all source files automatically (optional but convenient) ---
# Collect all source files automatically and add them to the executable
file(GLOB SRC_FILES
    src/*.cpp
)

# --- Add executable (custom name) ---
set(EXECUTABLE_NAME forests)
add_executable(${EXECUTABLE_NAME}
    ${SRC_FILES}
)

# --- Link OpenMP ---
target_link_libraries(${EXECUTABLE_NAME}
    PRIVATE OpenMP::OpenMP_CXX
)

# --- Optional: Compiler warnings ---
target_compile_options(${EXECUTABLE_NAME}
    PRIVATE -Wall -Wextra -Wpedantic
)

# --- Optional: Set output directory ---
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

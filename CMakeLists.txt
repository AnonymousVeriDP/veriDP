cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(VeriDP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Allow setting libtorch path via environment variable or CMake variable
# Usage: cmake -DTORCH_ROOT=/path/to/libtorch ..
# Or: export TORCH_ROOT=/path/to/libtorch && cmake ..
if(DEFINED ENV{TORCH_ROOT})
    set(CMAKE_PREFIX_PATH "$ENV{TORCH_ROOT}")
elseif(DEFINED TORCH_ROOT)
    set(CMAKE_PREFIX_PATH "${TORCH_ROOT}")
else()
    # Default to relative path (libtorch in project directory) or system path
    if(EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
        set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
    else()
        # If libtorch not found, CMake will use system paths
        message(STATUS "TORCH_ROOT not set. Checking system paths...")
    endif()
endif()

find_package(Torch REQUIRED)

# LibTorch include directories - ensure ATen is accessible
# LibTorch headers use <ATen/Tensor.h> which should resolve to ATen/core/Tensor.h
# The issue is that LibTorch expects ATen/ to be in the include path
if(EXISTS "${CMAKE_SOURCE_DIR}/libtorch/include")
    get_filename_component(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/libtorch" ABSOLUTE)
    # Add the include directory so <ATen/...> resolves correctly
    include_directories("${LIBTORCH_ROOT}/include")
endif()

# Use TORCH_INCLUDE_DIRS if provided by find_package
if(TORCH_INCLUDE_DIRS)
    include_directories("${TORCH_INCLUDE_DIRS}")
endif()

# Print debug info
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
if(EXISTS "${CMAKE_SOURCE_DIR}/libtorch/include/ATen/core/Tensor.h")
    message(STATUS "Found ATen/core/Tensor.h")
else()
    message(WARNING "ATen/core/Tensor.h not found - LibTorch may be incomplete")
endif()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/Summer code")
# For files that include with "Summer code/" prefix
add_compile_definitions(SUMMER_CODE_DIR="${CMAKE_SOURCE_DIR}/Summer code")

# Option to use 256-bit field (BLS12-381 scalar field)
option(USE_FIELD_256 "Use 256-bit prime field (BLS12-381) instead of 61-bit Mersenne" OFF)

if(USE_FIELD_256)
    message(STATUS "Building with 256-bit BLS12-381 scalar field using mcl library")
    
    # MCL library paths
    set(MCL_ROOT "${CMAKE_SOURCE_DIR}/mcl")
    if(EXISTS "${MCL_ROOT}")
        include_directories("${MCL_ROOT}/include")
        link_directories("${MCL_ROOT}/lib")
        message(STATUS "MCL library found at ${MCL_ROOT}")
    else()
        message(FATAL_ERROR "MCL library not found. Please run: git clone https://github.com/herumi/mcl.git && cd mcl && make")
    endif()
else()
    message(STATUS "Building with 61-bit Mersenne prime field")
endif()

# Get Summer code source files (excluding main.cpp and proof_utils.cpp)
# Only include files needed for VeriDP (exclude RNN-specific files)
# Core infrastructure:
#   - fieldElement: field arithmetic
#   - utils: utilities
#   - quantization: quantization functions
#   - field_arithmetic: advanced field operations
#   - flo-shani: SHA-NI hash implementation
#   - mimc: MIMC hash
#   - merkle_tree: Merkle tree operations
#   - polynomial: polynomial operations
#   - poly_commit: polynomial commitments (needs expanders, linear_code_encode)
#   - GKR: GKR protocol
#   - proof_utils: proof utilities
#   - expanders: expander graphs (needed by poly_commit)
#   - linear_code_encode: linear code encoding (needed by poly_commit)
# VeriDP-specific:
#   - ivc_adapter: IVC adapter for VeriDP
#   - finalise_proof: finalize proofs
#   - verifier: verifier
#   - proof_serialization: proof serialization
# NOTE: prove_leaf.h is included for LeafResult struct, but prove_leaf.cpp is RNN-specific
#       and excluded. If prove_leaf.cpp is needed, it should be stubbed for VeriDP.
set(SUMMER_SRC
    # Core infrastructure - field arithmetic
    "${CMAKE_SOURCE_DIR}/Summer code/fieldElement.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/fieldElement256.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/utils.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/quantization.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/field_arithmetic.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/flo-shani.c"
    "${CMAKE_SOURCE_DIR}/Summer code/mimc.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/merkle_tree.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/polynomial.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/expander.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/linear_code_encode.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/poly_commit.cpp"
    # Proof system
    "${CMAKE_SOURCE_DIR}/Summer code/GKR.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/proof_utils.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/pol_verifier.cpp"
    # Infrastructure needed by proof system (even if RNN-specific, needed for linking)
    "${CMAKE_SOURCE_DIR}/Summer code/circuit.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/prover.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/timer.cpp"
    # VeriDP-specific
    "${CMAKE_SOURCE_DIR}/Summer code/ivc_adapter.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/finalise_proof.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/verifier.cpp"
    "${CMAKE_SOURCE_DIR}/Summer code/proof_serialization.cpp"
    # Dataset commitment
    "${CMAKE_SOURCE_DIR}/dataset_commitment.cpp"
)

# Original training executable (without proofs)
add_executable(train train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")
set_property(TARGET train PROPERTY CXX_STANDARD 17)
# Enable AVX2 if fieldElement is used (though train.cpp might not need it)
target_compile_options(train PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
)

# VeriDP training executable (with proofs)
add_executable(train_veridp 
    train_veridp.cpp
    ${SUMMER_SRC}
)

# Ensure flo-shani.c is compiled as C (not C++)
set_source_files_properties("${CMAKE_SOURCE_DIR}/Summer code/flo-shani.c" 
    PROPERTIES LANGUAGE C
)

# Add SHA-NI and SSSE3 support for flo-shani.c (required for SHA intrinsics)
# SHA-NI intrinsics require -msha, and _mm_shuffle_epi8 requires -mssse3
set_source_files_properties("${CMAKE_SOURCE_DIR}/Summer code/flo-shani.c" 
    PROPERTIES COMPILE_FLAGS "-msha -mssse3"
)
# Link against torch - use target which handles all includes automatically
if(TARGET torch)
    # Use the torch target - it should handle all include paths
    target_link_libraries(train_veridp 
        torch
        OpenMP::OpenMP_CXX 
        Threads::Threads
    )
else()
    # Fallback: use TORCH_LIBRARIES and manually set includes
    target_link_libraries(train_veridp 
        "${TORCH_LIBRARIES}"
        OpenMP::OpenMP_CXX 
        Threads::Threads
    )
    # Add include directories
    if(TORCH_INCLUDE_DIRS)
        target_include_directories(train_veridp PRIVATE "${TORCH_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/libtorch/include")
        target_include_directories(train_veridp PRIVATE "${CMAKE_SOURCE_DIR}/libtorch/include")
    endif()
endif()

# Add compiler flags that LibTorch might need
# Enable AVX2 for fieldElement.cpp which uses AVX2 intrinsics (not just AVX)
target_compile_options(train_veridp PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
)
# Note: Add gmp and XKCP if available
# target_link_libraries(train_veridp gmp XKCP)
set_property(TARGET train_veridp PROPERTY CXX_STANDARD 17)
target_compile_definitions(train_veridp PRIVATE DEBUG_LOGUP=1)

# Add 256-bit field compile definition and link mcl library if enabled
if(USE_FIELD_256)
    target_compile_definitions(train_veridp PRIVATE USE_FIELD_256=1)
    # Link mcl library for BLS12-381 field arithmetic
    target_link_libraries(train_veridp mcl)
endif()
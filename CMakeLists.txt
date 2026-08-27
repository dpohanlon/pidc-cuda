cmake_minimum_required(VERSION 3.24)

project(pidc_cuda VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PIDC_PORTABLE_BUILD
       "Avoid host-specific CPU instructions in distributable wheels" ON)
set(PIDC_CUDA_ARCHITECTURES
    "75-real;80-real;86-real;89-real;90-real;100-real;103-real;110-real;120"
    CACHE STRING
    "CUDA cubins to embed; the final unsuffixed architecture also embeds PTX")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)
find_package(Boost 1.66 REQUIRED)
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_library(pidc_cuda_backend STATIC cpp/pidc_cuda_backend.cu)
set_target_properties(
    pidc_cuda_backend
    PROPERTIES
        CUDA_ARCHITECTURES "${PIDC_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Static
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
target_include_directories(pidc_cuda_backend PUBLIC cpp)
target_compile_options(
    pidc_cuda_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(pidc_cuda_backend PUBLIC CUDA::cudart_static)

pybind11_add_module(_core MODULE cpp/bindings.cpp)
target_include_directories(_core PRIVATE cpp)
target_include_directories(_core SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(
    _core PRIVATE
    Eigen3::Eigen
    pidc_cuda_backend)
target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()
if(NOT PIDC_PORTABLE_BUILD)
    target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

install(TARGETS _core LIBRARY DESTINATION pidc_cuda)

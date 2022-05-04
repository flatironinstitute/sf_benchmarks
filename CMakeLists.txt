cmake_minimum_required(VERSION 3.14)
project(
  sf_benchmarks
  LANGUAGES C CXX Fortran
  )
include(ExternalProject)
set(CMAKE_CXX_STANDARD 17)

find_package(GSL REQUIRED)
find_package(Boost)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")

set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "No build type specified. Setting build type to Release.")
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Valid options: Debug, RelWithDebInfo, Release" FORCE)
endif()

set(
  SF_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
)

ExternalProject_Add(
  libsleef
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/sleef
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
)

ExternalProject_Add(libbaobzi
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/baobzi
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib -DBAOBZI_BUILD_TESTS=OFF
             -DBAOBZI_BUILD_SHARED=OFF -DBAOBZI_BUILD_EXAMPLES=OFF -DBAOBZI_BUILD_STATIC=ON
)

include_directories(
  ${CMAKE_BINARY_DIR}/contrib/include
  ${PROJECT_SOURCE_DIR}/extern/baobzi/extern/msgpack-c/include
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/toml11
  ${PROJECT_SOURCE_DIR}/extern/vectorclass2
)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib64 ${PROJECT_SOURCE_DIR}/extern/amd-libm/lib)

file(GLOB SF_SOURCES "src/main.cpp" "src/hank103.f" "src/bessel.f")
add_executable(sf_benchmarks ${SF_SOURCES})
target_include_directories(sf_benchmarks PRIVATE ${SF_INCLUDES} ${GSL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(sf_benchmarks sleef gsl baobzi dl)
add_dependencies(sf_benchmarks libsleef libbaobzi)
target_compile_options(sf_benchmarks PRIVATE -march=native)

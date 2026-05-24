cmake_minimum_required(VERSION 3.16)
project(merge_sort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_library(merge_sort
  src/sort_serial.cpp
  src/sort_parallel.cpp
  src/sort_parallel_rank.cpp
)
target_include_directories(merge_sort PUBLIC include)
target_link_libraries(merge_sort PUBLIC OpenMP::OpenMP_CXX)

add_executable(merge_sort_tests tests/test_merge_sort.cpp)
target_link_libraries(merge_sort_tests PRIVATE merge_sort)

enable_testing()
add_test(
  NAME merge_sort_tests
  COMMAND merge_sort_tests
)
set_tests_properties(
  merge_sort_tests
  PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=12;OMP_DYNAMIC=false"
)

cmake_minimum_required(VERSION 3.20)

project(JsonManager LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(JsonManager src/main.cpp)

target_include_directories(JsonManager PRIVATE include)


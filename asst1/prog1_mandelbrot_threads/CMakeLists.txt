cmake_minimum_required(VERSION 3.3)
project(prog1_mandelbrot_threads)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    main.cpp
    Makefile
    mandelbrotSerial.cpp
    mandelbrotThread.cpp)

add_executable(prog1_mandelbrot_threads ${SOURCE_FILES})
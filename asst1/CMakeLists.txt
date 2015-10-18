cmake_minimum_required(VERSION 3.3)
project(asst1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    common/CycleTimer.h
    common/ppm.cpp
    common/tasksys.cpp
    prog1_mandelbrot_threads/main.cpp
    prog1_mandelbrot_threads/Makefile
    prog1_mandelbrot_threads/mandelbrotSerial.cpp
    prog1_mandelbrot_threads/mandelbrotThread.cpp
    prog2_vecintrin/CMU418intrin.cpp
    prog2_vecintrin/CMU418intrin.h
    prog2_vecintrin/logger.cpp
    prog2_vecintrin/logger.h
    prog2_vecintrin/main.cpp
    prog2_vecintrin/Makefile
    prog3_mandelbrot_ispc/main.cpp
    prog3_mandelbrot_ispc/Makefile
    prog3_mandelbrot_ispc/mandelbrotSerial.cpp
    prog4_sqrt/main.cpp
    prog4_sqrt/Makefile
    prog4_sqrt/sqrtSerial.cpp
    prog5_saxpy/main.cpp
    prog5_saxpy/Makefile
    prog5_saxpy/saxpySerial.cpp)

add_executable(asst1 ${SOURCE_FILES})
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu_task_01 LANGUAGES CXX CUDA)

include(CTest)

add_executable(main
    # stb_image.h
    # stb_image_write.h
    main.cu
)

set_property(TARGET main 
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET main 
    PROPERTY CXX_STANDARD 17)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET main 
        PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

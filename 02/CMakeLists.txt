cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu_task_02 LANGUAGES CXX CUDA)

include(CTest)

add_executable(denoiser
    stb_image.h
    stb_image_write.h
    util.h
    convolution.h
    convolution.cu
    modules.h
    modules.cu
    denoiser.cu
)

set_property(TARGET denoiser 
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET denoiser 
    PROPERTY CXX_STANDARD 17)

add_custom_command(TARGET denoiser POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/model/ $<TARGET_FILE_DIR:denoiser>/model/)

# find_package(OpenMP REQUIRED)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(denoiser PUBLIC OpenMP::OpenMP_CXX)
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /openmp")
# endif()

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET denoiser 
        PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

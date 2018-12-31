#
#
#
#  Created by David Chiu
#  Dec. 28th, 2018 based on tencent's original file
#
#
project(Imageutil)
cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)

#get_filename_component(TF_SRC_ROOT ${CMAKE_SOURCE_DIR}/../../../..  ABSOLUTE)
get_filename_component(IMGUTIL_SRC_DIR  ${CMAKE_SOURCE_DIR}  ABSOLUTE)
get_filename_component (NCNN_JNI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ncnn  ABSOLUTE)

if (ANDROID_ABI MATCHES "^armeabi-v7a$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=softfp -mfpu=neon -fopenmp")
elseif(ANDROID_ABI MATCHES "^arm64-v8a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ftree-vectorize -fopenmp")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTANDALONE_DEMO_LIB \
                    -std=c++11 -fno-exceptions -fno-rtti -O2 -Wno-narrowing \
                    -fPIE")

add_definitions(-DHEADER_FILE=${HEADER_FILE})
add_definitions(-DNAMESPACE=${NAMESPACE})
add_definitions(-DNCNN_INPUT=${NCNN_INPUT})
add_definitions(-DNCNN_OUTPUT=${NCNN_OUTPUT})

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
                              -Wl,--allow-multiple-definition \
                              -Wl,--whole-archive -fPIE -v -fopenmp")

file(GLOB_RECURSE imgutil_sources ${IMGUTIL_SRC_DIR}/*.cc)
file(GLOB ncnn_jni_src ${NCNN_JNI_SOURCE_DIR}/*.cpp)

add_library(imgutil SHARED
            ${imgutil_sources})
target_include_directories(imgutil PRIVATE
                           ${CMAKE_SOURCE_DIR}
			   ${NCNN_JNI_SOURCE_DIR}/include
                           )

target_link_libraries(imgutil
                      android
                      log
                      jnigraphics
                      m
                      atomic
                      z)

message("dir: cmake_src: ${CMAKE_SOURCE_DIR} imgutil_src: ${IMGUTIL_SRC_DIR} ncnn_jni src: ${NCNN_JNI_SOURCE_DIR} ANDROID_ABI: ${ANDROID_ABI}")
find_library(ncnnlib ncnn HINTS ${NCNN_JNI_SOURCE_DIR}/libs/${ANDROID_ABI})
add_library(ncnnlib STATIC IMPORTED)
set_target_properties(ncnnlib PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn/libs/${ANDROID_ABI}/libncnn.a)

add_library(ncnn_jni SHARED
            ${ncnn_jni_src})
target_include_directories(ncnn_jni PRIVATE
                           ${CMAKE_SOURCE_DIR}/ncnn/include
                           )
target_link_libraries(ncnn_jni
                      ncnnlib
                      android
                      log
                      jnigraphics
                      m
                      atomic
                      z)

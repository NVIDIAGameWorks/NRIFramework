cmake_minimum_required(VERSION 3.15)

include(FetchContent)

# Is submodule?
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(IS_SUBMODULE OFF)
else()
    set(IS_SUBMODULE ON)
endif()

# Cached
if(NOT IS_SUBMODULE)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")
endif()

set(NRIF_SHADERS_PATH "" CACHE STRING "Shader output path override")

# Create project
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Include/NRIFramework.h" ver_h)
string(REGEX MATCH "NRI_FRAMEWORK_VERSION_MAJOR ([0-9]*)" _ ${ver_h})
set(VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "NRI_FRAMEWORK_VERSION_MINOR ([0-9]*)" _ ${ver_h})
set(VERSION_MINOR ${CMAKE_MATCH_1})

message("NRI framework v${VERSION_MAJOR}.${VERSION_MINOR}")
project(NRIFramework LANGUAGES C CXX)

# Globals?
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS -msse4.1 -Wextra)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILE_OPTIONS -msse4.1 -Wextra)
elseif(MSVC)
    set(COMPILE_OPTIONS /W4 /WX /wd4324)
else()
    message(WARNING "Unknown compiler!")
endif()

# Compile definitions
if(WIN32)
    set(COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# Download dependencies
set(DEPS)

# ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG master
    GIT_SHALLOW 1
)
list(APPEND DEPS imgui)

# GLFW
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
option(GLFW_INSTALL "" OFF)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG master
    GIT_SHALLOW 1
)
list(APPEND DEPS glfw)

# cgltf
FetchContent_Declare(
    cgltf
    GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
    GIT_TAG master
    GIT_SHALLOW 1
)
list(APPEND DEPS cgltf)

# ShaderMake
if(NOT TARGET ShaderMake)
    set(SHADERMAKE_BIN_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} CACHE STRING "")

    FetchContent_Declare(
        shadermake
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/ShaderMake.git
        GIT_TAG main
        GIT_SHALLOW 1
    )
    list(APPEND DEPS shadermake)
endif()

if(DEPS)
    message("NRI framework: downloading dependencies:")
    message(STATUS "${DEPS} ...")
    FetchContent_MakeAvailable(${DEPS})
endif()

# External/Imgui
file(GLOB IMGUI_SOURCE "${imgui_SOURCE_DIR}/*.cpp" "${imgui_SOURCE_DIR}/*.h")
source_group("" FILES ${IMGUI_SOURCE})
add_library(imgui STATIC ${IMGUI_SOURCE})
target_compile_definitions(imgui PRIVATE ${COMPILE_DEFINITIONS})
target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}")
set_property(TARGET imgui PROPERTY FOLDER "${PROJECT_NAME}")

# External/Detex
file(GLOB DETEX_SOURCE "External/Detex/*.c" "External/Detex/*.h")
source_group("" FILES ${DETEX_SOURCE})
add_library(detex STATIC ${DETEX_SOURCE})
target_compile_definitions(detex PRIVATE ${COMPILE_DEFINITIONS})
set_property(TARGET detex PROPERTY FOLDER "${PROJECT_NAME}")

# External/NRI
option(NRI_ENABLE_AGILITY_SDK_SUPPORT "" OFF)
option(NRI_ENABLE_NVTX_SUPPORT "" ON)
add_subdirectory("External/NRI")

# External/MathLib
add_subdirectory("External/MathLib")

# External/MetalUtility
if(APPLE)
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(QUARTZ_CORE QuartzCore REQUIRED)
    add_library(MetalUtility SHARED "External/MetalUtility/MetalUtility.m")
    target_link_libraries(MetalUtility
        ${QUARTZ_CORE}
        ${COCOA_LIBRARY}
        glfw
    )
endif()

# NRIFramework
if(NRI_ENABLE_AGILITY_SDK_SUPPORT)
    set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} NRI_ENABLE_AGILITY_SDK_SUPPORT)
endif()

file(GLOB FRAMEWORK_SOURCE "Source/*.cpp" "Source/*.h")
source_group("Source" FILES ${FRAMEWORK_SOURCE})
file(GLOB FRAMEWORK_HEADERS "Include/*.hpp" "Include/*.h" "${cgltf_SOURCE_DIR}/cgltf.h")
source_group("Include" FILES ${FRAMEWORK_HEADERS})
add_library(${PROJECT_NAME} STATIC ${FRAMEWORK_SOURCE} ${FRAMEWORK_HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE
    detex
    glfw
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    NRI
    imgui
    MathLib
)

if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_DL_LIBS}
        pthread
        MetalUtility
    )
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CMAKE_DL_LIBS}
        pthread
        X11
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "Source"
    "External"
    "${cgltf_SOURCE_DIR}"
)
target_include_directories(${PROJECT_NAME} PUBLIC
    "Include"
    "${glfw_SOURCE_DIR}/include"
)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEFINITIONS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "${PROJECT_NAME}")

# Shaders output directory
if("${NRIF_SHADERS_PATH}" STREQUAL "")
    set(NRIF_SHADERS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")
endif()

message("NRI framework shaders output path: '${NRIF_SHADERS_PATH}'")

# Shaders
file(GLOB_RECURSE SHADERS "Shaders/*.hlsl")
set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

set(SHADERMAKE_GENERAL_ARGS
    --useAPI --binary --flatten --stripReflection --WX
    --sourceDir "Shaders"
    -c Shaders.cfg
    -o "${NRIF_SHADERS_PATH}"
    -I "External/NRI/Include"
    -I "External/MathLib"
    --sRegShift 100 --tRegShift 200 --bRegShift 300 --uRegShift 400
)

if(WIN32)
    add_custom_target(${PROJECT_NAME}_Shaders ALL
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p DXBC --compiler "${FXC_PATH}"
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p DXIL --compiler "${DXC_PATH}"
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}"
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
else()
    add_custom_target(${PROJECT_NAME}_Shaders ALL
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}"
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
endif()

set_property(TARGET ${PROJECT_NAME}_Shaders PROPERTY FOLDER "${PROJECT_NAME}")
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_Shaders)

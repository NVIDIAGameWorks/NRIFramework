cmake_minimum_required (VERSION 3.15)

# Is submodule?
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(IS_SUBMODULE OFF)
else()
    set(IS_SUBMODULE ON)
endif()

# Cached
if (NOT IS_SUBMODULE)
    set (CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")
endif()

set (NRIF_SHADERS_PATH "" CACHE STRING "Shader output path override")
set (GLOBAL_BIN_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_Build" CACHE STRING "")

# Create project
file (READ "${CMAKE_CURRENT_SOURCE_DIR}/Include/NRIFramework.h" ver_h)
string (REGEX MATCH "NRI_FRAMEWORK_VERSION_MAJOR ([0-9]*)" _ ${ver_h})
set (VERSION_MAJOR ${CMAKE_MATCH_1})
string (REGEX MATCH "NRI_FRAMEWORK_VERSION_MINOR ([0-9]*)" _ ${ver_h})
set (VERSION_MINOR ${CMAKE_MATCH_1})

message ("NRIFramework v${VERSION_MAJOR}.${VERSION_MINOR}")
project (NRIFramework LANGUAGES C CXX)

# Download dependencies using Packman
if (WIN32)
    set (PACKMAN_EXT ".cmd")
else ()
    set (PACKMAN_EXT ".sh")
endif ()

if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    if (WIN32)
        set (PACKMAN_PLATFORM "windows-x86_64")
    else ()
        set (PACKMAN_PLATFORM "linux-x86_64")
    endif ()
elseif ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    set (PACKMAN_PLATFORM "linux-aarch64")
else ()
    message (FATAL_ERROR "Unsupported architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif ()

message ("Packman platform: ${PACKMAN_PLATFORM}")

execute_process (COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/packman${PACKMAN_EXT}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -p ${PACKMAN_PLATFORM} -t nri_data_version=${NRI_DATA_VERSION}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE PACKMAN_RESULT)

if (NOT "${PACKMAN_RESULT}" STREQUAL "0")
    message (FATAL_ERROR "Packman failed(code = ${PACKMAN_RESULT})")
endif ()

# Setup dependencies
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set (BIN_ARCHITECTURE "x86_64")
else ()
    set (BIN_ARCHITECTURE "x86")
endif ()

if (WIN32)
    find_library (GLFW_RELEASE_LIB NAMES glfw3 PATHS "External/Glfw/build-${BIN_ARCHITECTURE}/Release/src" REQUIRED)
    find_library (GLFW_DEBUG_LIB NAMES glfw3 PATHS "External/Glfw/build-${BIN_ARCHITECTURE}/Debug/src" REQUIRED)
    find_library (ASSIMP_LIB NAMES assimp-vc142-mt PATHS "External/Assimp/lib/${BIN_ARCHITECTURE}" REQUIRED)
    find_file (INPUT_ASSIMP_SHARED_LIB NAMES assimp-vc142-mt.dll PATHS "External/Assimp/bin/${BIN_ARCHITECTURE}" REQUIRED)
else ()
    find_library (GLFW_RELEASE_LIB NAMES glfw3 PATHS "External/Glfw/build/Release/src" REQUIRED)
    find_library (GLFW_DEBUG_LIB NAMES glfw3 PATHS "External/Glfw/build/Debug/src" REQUIRED)
    find_library (INPUT_ASSIMP_SHARED_LIB NAMES assimp PATHS "External/Assimp/bin" REQUIRED)
    set (ASSIMP_LIB ${INPUT_ASSIMP_SHARED_LIB})

    if (UNIX)
        find_path (XLIB_HEADERS NAMES "X11/Xlib.h")

        if (NOT XLIB_HEADERS)
            message (FATAL_ERROR "Can't find 'X11/xlib.h'.(libx11-dev is not installed?)")
        endif ()

        find_path (XRANDR_HEADERS NAMES "X11/extensions/Xrandr.h")

        if (NOT XRANDR_HEADERS)
            message (FATAL_ERROR "Can't find 'X11/extensions/Xrandr.h'.(libxrandr-dev is not installed?)")
        endif ()
    endif ()
endif ()

set (GLFW_LIB optimized ${GLFW_RELEASE_LIB} debug ${GLFW_DEBUG_LIB})

# Globals?
set_property (GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

# Compile options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set (COMPILE_OPTIONS -msse4.1 -Wextra)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set (COMPILE_OPTIONS -msse4.1 -Wextra)
elseif (MSVC)
    set (COMPILE_OPTIONS /W4 /WX /wd4324)
else ()
    message (WARNING "Unknown compiler!")
endif ()

# Compile definitions
if (WIN32)
    set (COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE GLFW_EXPOSE_NATIVE_WIN32 GLFW_INCLUDE_NONE)
else ()
    set (COMPILE_DEFINITIONS GLFW_EXPOSE_NATIVE_X11 GLFW_INCLUDE_NONE)
endif ()

# External/ImGui
file (GLOB IMGUI_SOURCE "External/ImGui/*.cpp" "External/ImGui/*.h")
source_group ("" FILES ${IMGUI_SOURCE})
add_library (ImGui STATIC ${IMGUI_SOURCE})
target_compile_definitions (ImGui PRIVATE ${COMPILE_DEFINITIONS})
set_property (TARGET ImGui PROPERTY FOLDER "${PROJECT_NAME}/External")

# External/Detex
file (GLOB DETEX_SOURCE "External/Detex/*.c" "External/Detex/*.h")
source_group ("" FILES ${DETEX_SOURCE})
add_library (Detex STATIC ${DETEX_SOURCE})
target_compile_definitions (Detex PRIVATE ${COMPILE_DEFINITIONS})
set_property (TARGET Detex PROPERTY FOLDER "${PROJECT_NAME}/External")

# External/MathLib
file (READ "${CMAKE_CURRENT_SOURCE_DIR}/External/MathLib/MathLib.h" ver_h)
string (REGEX MATCH "MATHLIB_VERSION_MAJOR ([0-9]*)" _ ${ver_h})
set (MATHLIB_VERSION_MAJOR ${CMAKE_MATCH_1})
string (REGEX MATCH "MATHLIB_VERSION_MINOR ([0-9]*)" _ ${ver_h})
set (MATHLIB_VERSION_MINOR ${CMAKE_MATCH_1})
message ("MathLib v${MATHLIB_VERSION_MAJOR}.${MATHLIB_VERSION_MINOR}")

file (GLOB MATHLIB_HEADERS "External/MathLib/*.h" "External/MathLib/*.hpp")
source_group ("MathLib" FILES ${MATHLIB_HEADERS})

# External/NRI
add_subdirectory ("External/NRI")

# NRIFramework
file (GLOB FRAMEWORK_SOURCE "Source/*.cpp" "Source/*.h")
source_group ("Source" FILES ${FRAMEWORK_SOURCE})
file (GLOB FRAMEWORK_HEADERS "Include/*.hpp" "Include/*.h")
source_group ("Include" FILES ${FRAMEWORK_HEADERS})
add_library (${PROJECT_NAME} STATIC ${FRAMEWORK_SOURCE} ${FRAMEWORK_HEADERS} ${MATHLIB_HEADERS})
target_link_libraries (${PROJECT_NAME} PRIVATE ImGui Detex ${ASSIMP_LIB} ${GLFW_LIB})

if (UNIX)
    target_link_libraries (${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS} pthread X11)
endif ()

target_include_directories (${PROJECT_NAME} PRIVATE "Source" "Include" "External" "External/NRI/Include" "External/Assimp/include")
target_compile_definitions (${PROJECT_NAME} PRIVATE ${COMPILE_DEFINITIONS})
target_compile_options (${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
set_property (TARGET ${PROJECT_NAME} PROPERTY FOLDER ${PROJECT_NAME})

get_filename_component (INPUT_ASSIMP_SHARED_LIB_NAME ${INPUT_ASSIMP_SHARED_LIB} NAME)
set (OUTPUT_ASSIMP_SHARED_LIB ${GLOBAL_BIN_OUTPUT_PATH}/$<CONFIG>/${INPUT_ASSIMP_SHARED_LIB_NAME})
add_custom_command (TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INPUT_ASSIMP_SHARED_LIB} ${OUTPUT_ASSIMP_SHARED_LIB}
    COMMAND_EXPAND_LISTS)

# External/ShaderMake
if (NOT TARGET ShaderMake)
    set (SHADERMAKE_BIN_OUTPUT_PATH ${GLOBAL_BIN_OUTPUT_PATH} CACHE STRING "")
    add_subdirectory (External/ShaderMake)
endif ()

# Shaders output directory
if ("${NRIF_SHADERS_PATH}" STREQUAL "")
    set (NRIF_SHADERS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")
endif ()

message ("NRI framework shaders output path: '${NRIF_SHADERS_PATH}'")

# Shaders
file (GLOB_RECURSE SHADERS "Shaders/*.hlsl")
set_source_files_properties (${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

if (WIN32)
    add_custom_target (${PROJECT_NAME}_Shaders ALL
        COMMAND ShaderMake --useAPI
            --binary --flatten --stripReflection --compiler "${DXC_PATH}"
            --sourceDir "Shaders"
            -p DXIL --WX
            -c Shaders.cfg
            -o "${NRIF_SHADERS_PATH}"
            -I "Shaders"
            -D COMPILER_DXC
        COMMAND ShaderMake --useAPI
            --binary --flatten --stripReflection --compiler "${DXC_SPIRV_PATH}"
            --sourceDir "Shaders"
            -p SPIRV --WX
            -c Shaders.cfg
            -o "${NRIF_SHADERS_PATH}"
            -I "Shaders"
            -D COMPILER_DXC
            -D VULKAN
        COMMAND ShaderMake --useAPI
            --binary --flatten --stripReflection --compiler "${FXC_PATH}"
            --sourceDir "Shaders"
            -p DXBC --WX
            -c Shaders.cfg
            -o "${NRIF_SHADERS_PATH}"
            -I "Shaders"
            -D COMPILER_FXC
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
else ()
    add_custom_target (${PROJECT_NAME}_Shaders ALL
        COMMAND ShaderMake
            --binary --flatten --stripReflection --compiler "${DXC_SPIRV_PATH}"
            --sourceDir "Shaders"
            -p SPIRV --WX
            -c Shaders.cfg
            -o "${NRIF_SHADERS_PATH}"
            -I "Shaders"
            -D COMPILER_DXC
            -D VULKAN
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
endif ()

set_property (TARGET ${PROJECT_NAME}_Shaders PROPERTY FOLDER ${PROJECT_NAME})
add_dependencies (${PROJECT_NAME} ${PROJECT_NAME}_Shaders)

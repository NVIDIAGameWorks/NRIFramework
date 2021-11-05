cmake_minimum_required(VERSION 3.10)

# Options:
#set(NRIF_OUTPUT_PATH "my/output/")

project(NRIFramework LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(NRIF_BUILD_ARCHITECTURE_SUFFIX "x86_64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(NRIF_BUILD_ARCHITECTURE_SUFFIX "x86")
endif()

# Detecting target architecture
if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(NRIF_PACKMAN_PLATFORM_SUFFIX "x86_64")
elseif ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    set(NRIF_PACKMAN_PLATFORM_SUFFIX "aarch64")
else()
    message(FATAL_ERROR "Unknown architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

# Use different scripts on different platforms
if (WIN32)
    set(NRIF_PACKMAN_SCRIPT "packman.cmd")
    set(NRIF_PACKMAN_PLATFORM "windows-${NRIF_PACKMAN_PLATFORM_SUFFIX}")
else()
    set(NRIF_PACKMAN_SCRIPT "packman")
    set(NRIF_PACKMAN_PLATFORM "linux-${NRIF_PACKMAN_PLATFORM_SUFFIX}")
endif()

# Verify that packman is available
find_file(NRIF_PACKMAN NAMES ${NRIF_PACKMAN_SCRIPT} PATHS "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman")
if (NOT NRIF_PACKMAN)
    message(FATAL_ERROR "Can't find packman: '${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/${NRIF_PACKMAN_SCRIPT}' (did you init submodules?)")
endif()

# Loading dependcies using packman
execute_process(COMMAND "${NRIF_PACKMAN}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -t nri_data_version=${NRI_DATA_VERSION}
    -p ${NRIF_PACKMAN_PLATFORM} RESULT_VARIABLE NRIF_PACKMAN_RESULT)
if ("${NRIF_PACKMAN_RESULT}" STREQUAL "0")
    message(STATUS "Packman result: success (platform: '${NRIF_PACKMAN_PLATFORM}')")
else()
    message(FATAL_ERROR "Packman result: ${NRIF_PACKMAN_RESULT} (platform: '${NRIF_PACKMAN_PLATFORM}')")
endif()

# Defines for all projects
if (WIN32)
    set(NRIF_PLATFORM_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE GLFW_EXPOSE_NATIVE_WIN32 GLFW_INCLUDE_NONE)
else()
    set(NRIF_PLATFORM_DEFINITIONS GLFW_EXPOSE_NATIVE_X11 GLFW_INCLUDE_NONE)
endif()

# Verify that required static and shared libraries are available
if (WIN32)
    find_library(NRIF_INPUT_GLFW_LIB NAMES glfw3 PATHS "External/Glfw/build-${NRIF_BUILD_ARCHITECTURE_SUFFIX}/Release/src" REQUIRED)
    find_library(NRIF_INPUT_ASSIMP_LIB NAMES assimp-vc142-mt PATHS "External/Assimp/lib/${NRIF_BUILD_ARCHITECTURE_SUFFIX}" REQUIRED)
    find_file(NRIF_INPUT_ASSIMP_SHARED_LIB NAMES assimp-vc142-mt.dll PATHS "External/Assimp/bin/${NRIF_BUILD_ARCHITECTURE_SUFFIX}" REQUIRED)
else()
    find_library(NRIF_INPUT_GLFW_LIB NAMES glfw3 PATHS "External/Glfw/build/Release/src" REQUIRED)
    find_library(NRIF_INPUT_ASSIMP_SHARED_LIB NAMES assimp PATHS "External/Assimp/bin" REQUIRED)
    set(NRIF_INPUT_ASSIMP_LIB ${NRIF_INPUT_ASSIMP_SHARED_LIB})
endif()

# Verify that required Linux packages are installed
if (UNIX)
    find_path(NRIF_XLIB_HEADERS NAMES "X11/Xlib.h")
    if (NOT NRIF_XLIB_HEADERS)
        message(FATAL_ERROR "Can't find 'X11/xlib.h'. (libx11-dev is not installed?)")
    endif()
    find_path(NRIF_XRANDR_HEADERS NAMES "X11/extensions/Xrandr.h")
    if (NOT NRIF_XRANDR_HEADERS)
        message(FATAL_ERROR "Can't find 'X11/extensions/Xrandr.h'. (libxrandr-dev is not installed?)")
    endif()
endif()

# MathLib requires SSE4.1
if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(NRIF_ARCHITECTURE_COMPILE_OPTIONS -msse4.1)
endif()

# Compiler-specific options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # -Wno-pragma-pack: assimp causes a number of these warnings
    # warning: the current #pragma pack alignment value is modified in the included file [-Wpragma-pack]
    add_compile_options(${NRIF_ARCHITECTURE_COMPILE_OPTIONS} -Wno-missing-braces -Wno-return-type-c-linkage -Wno-pragma-pack)
    set(DETEX_COMPILE_OPTIONS -Wno-unused-function)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(${NRIF_ARCHITECTURE_COMPILE_OPTIONS})
endif()

if (WIN32)
    # On Windows, GLFW requires /MT
    add_compile_options($<$<CONFIG:Debug>:/MTd> $<$<CONFIG:Release>:/MT> $<$<CONFIG:RelWithDebInfo>:/MT> $<$<CONFIG:MinSizeRel>:/MT>)
endif()

# External static library: ImGui
file(GLOB NRIF_IMGUI_SOURCE "External/ImGui/*.cpp" "External/ImGui/*.h")
source_group("" FILES ${NRIF_IMGUI_SOURCE})
add_library(ImGui STATIC ${NRIF_IMGUI_SOURCE})
target_compile_definitions(ImGui PRIVATE ${NRIF_PLATFORM_DEFINITIONS})
set_property(TARGET ImGui PROPERTY FOLDER "External")

# External static library: Detex
file(GLOB NRIF_DETEX_SOURCE "External/Detex/*.c" "External/Detex/*.h")
source_group("" FILES ${NRIF_DETEX_SOURCE})
add_library(Detex STATIC ${NRIF_DETEX_SOURCE})
target_compile_definitions(Detex PRIVATE ${NRIF_PLATFORM_DEFINITIONS})
target_compile_options(Detex PRIVATE ${DETEX_COMPILE_OPTIONS})
set_property(TARGET Detex PROPERTY FOLDER "External")

# Static library: NRIFramework
file(GLOB NRIF_FRAMEWORK_SOURCE "Source/*.cpp" "Source/*.h")
source_group("" FILES ${NRIF_FRAMEWORK_SOURCE})
file(GLOB NRIF_FRAMEWORK_HEADERS "Include/*.hpp" "Include/*.h")
source_group("Include" FILES ${NRIF_FRAMEWORK_HEADERS})
add_library(NRIFramework STATIC ${NRIF_FRAMEWORK_SOURCE} ${NRIF_FRAMEWORK_HEADERS})
target_link_libraries(NRIFramework PRIVATE ImGui Detex ${NRIF_INPUT_GLFW_LIB} ${NRIF_INPUT_ASSIMP_LIB})

if (UNIX)
    target_link_libraries(NRIFramework PRIVATE ${CMAKE_DL_LIBS} pthread X11)
endif()

target_include_directories(NRIFramework PRIVATE "Source" "Include" "External" "External/NRI/Include" "External/Assimp/include")
target_compile_definitions(NRIFramework PRIVATE ${NRIF_PLATFORM_DEFINITIONS})
set_property(TARGET NRIFramework PROPERTY FOLDER "Samples")

if (NRIF_OUTPUT_PATH)
    get_filename_component(NRIF_INPUT_ASSIMP_SHARED_LIB_NAME ${NRIF_INPUT_ASSIMP_SHARED_LIB} NAME)
    set(NRIF_OUTPUT_ASSIMP_SHARED_LIB ${NRIF_OUTPUT_PATH}/$<CONFIG>/${NRIF_INPUT_ASSIMP_SHARED_LIB_NAME})
    add_custom_command(TARGET NRIFramework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NRIF_INPUT_ASSIMP_SHARED_LIB} ${NRIF_OUTPUT_ASSIMP_SHARED_LIB}
        COMMAND_EXPAND_LISTS)
endif()

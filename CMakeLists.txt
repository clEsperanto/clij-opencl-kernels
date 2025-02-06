cmake_minimum_required(VERSION 3.20)

project(cleKernels VERSION 3.1.1)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(GLOB_RECURSE KERNELS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cl" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")

set(KERNELS_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(KERNELS_HEADER_FILE ${KERNELS_HEADER_DIR}/dummy_header.h)

add_custom_command(
    OUTPUT ${KERNELS_HEADER_FILE}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernel_to_header.py ${CMAKE_CURRENT_SOURCE_DIR} ${KERNELS_HEADER_DIR}
    COMMENT "Generating headers from ocl kernels files."
    DEPENDS ${KERNELS_FILES}
)
add_custom_target(generate_headers ALL DEPENDS ${KERNELS_HEADER_FILE})

file(GLOB KERNEL_HEADERS "${KERNELS_HEADER_DIR}/*.h")

add_library(${PROJECT_NAME} INTERFACE)
add_library(cleKernels::cleKernels ALIAS cleKernels)
add_dependencies(${PROJECT_NAME} generate_headers)
target_include_directories(${PROJECT_NAME} INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_BUILD_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Development
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION include COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT cleKernelsTargets
    FILE cleKernelsTargets.cmake
    NAMESPACE cleKernels::
    DESTINATION lib/cmake/cleKernels
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}Targets DESTINATION lib/cmake/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

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
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
add_dependencies(${PROJECT_NAME} generate_headers)

target_include_directories(${PROJECT_NAME} INTERFACE 
    $<BUILD_INTERFACE:${KERNELS_HEADER_DIR}>
    $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)

# Export the cleKernels target
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# Export the cleKernelsTargets export set
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
)

# Ensure cleKernels target is included in the export set
export(
    TARGETS ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}::
    FILE ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake
)

install(DIRECTORY ${KERNELS_HEADER_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

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

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
)
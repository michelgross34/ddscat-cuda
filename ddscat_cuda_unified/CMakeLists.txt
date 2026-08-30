cmake_minimum_required(VERSION 3.20)

# This directory can be configured standalone, but is intended to be included
# from the DDSCAT root CMakeLists with add_subdirectory(ddscat_cuda_unified).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(DDSCAT_CUDA_UNIFIED VERSION 7.3.4 LANGUAGES Fortran C)
endif()

get_filename_component(_DDSCAT_ROOT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(DDSCAT_CUDA_ROOT "${_DDSCAT_ROOT_DEFAULT}" CACHE PATH "DDSCAT root used by the unified CUDA targets")
set(DDSCAT_CUDA_SRC "${DDSCAT_CUDA_ROOT}/src")
set(DDSCAT_CUDA_FFTW_ROOT "${DDSCAT_CUDA_ROOT}/fftw" CACHE PATH "FFTW3 float installation for unified CUDA DDSCAT")

if(NOT EXISTS "${DDSCAT_CUDA_SRC}/DDSCAT.f90")
  message(FATAL_ERROR "DDSCAT_CUDA_ROOT is incorrect: ${DDSCAT_CUDA_SRC}/DDSCAT.f90 not found")
endif()

find_library(DDSCAT_CUDA_FFTW_LIBRARY
  NAMES fftw3f
  PATHS "${DDSCAT_CUDA_FFTW_ROOT}"
  NO_DEFAULT_PATH
  REQUIRED)

set(DDSCAT_CUDA_CORE_SOURCES
  ${DDSCAT_CUDA_SRC}/ddprecision.f90
  ${DDSCAT_CUDA_SRC}/cgmodule.f90
  ${DDSCAT_CUDA_SRC}/alphadiag.f90
  ${DDSCAT_CUDA_SRC}/blas.f90
  ${DDSCAT_CUDA_SRC}/bself.f90
  ${DDSCAT_CUDA_SRC}/ccgpack.f90
  ${DDSCAT_CUDA_SRC}/cgcommon.f90
  ${DDSCAT_CUDA_SRC}/cglib2.f90
  ${DDSCAT_CUDA_SRC}/cglib3.f90
  ${DDSCAT_CUDA_SRC}/cgsarkar2.f90
  ${DDSCAT_CUDA_SRC}/cgsarkar3.f90
  ${DDSCAT_CUDA_SRC}/cisi.f90
  ${DDSCAT_CUDA_SRC}/copyit.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/fortran/cprod_cuda.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/fortran/cuda_matvec_bridge.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ddscat_cuda_loader.c
  ${DDSCAT_CUDA_SRC}/cxfft3n.f90
  ${DDSCAT_CUDA_SRC}/cxfft3_mkl_fake.f90
  ${DDSCAT_CUDA_SRC}/cxfftw_fftw3.f90
  ${DDSCAT_CUDA_SRC}/ddcommon.f90
  ${DDSCAT_CUDA_SRC}/dielec.f90
  ${DDSCAT_CUDA_SRC}/direct_calc.f90
  ${DDSCAT_CUDA_SRC}/direct_calcb.f90
  ${DDSCAT_CUDA_SRC}/divide.f90
  ${DDSCAT_CUDA_SRC}/dsyevj3.f90
  ${DDSCAT_CUDA_SRC}/dummy.f90
  ${DDSCAT_CUDA_SRC}/errmsg.f90
  ${DDSCAT_CUDA_SRC}/eself.f90
  ${DDSCAT_CUDA_SRC}/evala.f90
  ${DDSCAT_CUDA_SRC}/evale.f90
  ${DDSCAT_CUDA_SRC}/evalq.f90
  ${DDSCAT_CUDA_SRC}/extend.f90
  ${DDSCAT_CUDA_SRC}/gasdev.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/fortran/getfml_cuda.f90
  ${DDSCAT_CUDA_SRC}/getmueller.f90
  ${DDSCAT_CUDA_SRC}/gpbicg.f90
  ${DDSCAT_CUDA_SRC}/gpfa.f90
  ${DDSCAT_CUDA_SRC}/interp.f90
  ${DDSCAT_CUDA_SRC}/mpi_fake.f90
  ${DDSCAT_CUDA_SRC}/namer.f90
  ${DDSCAT_CUDA_SRC}/namer2.f90
  ${DDSCAT_CUDA_SRC}/namid.f90
  ${DDSCAT_CUDA_SRC}/nearfield.f90
  ${DDSCAT_CUDA_SRC}/nuller.f90
  ${DDSCAT_CUDA_SRC}/orient.f90
  ${DDSCAT_CUDA_SRC}/p_lm.f90
  ${DDSCAT_CUDA_SRC}/pad.f90
  ${DDSCAT_CUDA_SRC}/pbcscavec.f90
  ${DDSCAT_CUDA_SRC}/pim.f90
  ${DDSCAT_CUDA_SRC}/prinaxis.f90
  ${DDSCAT_CUDA_SRC}/qmrpim2.f90
  ${DDSCAT_CUDA_SRC}/ran3.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/fortran/reapar_cuda.f90
  ${DDSCAT_CUDA_SRC}/reashp.f90
  ${DDSCAT_CUDA_SRC}/reduce.f90
  ${DDSCAT_CUDA_SRC}/restore.f90
  ${DDSCAT_CUDA_SRC}/rot2.f90
  ${DDSCAT_CUDA_SRC}/rotate.f90
  ${DDSCAT_CUDA_SRC}/scat.f90
  ${DDSCAT_CUDA_SRC}/scavec.f90
  ${DDSCAT_CUDA_SRC}/sizer.f90
  ${DDSCAT_CUDA_SRC}/tangcg.f90
  ${DDSCAT_CUDA_SRC}/tar2el.f90
  ${DDSCAT_CUDA_SRC}/tar2sp.f90
  ${DDSCAT_CUDA_SRC}/tar3el.f90
  ${DDSCAT_CUDA_SRC}/taranirec.f90
  ${DDSCAT_CUDA_SRC}/tarblocks.f90
  ${DDSCAT_CUDA_SRC}/tarcel.f90
  ${DDSCAT_CUDA_SRC}/tarcyl.f90
  ${DDSCAT_CUDA_SRC}/tarcylcap.f90
  ${DDSCAT_CUDA_SRC}/tarell.f90
  ${DDSCAT_CUDA_SRC}/target.f90
  ${DDSCAT_CUDA_SRC}/targspher.f90
  ${DDSCAT_CUDA_SRC}/tarhex.f90
  ${DDSCAT_CUDA_SRC}/tarlyrslab.f90
  ${DDSCAT_CUDA_SRC}/tarnas.f90
  ${DDSCAT_CUDA_SRC}/tarnsp.f90
  ${DDSCAT_CUDA_SRC}/taroct.f90
  ${DDSCAT_CUDA_SRC}/taronion.f90
  ${DDSCAT_CUDA_SRC}/tarpbxn.f90
  ${DDSCAT_CUDA_SRC}/tarprsm.f90
  ${DDSCAT_CUDA_SRC}/tarrctblk3.f90
  ${DDSCAT_CUDA_SRC}/tarrctell.f90
  ${DDSCAT_CUDA_SRC}/tarrec.f90
  ${DDSCAT_CUDA_SRC}/tarrecrec.f90
  ${DDSCAT_CUDA_SRC}/tarsell.f90
  ${DDSCAT_CUDA_SRC}/tarslbhol.f90
  ${DDSCAT_CUDA_SRC}/tartet.f90
  ${DDSCAT_CUDA_SRC}/timeit.f90
  ${DDSCAT_CUDA_SRC}/unreduce.f90
  ${DDSCAT_CUDA_SRC}/version.f90
  ${DDSCAT_CUDA_SRC}/wrimsg.f90
  ${DDSCAT_CUDA_SRC}/writebin.f90
  ${DDSCAT_CUDA_SRC}/writefml.f90
  ${DDSCAT_CUDA_SRC}/writepol.f90
  ${DDSCAT_CUDA_SRC}/writesca.f90
  ${DDSCAT_CUDA_SRC}/zbcg2wp.f90)

set(_CUDA_MOD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mod")
file(MAKE_DIRECTORY "${_CUDA_MOD_DIR}")

add_library(ddscat_cuda_core STATIC ${DDSCAT_CUDA_CORE_SOURCES} ${DDSCAT_CUDA_SRC}/cxfftw_fftw3.c)
set_target_properties(ddscat_cuda_core PROPERTIES Fortran_MODULE_DIRECTORY "${_CUDA_MOD_DIR}")
target_include_directories(ddscat_cuda_core PRIVATE
  "${DDSCAT_CUDA_FFTW_ROOT}" "${DDSCAT_CUDA_SRC}" "${_CUDA_MOD_DIR}")
target_link_libraries(ddscat_cuda_core PUBLIC "${DDSCAT_CUDA_FFTW_LIBRARY}")
target_compile_definitions(ddscat_cuda_core PRIVATE sp DDSCAT_CUDA_MATVEC DDSCAT_CUDA_SOLVERS)
set_property(TARGET ddscat_cuda_core PROPERTY Fortran_PREPROCESS ON)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(ddscat_cuda_core PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(ddscat_cuda_core PUBLIC dl)
endif()

function(ddscat_add_cuda_executable target_name)
  add_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/fortran/DDSCAT_cuda.f90)
  target_link_libraries(${target_name} PRIVATE ddscat_cuda_core)
  target_compile_definitions(${target_name} PRIVATE sp DDSCAT_CUDA_MATVEC DDSCAT_CUDA_SOLVERS)
  set_property(TARGET ${target_name} PROPERTY Fortran_PREPROCESS ON)
  set_target_properties(${target_name} PROPERTIES
    Fortran_MODULE_DIRECTORY "${_CUDA_MOD_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_include_directories(${target_name} PRIVATE "${_CUDA_MOD_DIR}" "${DDSCAT_CUDA_SRC}")
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target_name} PRIVATE -ffree-line-length-none)
  endif()
endfunction()

ddscat_add_cuda_executable(ddscat_cuda)
ddscat_add_cuda_executable(ddscat_cuda_slice)

# Both CUDA DLLs are produced from the SAME .cu source. The only difference is
# the compile definition DDSCAT_CUDA_BACKEND_SLICES for the low-memory backend.
if(WIN32)
  set(DDSCAT_CUDA_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
  set(DDSCAT_CUDA_BACKEND_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_backend.bat")
  set(DDSCAT_CUDA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/cuda/ddscat_matvec_cuda.cu")
  set(DDSCAT_CUDA_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/cuda/ddscat_matvec_cuda.h")

  add_custom_command(
    OUTPUT "${DDSCAT_CUDA_BIN_DIR}/ddscat_matvec_cuda.dll"
    COMMAND cmd /c call "${DDSCAT_CUDA_BACKEND_SCRIPT}" FFT3D "${DDSCAT_CUDA_BIN_DIR}"
    DEPENDS "${DDSCAT_CUDA_SOURCE}" "${DDSCAT_CUDA_HEADER}" "${DDSCAT_CUDA_BACKEND_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM)
  add_custom_target(ddscat_cuda_fft3d_dll ALL DEPENDS "${DDSCAT_CUDA_BIN_DIR}/ddscat_matvec_cuda.dll")

  add_custom_command(
    OUTPUT "${DDSCAT_CUDA_BIN_DIR}/ddscat_matvec_cuda_slice.dll"
    COMMAND cmd /c call "${DDSCAT_CUDA_BACKEND_SCRIPT}" SLICES "${DDSCAT_CUDA_BIN_DIR}"
    DEPENDS "${DDSCAT_CUDA_SOURCE}" "${DDSCAT_CUDA_HEADER}" "${DDSCAT_CUDA_BACKEND_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM)
  add_custom_target(ddscat_cuda_slices_dll ALL DEPENDS "${DDSCAT_CUDA_BIN_DIR}/ddscat_matvec_cuda_slice.dll")

  add_dependencies(ddscat_cuda ddscat_cuda_fft3d_dll)
  add_dependencies(ddscat_cuda_slice ddscat_cuda_slices_dll)

  set(DDSCAT_CUDA_FFTW_DLL "${DDSCAT_CUDA_FFTW_ROOT}/libfftw3f-3.dll")
  if(EXISTS "${DDSCAT_CUDA_FFTW_DLL}")
    add_custom_command(TARGET ddscat_cuda POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DDSCAT_CUDA_FFTW_DLL}" "$<TARGET_FILE_DIR:ddscat_cuda>")
    add_custom_command(TARGET ddscat_cuda_slice POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DDSCAT_CUDA_FFTW_DLL}" "$<TARGET_FILE_DIR:ddscat_cuda_slice>")
  endif()
endif()

# Convenient CLion target: builds the two CUDA executables and, on Windows,
# their two DLL backends. CPU ddscat remains a separate root target.
add_custom_target(ddscat_cuda_all DEPENDS ddscat_cuda ddscat_cuda_slice)

message(STATUS "Unified DDSCAT CUDA source: ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ddscat_matvec_cuda.cu")
message(STATUS "  target ddscat_cuda       -> FFT3D MATVEC backend")
message(STATUS "  target ddscat_cuda_slice -> Z-FFT + XY-slice MATVEC backend")
message(STATUS "Both targets share Green CUDA, solvers, chunked cublasZdot and Fortran bridge.")

cmake_minimum_required(VERSION 3.20)

project(DDSCAT VERSION 7.3.4 LANGUAGES Fortran C)

# Equivalent to the default non-MPI configuration in src/Makefile.
set(DDSCAT_PRECISION sp CACHE STRING "Floating-point precision: sp or dp")
set_property(CACHE DDSCAT_PRECISION PROPERTY STRINGS sp dp)
option(DDSCAT_OPENMP "Enable OpenMP support" OFF)

if(NOT DDSCAT_PRECISION STREQUAL "sp" AND NOT DDSCAT_PRECISION STREQUAL "dp")
    message(FATAL_ERROR "DDSCAT_PRECISION must be sp or dp")
endif()

set(DDSCAT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(FFTW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fftw" CACHE PATH "FFTW3 installation directory")

if(DDSCAT_PRECISION STREQUAL "dp")
    set(FFTW_LIBRARY_NAME fftw3)
else()
    set(FFTW_LIBRARY_NAME fftw3f)
endif()
find_library(FFTW_LIBRARY
    NAMES ${FFTW_LIBRARY_NAME}
    PATHS "${FFTW_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED
)
set(FFTW_INCLUDE_DIR "${FFTW_ROOT}")

# Sources shared by the main DDSCAT executable. mpi_fake.f90 deliberately
# replaces the MPI implementation; no MPI package or MPI compiler is used.
set(DDSCAT_CORE_SOURCES
    ${DDSCAT_SRC}/ddprecision.f90
    ${DDSCAT_SRC}/cgmodule.f90
    ${DDSCAT_SRC}/alphadiag.f90
    ${DDSCAT_SRC}/blas.f90
    ${DDSCAT_SRC}/bself.f90
    ${DDSCAT_SRC}/ccgpack.f90
    ${DDSCAT_SRC}/cgcommon.f90
    ${DDSCAT_SRC}/cglib2.f90
    ${DDSCAT_SRC}/cglib3.f90
    ${DDSCAT_SRC}/cgsarkar2.f90
    ${DDSCAT_SRC}/cgsarkar3.f90
    ${DDSCAT_SRC}/cisi.f90
    ${DDSCAT_SRC}/copyit.f90
    ${DDSCAT_SRC}/cprod.f90
    ${DDSCAT_SRC}/cxfft3n.f90
    ${DDSCAT_SRC}/cxfft3_mkl_fake.f90
    ${DDSCAT_SRC}/cxfftw_fftw3.f90
    ${DDSCAT_SRC}/ddcommon.f90
    ${DDSCAT_SRC}/dielec.f90
    ${DDSCAT_SRC}/direct_calc.f90
    ${DDSCAT_SRC}/direct_calcb.f90
    ${DDSCAT_SRC}/divide.f90
    ${DDSCAT_SRC}/dsyevj3.f90
    ${DDSCAT_SRC}/dummy.f90
    ${DDSCAT_SRC}/errmsg.f90
    ${DDSCAT_SRC}/eself.f90
    ${DDSCAT_SRC}/evala.f90
    ${DDSCAT_SRC}/evale.f90
    ${DDSCAT_SRC}/evalq.f90
    ${DDSCAT_SRC}/extend.f90
    ${DDSCAT_SRC}/gasdev.f90
    ${DDSCAT_SRC}/getfml.f90
    ${DDSCAT_SRC}/getmueller.f90
    ${DDSCAT_SRC}/gpbicg.f90
    ${DDSCAT_SRC}/gpfa.f90
    ${DDSCAT_SRC}/interp.f90
    ${DDSCAT_SRC}/mpi_fake.f90
    ${DDSCAT_SRC}/namer.f90
    ${DDSCAT_SRC}/namer2.f90
    ${DDSCAT_SRC}/namid.f90
    ${DDSCAT_SRC}/nearfield.f90
    ${DDSCAT_SRC}/nuller.f90
    ${DDSCAT_SRC}/orient.f90
    ${DDSCAT_SRC}/p_lm.f90
    ${DDSCAT_SRC}/pad.f90
    ${DDSCAT_SRC}/pbcscavec.f90
    ${DDSCAT_SRC}/pim.f90
    ${DDSCAT_SRC}/prinaxis.f90
    ${DDSCAT_SRC}/qmrpim2.f90
    ${DDSCAT_SRC}/ran3.f90
    ${DDSCAT_SRC}/reapar.f90
    ${DDSCAT_SRC}/reashp.f90
    ${DDSCAT_SRC}/reduce.f90
    ${DDSCAT_SRC}/restore.f90
    ${DDSCAT_SRC}/rot2.f90
    ${DDSCAT_SRC}/rotate.f90
    ${DDSCAT_SRC}/scat.f90
    ${DDSCAT_SRC}/scavec.f90
    ${DDSCAT_SRC}/sizer.f90
    ${DDSCAT_SRC}/tangcg.f90
    ${DDSCAT_SRC}/tar2el.f90
    ${DDSCAT_SRC}/tar2sp.f90
    ${DDSCAT_SRC}/tar3el.f90
    ${DDSCAT_SRC}/taranirec.f90
    ${DDSCAT_SRC}/tarblocks.f90
    ${DDSCAT_SRC}/tarcel.f90
    ${DDSCAT_SRC}/tarcyl.f90
    ${DDSCAT_SRC}/tarcylcap.f90
    ${DDSCAT_SRC}/tarell.f90
    ${DDSCAT_SRC}/target.f90
    ${DDSCAT_SRC}/targspher.f90
    ${DDSCAT_SRC}/tarhex.f90
    ${DDSCAT_SRC}/tarlyrslab.f90
    ${DDSCAT_SRC}/tarnas.f90
    ${DDSCAT_SRC}/tarnsp.f90
    ${DDSCAT_SRC}/taroct.f90
    ${DDSCAT_SRC}/taronion.f90
    ${DDSCAT_SRC}/tarpbxn.f90
    ${DDSCAT_SRC}/tarprsm.f90
    ${DDSCAT_SRC}/tarrctblk3.f90
    ${DDSCAT_SRC}/tarrctell.f90
    ${DDSCAT_SRC}/tarrec.f90
    ${DDSCAT_SRC}/tarrecrec.f90
    ${DDSCAT_SRC}/tarsell.f90
    ${DDSCAT_SRC}/tarslbhol.f90
    ${DDSCAT_SRC}/tartet.f90
    ${DDSCAT_SRC}/timeit.f90
    ${DDSCAT_SRC}/unreduce.f90
    ${DDSCAT_SRC}/version.f90
    ${DDSCAT_SRC}/wrimsg.f90
    ${DDSCAT_SRC}/writebin.f90
    ${DDSCAT_SRC}/writefml.f90
    ${DDSCAT_SRC}/writepol.f90
    ${DDSCAT_SRC}/writesca.f90
    ${DDSCAT_SRC}/zbcg2wp.f90
)

add_library(ddscat_core STATIC ${DDSCAT_CORE_SOURCES} ${DDSCAT_SRC}/cxfftw_fftw3.c)
target_include_directories(ddscat_core PRIVATE "${FFTW_INCLUDE_DIR}")
target_link_libraries(ddscat_core PUBLIC "${FFTW_LIBRARY}")
add_executable(ddscat ${DDSCAT_SRC}/DDSCAT.f90)
target_link_libraries(ddscat PRIVATE ddscat_core)

add_executable(calltarget ${DDSCAT_SRC}/CALLTARGET.f90)
target_link_libraries(calltarget PRIVATE ddscat_core)

add_executable(ddpostprocess
    ${DDSCAT_SRC}/ddprecision.f90
    ${DDSCAT_SRC}/DDPOSTPROCESS.f90
    ${DDSCAT_SRC}/readnf_bcom.f90
    ${DDSCAT_SRC}/readnf_ecom.f90
    ${DDSCAT_SRC}/readnf.f90
    ${DDSCAT_SRC}/version.f90
    ${DDSCAT_SRC}/vtr.f90
)

add_executable(vtrconvert ${DDSCAT_SRC}/VTRCONVERT.f90 ${DDSCAT_SRC}/vtr.f90)

set(DDSCAT_TARGETS ddscat_core ddscat calltarget ddpostprocess vtrconvert)
foreach(target IN LISTS DDSCAT_TARGETS)
    target_compile_definitions(${target} PRIVATE ${DDSCAT_PRECISION})
    set_property(TARGET ${target} PROPERTY Fortran_PREPROCESS ON)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${DDSCAT_SRC}")
endforeach()

if(DDSCAT_PRECISION STREQUAL "dp")
    target_compile_definitions(ddscat_core PRIVATE DDSCAT_DP)
endif()

if(WIN32)
    if(DDSCAT_PRECISION STREQUAL "dp")
        set(FFTW_DLL "${FFTW_ROOT}/libfftw3-3.dll")
    else()
        set(FFTW_DLL "${FFTW_ROOT}/libfftw3f-3.dll")
    endif()
    if(EXISTS "${FFTW_DLL}")
        add_custom_command(TARGET ddscat POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW_DLL}" "$<TARGET_FILE_DIR:ddscat>")
    endif()
endif()

if(DDSCAT_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS Fortran)
    foreach(target IN LISTS DDSCAT_TARGETS)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_Fortran)
        target_compile_definitions(${target} PRIVATE openmp)
    endforeach()
endif()

set_target_properties(ddscat calltarget ddpostprocess vtrconvert
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Unified CUDA variants: one shared codebase builds both FFT3D and SLICES.
option(DDSCAT_BUILD_CUDA_UNIFIED "Build ddscat_cuda and ddscat_cuda_slice from ddscat_cuda_unified" ON)
if(DDSCAT_BUILD_CUDA_UNIFIED)
    add_subdirectory(ddscat_cuda_unified)
endif()

if(BUILD_GPU)
  file (GLOB PLAS_SRC
  *.cc
  cpu/*.cc
  )
  
  if(CUDA_FOUND)
    #compile cuda files and add the compiled object files to your normal source files
    file(GLOB_RECURSE CUDA_SRC  gpu/*.cu)
    include_directories(SYSTEM BEFORE PRIVATE ${CUDA_INCLUDE_DIRS}/cub)
    include_directories(plas BEFORE PRIVATE ${PROJECT_SOURCE_DIR})
    include_directories(SYSTEM BEFORE PRIVATE ${TF_INC_DIR})
    include_directories(SYSTEM BEFORE PRIVATE ${TF_INC_DIR}/external/nsync/public)
    CUDA_ADD_LIBRARY(plas_gpu ${CUDA_SRC} STATIC)
  endif(CUDA_FOUND)

  add_library(plas STATIC ${PLAS_SRC})
  target_link_libraries(plas
  plas_gpu
  -L/opt/intel/lib/intel64
  -liomp5 # use intel iomp to make it compatiable with MKL
  )

  target_include_directories(plas BEFORE PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(plas SYSTEM BEFORE PRIVATE ${TF_INC_DIR})
  target_include_directories(plas SYSTEM BEFORE PRIVATE ${TF_INC_DIR}/external/nsync/public)
  target_include_directories(plas SYSTEM BEFORE PRIVATE ${MKL_INCLUDE_DIRS})

  target_compile_options(plas PRIVATE -Wno-unused-variable -Wno-unused-parameter)
else(BUILD_GPU)
  file (GLOB PLAS_SRC
  *.cc
  cpu/*.cc
  )

  add_library(plas STATIC ${PLAS_SRC})
  target_link_libraries(plas
  -L/opt/intel/lib/intel64
  -liomp5 # use intel iomp to make it compatiable with MKL
  )

  target_include_directories(plas BEFORE PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(plas SYSTEM BEFORE PRIVATE ${TF_INC_DIR})
  target_include_directories(plas SYSTEM BEFORE PRIVATE ${TF_INC_DIR}/external/nsync/public)
  target_include_directories(plas SYSTEM BEFORE PRIVATE ${MKL_INCLUDE_DIRS})

  # enable asan
  #target_compile_options(plas PRIVATE -Wno-unused-variable -Wno-unused-parameter -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address)
  #target_link_options(plas PRIVATE -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address)

  target_compile_options(plas PRIVATE -Wno-unused-variable -Wno-unused-parameter)

endif(BUILD_GPU)

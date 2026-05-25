cmake_minimum_required(VERSION 3.22)
# 设定项目名称
project(dots_and_boxes)
# 设定语言标准
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

IF(WIN32)
# 定义第三方库路径
# 使用本地 GCC 15.2 编译的 SFML
if(NOT DEFINED CACHE{SFML_DIR})
    set(SFML_DIR 3rdparty/SFML_new/lib/cmake/SFML)
endif()
ENDIF()

#set(nowide_DIR 3rdparty/nowide/lib/cmake/nowide)

# 动态链接SFML（已用当前 GCC 15.2 重新编译）
#set(SFML_STATIC_LIBRARIES TRUE)

# 查找第三方库
find_package(SFML 2.5 COMPONENTS
        system window graphics REQUIRED)
find_package(nowide)
# ========== 可选: ONNX Runtime CNN 推理 ==========
option(USE_ONNX "Enable ONNX Runtime for CNN model inference" OFF)

if(USE_ONNX)
    # ONNX Runtime 查找路径
    if(NOT DEFINED ONNXRUNTIME_DIR)
        set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime")
    endif()

    if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime found at: ${ONNXRUNTIME_DIR}")
        set(ONNX_FOUND TRUE)
    else()
        message(WARNING "ONNX Runtime NOT found at ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        message(WARNING "Please download ONNX Runtime and place in 3rdparty/onnxruntime/")
        set(ONNX_FOUND FALSE)
        set(USE_ONNX OFF)
    endif()
endif()

# AI 核心源文件（GUI 和 SelfPlay 共享）
set(AI_SOURCES
        src/AI/assess.cpp
        src/AI/board.cpp
        src/AI/Node.cpp
        src/AI/UCT.cpp
        src/AI/az/az_action.cpp
        src/AI/az/az_encoder.cpp
        src/AI/az/az_evaluator.cpp
        src/AI/az/az_expert.cpp
        src/AI/az/az_node.cpp
        src/AI/az/az_mcts.cpp
        src/AI/az/az_move.cpp
        src/AI/az/az_neural_net.cpp
        src/AI/az/az_selfplay.cpp
        )

# ONNX 评估器源文件（条件编译）
if(USE_ONNX AND ONNX_FOUND)
    list(APPEND AI_SOURCES src/AI/az/az_onnx_evaluator.cpp)
endif()

# GUI 主程序源文件
set(GUI_SOURCES
        src/main.cpp
        src/element/Time.cpp
        src/CJSON/cJSON.c
        src/CJSON/datarecorder.cpp
        )

# 如果想要编译出来的程序不带控制台，使用这个
#add_executable(dots_and_boxes WIN32 ${AI_SOURCES} ${GUI_SOURCES})
# 如果想要编译出来的程序带控制台，使用这个
add_executable(dots_and_boxes ${AI_SOURCES} ${GUI_SOURCES})

# 自对弈训练数据生成工具（无 SFML 依赖）
add_executable(selfplay src/selfplay_main.cpp ${AI_SOURCES})

add_executable(evaluate_ai src/evaluate_ai_main.cpp ${AI_SOURCES})

# BoxesZero 反向训练自对弈工具
add_executable(backward_selfplay src/backward_selfplay_main.cpp ${AI_SOURCES})

# 单元测试
add_executable(az_tests test/az_tests.cpp ${AI_SOURCES})
target_include_directories(az_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

# ========== ONNX Runtime 配置 ==========
if(USE_ONNX AND ONNX_FOUND)
    set(ONNX_TARGETS dots_and_boxes selfplay evaluate_ai backward_selfplay az_tests)
    foreach(target ${ONNX_TARGETS})
        target_compile_definitions(${target} PRIVATE USE_ONNX)
        target_include_directories(${target} PRIVATE "${ONNXRUNTIME_DIR}/include")
        if(WIN32)
            target_link_directories(${target} PRIVATE "${ONNXRUNTIME_DIR}/lib")
            target_link_libraries(${target} PRIVATE onnxruntime)
        else()
            target_link_libraries(${target} PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
        endif()
    endforeach()
    message(STATUS "ONNX Runtime enabled for: ${ONNX_TARGETS}")
endif()

# 链接头文件查找目录
IF(WIN32)
target_include_directories(dots_and_boxes PUBLIC 3rdparty/SFML_new/include)
ENDIF()
target_include_directories(dots_and_boxes PUBLIC 3rdparty/nowide/include)

#set(CMAKE_EXE_LINKER_FLAGS "-static")
# 链接库
target_link_libraries(dots_and_boxes PRIVATE sfml-system sfml-window sfml-graphics)

IF (WIN32)
# 复制dll（仅新编译的 SFML + ONNX Runtime）
file(GLOB SFML_NEW_DLL 3rdparty/SFML_new/bin/*.dll)
set(PROJECT_DLL ${SFML_NEW_DLL})
if(USE_ONNX)
    # 主 DLL
    if(EXISTS "${ONNXRUNTIME_DIR}/bin/onnxruntime.dll")
        list(APPEND PROJECT_DLL "${ONNXRUNTIME_DIR}/bin/onnxruntime.dll")
    elseif(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
        list(APPEND PROJECT_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
    endif()
    # GPU provider DLLs (CUDA)
    foreach(provider_dll onnxruntime_providers_cuda onnxruntime_providers_shared)
        if(EXISTS "${ONNXRUNTIME_DIR}/bin/${provider_dll}.dll")
            list(APPEND PROJECT_DLL "${ONNXRUNTIME_DIR}/bin/${provider_dll}.dll")
        elseif(EXISTS "${ONNXRUNTIME_DIR}/lib/${provider_dll}.dll")
            list(APPEND PROJECT_DLL "${ONNXRUNTIME_DIR}/lib/${provider_dll}.dll")
        endif()
    endforeach()
    # CUDA runtime DLLs (from PyTorch installation)
    if(WIN32)
        execute_process(
            COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'lib')"
            OUTPUT_VARIABLE TORCH_LIB_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TORCH_LIB_DIR AND EXISTS "${TORCH_LIB_DIR}")
            foreach(cuda_dll cublas64_12 cublasLt64_12 cudart64_12 cudnn64_9 cudnn_adv64_9
                             cudnn_cnn64_9 cudnn_graph64_9 cudnn_heuristic64_9 cudnn_ops64_9
                             cufft64_11 curand64_10 cusolver64_11 cusparse64_12)
                if(EXISTS "${TORCH_LIB_DIR}/${cuda_dll}.dll")
                    list(APPEND PROJECT_DLL "${TORCH_LIB_DIR}/${cuda_dll}.dll")
                endif()
            endforeach()
        endif()
    endif()
endif()
add_custom_command(TARGET dots_and_boxes POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_DLL}
        $<TARGET_FILE_DIR:dots_and_boxes>
        )
ENDIF ()

# 复制资源文件
add_custom_command(TARGET dots_and_boxes POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/res
        $<TARGET_FILE_DIR:dots_and_boxes>/res
        )

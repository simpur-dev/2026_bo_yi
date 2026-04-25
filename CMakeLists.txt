cmake_minimum_required(VERSION 3.22)
# 设定项目名称
project(dots_and_boxes)
# 设定语言标准
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

IF(WIN32)
# 定义第三方库路径
# 如果通过 -DSFML_DIR 指定了路径，则使用指定的路径
if(NOT DEFINED CACHE{SFML_DIR})
    set(SFML_DIR 3rdparty/SFML/lib/cmake/SFML)
endif()
ENDIF()

#set(nowide_DIR 3rdparty/nowide/lib/cmake/nowide)

# 如需静态链接SFML请取消下句注释
#set(SFML_STATIC_LIBRARIES TRUE)

# 查找第三方库
find_package(SFML 2.5 COMPONENTS
        system window graphics REQUIRED)
find_package(nowide)
# AI 核心源文件（GUI 和 SelfPlay 共享）
set(AI_SOURCES
        src/AI/assess.cpp
        src/AI/board.cpp
        src/AI/Node.cpp
        src/AI/UCT.cpp
        src/AI/az/az_action.cpp
        src/AI/az/az_encoder.cpp
        src/AI/az/az_evaluator.cpp
        src/AI/az/az_node.cpp
        src/AI/az/az_mcts.cpp
        src/AI/az/az_move.cpp
        src/AI/az/az_neural_net.cpp
        src/AI/az/az_selfplay.cpp
        )

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

# 链接头文件查找目录
IF(WIN32)
target_include_directories(dots_and_boxes PUBLIC 3rdparty/SFML/include)
ENDIF()
target_include_directories(dots_and_boxes PUBLIC 3rdparty/nowide/include)

#如需静态链接
#set(CMAKE_EXE_LINKER_FLAGS "-static")
# 链接库
target_link_libraries(dots_and_boxes PRIVATE sfml-system sfml-window sfml-graphics)

IF (WIN32)
# 复制dll
file(GLOB_RECURSE PROJECT_DLL 3rdparty/*.dll)
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

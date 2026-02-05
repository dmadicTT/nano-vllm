cmake_minimum_required(VERSION 3.16)
project(nanovllm_cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# Symlink so clangd finds compile_commands.json when editing under cpp/
if(CMAKE_EXPORT_COMPILE_COMMANDS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

set(ENGINE_SOURCES
  src/engine/block_manager.cpp
  src/engine/hash.cpp
  src/engine/llm_engine.cpp
  src/engine/model_runner.cpp
  src/engine/scheduler.cpp
  src/engine/sequence.cpp
)

option(NANOVLLM_DEBUG "Enable engine debug logs (scheduler, block_manager, model_runner)" OFF)
add_library(nanovllm_engine ${ENGINE_SOURCES})
target_include_directories(nanovllm_engine PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(nanovllm_engine PUBLIC cxx_std_17)
if(NANOVLLM_DEBUG)
  target_compile_definitions(nanovllm_engine PUBLIC NANOVLLM_DEBUG)
endif()

add_executable(engine_demo src/demo.cpp)
target_link_libraries(engine_demo PRIVATE nanovllm_engine)

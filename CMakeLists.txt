cmake_minimum_required(VERSION 3.6)
project(tp2_metodos CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -Wno-long-long -pedantic -Wno-variadic-macros")

set(SOURCE_FILES
        src/Matrix.cpp
        src/empty_file.cpp
        src/Pca.cpp
        src/Dataset.cpp
        src/ppmloader.cpp
		src/kNN.cpp
		src/Metrics.cpp data/pca_cuantitative.cpp)

add_library(metodos-tp2-base ${SOURCE_FILES})

add_executable(main main.cpp)
target_link_libraries(main metodos-tp2-base)

add_subdirectory(test)

add_subdirectory(data)
#Cosas importantes:
#   1) Google test genera su propio main, dejar main.cpp por afuera de SOURCE_FILES para evitar problemas
#   2) Al compilar "main", agregar el archivo main.cpp para que lo ejecute

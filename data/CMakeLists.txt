set(DATA_FILES pca_cuantitative.cpp)

add_executable(correrExp ${DATA_FILES})

target_link_libraries(correrExp metodos-tp2-base)
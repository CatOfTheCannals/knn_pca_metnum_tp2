set(DATA_FILES
        knn_cuantitative.cpp
        pca_knn_cuantitative.cpp
        )

add_library(exp_library ${DATA_FILES})

add_executable(correrExp correrExp.cpp)

target_link_libraries(correrExp exp_library metodos-tp2-base)
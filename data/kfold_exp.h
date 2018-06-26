#ifndef TP2_METODOS_KFOLD_KNN_H
#define TP2_METODOS_KFOLD_KNN_H

#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <tuple>

#include "../src/Dataset.h"

void run_diff_kfold_knn();
void kfold_knn(int neighbours);
void run_diff_kfold_knn_pca();
void kfold_knn_with_pca(int neighbours, int alpha);

#endif //TP2_METODOS_KFOLD_KNN_H

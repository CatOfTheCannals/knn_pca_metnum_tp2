#include "pca_cuantitative.h"
#include "knn_cuantitative.h"
#include "pca_knn_cuantitative.h"
#include "kfold_exp.h"

int main(){
    std::cout << "Starting exp..." << std::endl << std::endl;
    // pca_cuantitative();
    // knn_cuantitative();
    // pca_knn_cuantitative();
    // kfold_knn(5);
    // kfold_knn_with_pca(5, 15);
    run_diff_kfold_knn_pca();
    std::cout << std::endl << "Finished exp." << std::endl;
}

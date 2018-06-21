#include "pca_cuantitative.h"
#include "knn_cuantitative.h"
#include "pca_knn_cuantitative.h"

int main(){
    std::cout << "Starting exp..." << std::endl << std::endl;
    // pca_cuantitative();
    // knn_cuantitative();
    pca_knn_cuantitative();
    std::cout << std::endl << "Finished exp." << std::endl;
}
